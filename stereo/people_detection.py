"""Find people, their joints and outlines with local pretrained models, downloaded once on request.

All three models are Apache-2.0 and need no packages beyond the bundled torch and transformers.
"""
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.request

import cv2
import numpy as np
from PIL import Image

from .body_measure import KEYPOINT_COUNT, MIN_KEYPOINT_SCORE, PersonObservation, fill_holes
from .gpu_pack import is_integrated

DOWNLOAD_SIZE_MB = 252
DETECTION_THRESHOLD = .5
MIN_BOX_PX = 24
COCO_DATASET_INDEX = 0
# Joints that lie inside the body, as hints for the outline. Wrists and the face are left out
# because a pose model places them less surely when the person turns away.
OUTLINE_HINTS = (5, 6, 7, 8, 11, 12, 13, 14, 15, 16)
TRIAL_SIZE = 256
# Measured on NVIDIA and AMD alike: RT-DETR kills the whole process on DirectML with a fatal
# native check, which no exception handler can catch, so it is never tried there.
DIRECTML_UNSAFE = ("detector",)
POSE_BOX_PADDING = 1.25
HEATMAP_BLUR = (5, 5)


@dataclass(frozen=True)
class ModelSource:
    folder: str
    repository: str
    revision: str
    weights_size: int
    weights_sha256: str

    def url(self, name: str) -> str:
        kind = "resolve" if name.endswith("safetensors") else "raw"
        return f"https://huggingface.co/{self.repository}/{kind}/{self.revision}/{name}"


DETECTOR = ModelSource("rtdetr-r18vd", "PekingU/rtdetr_r18vd", "ac77a11ff0170a41b771c03264987f8ce2b0d753",
                       80904152, "fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093")
POSE = ModelSource("vitpose-plus-small", "usyd-community/vitpose-plus-small",
                   "0c30b6534bb621af0162b481176742577264e36e",
                   132619932, "f7bad8ed09eeeb2a7de6b38faaa8a88d07838e23e9c06a2a782099bca7467cb9")
SEGMENTER = ModelSource("slimsam-uniform-77", "Zigeng/SlimSAM-uniform-77", "79c09c1ce6b4ae51f00634ed171d9b8e888f6911",
                        38931272, "b0ee221e3d33935807a820c157fec81a537b3b9979edc88b148a8dae8bbdd5d2")
SOURCES = (DETECTOR, POSE, SEGMENTER)
SETTINGS_FILES = ("config.json", "preprocessor_config.json")
WEIGHTS_FILE = "model.safetensors"


def model_path(root, source) -> Path:
    return Path(root) / "models" / source.folder


def models_ready(root) -> bool:
    for source in SOURCES:
        directory = model_path(root, source)
        if not all((directory / name).is_file() for name in SETTINGS_FILES):
            return False
        weights = directory / WEIGHTS_FILE
        if not weights.is_file() or weights.stat().st_size != source.weights_size:
            return False
    return True


def fetch(url, target, opener=urllib.request.urlopen) -> str:
    """Download to a temporary name, returning the SHA-256; only verified files are renamed."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "StereoMapper-Models"})
    received = 0
    with opener(request, timeout=45) as response, target.open("wb") as output:
        headers = getattr(response, "headers", None)
        expected = headers.get("Content-Length") if headers is not None else None
        while block := response.read(1024 * 1024):
            digest.update(block)
            output.write(block)
            received += len(block)
    # A dropped connection ends the read without an error, and would pass for a complete file.
    if expected is not None and received != int(expected):
        raise ValueError("связь прервалась")
    return digest.hexdigest()


def download_models(root, status=lambda message: None, opener=urllib.request.urlopen) -> None:
    for source in SOURCES:
        directory = model_path(root, source)
        directory.mkdir(parents=True, exist_ok=True)
        for name in (*SETTINGS_FILES, WEIGHTS_FILE):
            final = directory / name
            weights = name == WEIGHTS_FILE
            if final.is_file() and (not weights or final.stat().st_size == source.weights_size):
                continue
            status(f"Скачиваю {source.repository}: {name}…")
            partial = directory / (name + ".part")
            try:
                digest = fetch(source.url(name), partial, opener)
                if weights and digest != source.weights_sha256:
                    raise ValueError("контрольная сумма не совпала")
                if not weights:
                    json.loads(partial.read_text(encoding="utf-8"))
                partial.replace(final)
            except (OSError, ValueError, http.client.HTTPException) as error:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Не удалось скачать модель {source.repository}: {error}") from error


def pose_crop(box, width, height):
    """Crop for one xywh box at the model's aspect: (centre x, centre y, crop width, crop height)."""
    x, y, box_width, box_height = box
    aspect = width / height
    if box_width > aspect * box_height:
        box_height = box_width / aspect
    else:
        box_width = box_height * aspect
    return x + box[2] / 2, y + box[3] / 2, box_width * POSE_BOX_PADDING, box_height * POSE_BOX_PADDING


def crop_matrix(crop, width, height) -> np.ndarray:
    """Image to model input, in the unbiased form the pose model was trained with."""
    center_x, center_y, crop_width, crop_height = crop
    scale_x, scale_y = (width - 1) / crop_width, (height - 1) / crop_height
    return np.array(((scale_x, 0, (crop_width / 2 - center_x) * scale_x),
                     (0, scale_y, (crop_height / 2 - center_y) * scale_y)), np.float64)


def peak(values, index) -> float:
    """Sub-pixel offset of a maximum, from a parabola through it and its two neighbours."""
    if not 0 < index < len(values) - 1:
        return 0.
    left, middle, right = values[index - 1], values[index], values[index + 1]
    curvature = 2 * middle - left - right
    return float(np.clip((right - left) / (2 * curvature), -.5, .5)) if curvature > 0 else 0.


def decode_heatmaps(heatmaps, crop) -> tuple:
    """Joint positions in the image and their confidences, from one person's heatmaps."""
    center_x, center_y, crop_width, crop_height = crop
    count, height, width = heatmaps.shape
    keypoints = np.zeros((count, 2), np.float32)
    scores = np.zeros(count, np.float32)
    for joint, heatmap in enumerate(heatmaps):
        scores[joint] = float(heatmap.max())
        smooth = cv2.GaussianBlur(heatmap, HEATMAP_BLUR, 0)
        row, column = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
        x = column + peak(smooth[row], column)
        y = row + peak(smooth[:, column], row)
        keypoints[joint] = (center_x - crop_width / 2 + x * crop_width / (width - 1),
                            center_y - crop_height / 2 + y * crop_height / (height - 1))
    return keypoints, scores


def outline_hints(observation) -> list:
    """Points known to be on the body, including across the clothes between the joints.

    With hints only on joints the outline model took a dark top for a separate object, and
    the chest went missing from the volume.
    """
    points, scores = observation.keypoints, observation.keypoint_scores
    seen = scores >= MIN_KEYPOINT_SCORE
    hints = [points[i] for i in OUTLINE_HINTS if seen[i]]
    if seen[5] and seen[6] and seen[11] and seen[12]:
        neck, pelvis = (points[5] + points[6]) / 2, (points[11] + points[12]) / 2
        hints += [neck + (pelvis - neck) * share for share in (.2, .4, .6, .8)]
    for hip, knee in ((11, 13), (12, 14)):
        if seen[hip] and seen[knee]:
            hints.append((points[hip] + points[knee]) / 2)
    return [[float(x), float(y)] for x, y in hints]


def timed(action) -> float:
    started = time.perf_counter()
    action()
    return time.perf_counter() - started


def select_device(torch):
    """The fastest thing available to run models on, and its name for the interface."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    try:
        import torch_directml
    except ImportError:
        return "cpu", "процессор"
    names = [torch_directml.device_name(index).strip("\0 ") for index in range(torch_directml.device_count())]
    if not names:
        return "cpu", "процессор"
    # Windows may list a weak integrated adapter first.
    discrete = [index for index, name in enumerate(names) if not is_integrated(name)]
    index = discrete[0] if discrete else 0
    return torch_directml.device(index), names[index]


class PersonDetector:
    """RT-DETR finds people, ViTPose places 17 joints, SlimSAM traces the outline.

    A graphics card may not run every model, DirectML especially, so each is tried at load and
    the ones that fail stay on the processor while the others keep the card.
    """

    def __init__(self, root, status=lambda message: None):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from transformers import (RTDetrForObjectDetection, RTDetrImageProcessor, SamImageProcessor, SamModel,
                                  SamProcessor, VitPoseForPoseEstimation, VitPoseImageProcessor)
        self.torch = torch
        self.device, self.device_name = select_device(torch)
        # On CUDA the processor only prepares data. Otherwise it does real work, leaving room for capture.
        torch.set_num_threads(2 if self.device == "cuda" else max(2, (os.cpu_count() or 4) - 2))
        self.devices = dict(detector=self.device, pose=self.device, outline=self.device)
        self.fallbacks = []
        options = dict(local_files_only=True)
        detector, pose = model_path(root, DETECTOR), model_path(root, POSE)
        status("Загружаю модели поиска людей…")
        self.detector_processor = RTDetrImageProcessor.from_pretrained(detector, **options)
        self.detector = RTDetrForObjectDetection.from_pretrained(detector, use_safetensors=True, **options).eval()
        # Only its settings: its own image steps need SciPy, which the application does not ship.
        self.pose_processor = VitPoseImageProcessor.from_pretrained(pose, **options)
        self.pose = VitPoseForPoseEstimation.from_pretrained(pose, use_safetensors=True, **options).eval()
        segmenter = model_path(root, SEGMENTER)
        self.segment_images = SamImageProcessor.from_pretrained(segmenter, **options)
        self.segment_prompts = SamProcessor(self.segment_images)
        self.segmenter = SamModel.from_pretrained(segmenter, use_safetensors=True, **options).eval()
        self.person_label = self.detector.config.label2id.get("person", 0)
        self.pose_width = int(self.pose_processor.size["width"])
        self.pose_height = int(self.pose_processor.size["height"])
        self.pose_mean = np.array(self.pose_processor.image_mean, np.float32)
        self.pose_std = np.array(self.pose_processor.image_std, np.float32)
        if self.device != "cpu":
            status(f"Проверяю модели на устройстве: {self.device_name}…")
            self.place_models()

    def place_models(self) -> None:
        """Try every model on the card with a blank image; a failing one returns to the processor."""
        blank = np.zeros((TRIAL_SIZE, TRIAL_SIZE, 3), np.uint8)
        rgb = Image.fromarray(blank)
        box = np.array([[TRIAL_SIZE * .25, TRIAL_SIZE * .1, TRIAL_SIZE * .75, TRIAL_SIZE * .9]], np.float32)
        points = np.zeros((KEYPOINT_COUNT, 2), np.float32) + TRIAL_SIZE / 2
        scores = np.ones(KEYPOINT_COUNT, np.float32)
        observation = PersonObservation(tuple(float(value) for value in box[0]), 1., points, scores)
        trials = (("detector", self.detector, lambda: self.find_boxes(rgb, blank.shape[:2])),
                  ("pose", self.pose, lambda: self.find_joints(rgb, box)),
                  ("outline", self.segmenter, lambda: self.outline(blank, observation)))
        direct_ml = self.device != "cuda"
        for name, model, trial in trials:
            if direct_ml and name in DIRECTML_UNSAFE:
                self.devices[name] = "cpu"
                continue
            try:
                # A weak integrated card can be slower than the processor, so measure rather than assume.
                processor_time = None
                if direct_ml:
                    self.devices[name] = "cpu"
                    processor_time = timed(trial)
                    self.devices[name] = self.device
                model.to(self.device)
                trial()
                if processor_time is not None and timed(trial) >= processor_time:
                    raise RuntimeError("slower than the processor")
            except Exception:  # noqa: BLE001 - backends raise anything for an unsupported operator.
                self.devices[name] = "cpu"
                model.to("cpu")
        self.fallbacks = [name for name, device in self.devices.items() if device == "cpu"]

    @property
    def device_label(self) -> str:
        """For the status line: where the models actually run."""
        if self.device == "cpu":
            return "процессор"
        if len(self.fallbacks) == len(self.devices):
            return f"процессор ({self.device_name} не быстрее)"
        return self.device_name + (" и процессор" if self.fallbacks else "")

    def outline(self, bgr, observation) -> np.ndarray | None:
        """Exact body pixels for one person, from their box and joints."""
        hints = outline_hints(observation)
        if len(hints) < 2:
            return None
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.segment_prompts(image, input_boxes=[[list(observation.box)]], input_points=[[hints]],
                                      input_labels=[[[1] * len(hints)]], return_tensors="pt")
        device = self.devices["outline"]
        floats = self.torch.float32
        # The processor gives 64-bit coordinates, which DirectML does not have.
        with self.torch.no_grad():
            output = self.segmenter(pixel_values=inputs["pixel_values"].to(device, floats),
                                    input_boxes=inputs["input_boxes"].to(device, floats),
                                    input_points=inputs["input_points"].to(device, floats),
                                    input_labels=inputs["input_labels"].to(device), multimask_output=False)
        masks = self.segment_images.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"],
                                                       inputs["reshaped_input_sizes"])
        return fill_holes(masks[0][0, 0].numpy().astype(bool))

    def find_boxes(self, image, shape) -> tuple:
        torch = self.torch
        device = self.devices["detector"]
        with torch.no_grad():
            inputs = self.detector_processor(images=image, return_tensors="pt").to(device)
            output = self.detector(**inputs)
            # Post-processing runs here on the processor: it needs no card and not every card has its operators.
            found = self.detector_processor.post_process_object_detection(
                SimpleNamespace(logits=output.logits.cpu(), pred_boxes=output.pred_boxes.cpu()),
                target_sizes=torch.tensor([shape]), threshold=DETECTION_THRESHOLD)[0]
        keep = found["labels"] == self.person_label
        boxes, scores = found["boxes"][keep].numpy(), found["scores"][keep].numpy()
        large = (boxes[:, 2] - boxes[:, 0] >= MIN_BOX_PX) & (boxes[:, 3] - boxes[:, 1] >= MIN_BOX_PX)
        return boxes[large], scores[large]

    def find_joints(self, image, boxes) -> tuple:
        torch = self.torch
        device = self.devices["pose"]
        crops = [pose_crop((x0, y0, x1 - x0, y1 - y0), self.pose_width, self.pose_height) for x0, y0, x1, y1 in boxes]
        rgb = np.asarray(image, np.float32) / 255
        batch = np.stack([((cv2.warpAffine(rgb, crop_matrix(crop, self.pose_width, self.pose_height),
                                           (self.pose_width, self.pose_height), flags=cv2.INTER_LINEAR)
                            - self.pose_mean) / self.pose_std).transpose(2, 0, 1) for crop in crops])
        with torch.no_grad():
            heatmaps = self.pose(pixel_values=torch.from_numpy(batch).to(device),
                                 dataset_index=torch.full((len(crops),), COCO_DATASET_INDEX, dtype=torch.long).to(device)
                                 ).heatmaps.cpu().numpy()
        return crops, heatmaps

    def detect(self, bgr) -> list:
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes, scores = self.find_boxes(image, bgr.shape[:2])
        if not len(boxes):
            return []
        crops, heatmaps = self.find_joints(image, boxes)
        results = []
        for box, score, crop, heatmap in zip(boxes, scores, crops, heatmaps):
            if len(heatmap) != KEYPOINT_COUNT:
                continue
            keypoints, confidence = decode_heatmaps(heatmap, crop)
            results.append(PersonObservation(tuple(float(v) for v in box), float(score), keypoints, confidence))
        return results
