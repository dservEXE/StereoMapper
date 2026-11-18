"""Find pigs in a picture, trace their outlines and estimate the distance to every pixel.

D-FINE trained on Objects365 knows pigs, which the people detector does not. SlimSAM, shared with
the people tab, traces every found animal from one pass over the picture. Depth Anything gives
metres from the camera for every pixel of an ordinary photograph. All are downloaded on request.
"""
import os
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from .body_measure import fill_holes
from .people_detection import WEIGHTS_FILE, ModelSource, SEGMENTER, model_path, select_device, timed

DETECTOR = ModelSource("dfine-medium-obj365", "ustc-community/dfine-medium-obj365",
                       "f0fa3695138f3886d0da3f8dd17697e2ee8fc00f",
                       80462640, "0612b5b23f1438eae4d0eaeca07e0efef0b9e44cb23f14610eb7a563144094e8")
DEPTH = ModelSource("depth-anything-v2-metric-indoor-small", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                    "8078d68a9c75a972131914f6afd0c1723be0da7f",
                    99173660, "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3")
SOURCES = (DETECTOR, SEGMENTER, DEPTH)
PIG_LABEL = "Pig"
# Seen from above, a pig is often taken for another four-legged animal. On this tab they are all pigs.
LOOKALIKE_LABELS = ("Sheep", "Cow", "Dog", "Bear", "Horse", "Donkey", "Yak", "Elephant", "Antelope", "Deer", "Cat")
PIG_THRESHOLD = .4
LOOKALIKE_THRESHOLD = .5
MAX_OVERLAP = .6
MIN_BOX_PX = 24
MAX_ANIMALS = 40
# An outline lying this much inside another is the same animal found twice, or a part of it.
SAME_ANIMAL_SHARE = .6
# An outline that holds two or more others of at least this share of its size has swallowed several animals.
MERGED_PART_SHARE = .2
MIN_OWN_SHARE = .5
TRIAL_SIZE = 256
# D-FINE samples the picture the way RT-DETR does, which kills the whole process on DirectML.
DIRECTML_UNSAFE = ("detector",)


def missing_size_mb(root) -> int:
    """How much is left to download; the outline model may already be here for the people tab."""
    missing = [source for source in SOURCES
               if not (model_path(root, source) / WEIGHTS_FILE).is_file()
               or (model_path(root, source) / WEIGHTS_FILE).stat().st_size != source.weights_size]
    return round(sum(source.weights_size for source in missing) / 1e6)


def overlap(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    shared = max(0., x1 - x0) * max(0., y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union if union > 0 else 0.


def keep_distinct(boxes, scores, limit=MAX_OVERLAP) -> list:
    """Indices of the best box for each animal: one animal may be found under several names."""
    kept = []
    for index in np.argsort(-np.asarray(scores)):
        if all(overlap(boxes[index], boxes[other]) <= limit for other in kept):
            kept.append(int(index))
    return kept


def distinct_outlines(masks, scores) -> list:
    """(index, outline) for every real animal, so that none is counted twice.

    A box around a head or a leg gives an outline inside the outline of the whole animal; a box
    around two animals gives one outline over both. Both kinds are dropped, and every remaining
    pixel is given to one animal only.
    """
    areas = [int(mask.sum()) for mask in masks]
    live = [index for index, area in enumerate(areas) if area]

    def inside(inner, outer) -> bool:
        return (masks[inner] & masks[outer]).sum() > SAME_ANIMAL_SHARE * areas[inner]
    merged = {outer for outer in live
              if sum(inner != outer and MERGED_PART_SHARE * areas[outer] <= areas[inner] < areas[outer] and inside(inner, outer)
                     for inner in live) >= 2}
    kept = []
    for index in sorted((index for index in live if index not in merged), key=lambda index: -areas[index]):
        if not any(inside(index, other) for other in kept):
            kept.append(index)
    taken = np.zeros(masks[0].shape, bool) if masks else None
    result = []
    for index in sorted(kept, key=lambda index: -scores[index]):
        own = masks[index] & ~taken
        if own.sum() >= MIN_OWN_SHARE * areas[index]:
            taken |= own
            result.append((index, own))
    return sorted(result, key=lambda item: item[0])


class PigDetector:
    """D-FINE finds the animals, SlimSAM traces them, Depth Anything gives the distances.

    Each model is tried on the graphics card at load; one that fails there, or on DirectML is
    no faster than the processor, stays on the processor while the others keep the card.
    """

    def __init__(self, root, status=lambda message: None):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from transformers import (AutoImageProcessor, AutoModelForDepthEstimation, DFineForObjectDetection,
                                  RTDetrImageProcessor, SamImageProcessor, SamModel, SamProcessor)
        self.torch = torch
        self.device, self.device_name = select_device(torch)
        # On CUDA the processor only prepares data. Otherwise it does the real work.
        torch.set_num_threads(2 if self.device == "cuda" else max(2, (os.cpu_count() or 4) - 2))
        options = dict(local_files_only=True)
        status("Загружаю модели поиска свиней…")
        detector, segmenter, depth = (model_path(root, source) for source in SOURCES)
        self.detector_processor = RTDetrImageProcessor.from_pretrained(detector, **options)
        self.detector = DFineForObjectDetection.from_pretrained(detector, use_safetensors=True, **options).eval()
        self.segment_images = SamImageProcessor.from_pretrained(segmenter, **options)
        self.segment_prompts = SamProcessor(self.segment_images)
        self.segmenter = SamModel.from_pretrained(segmenter, use_safetensors=True, **options).eval()
        self.depth_processor = AutoImageProcessor.from_pretrained(depth, use_fast=False, **options)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth, use_safetensors=True, **options).eval()
        self.devices = dict(detector="cpu", outline="cpu", depth="cpu")
        names = self.detector.config.label2id
        self.thresholds = {names[PIG_LABEL]: PIG_THRESHOLD}
        self.thresholds.update({names[name]: LOOKALIKE_THRESHOLD for name in LOOKALIKE_LABELS if name in names})
        if self.device != "cpu":
            status(f"Проверяю модели на устройстве: {self.device_name}…")
            self.place_models()

    def place_models(self) -> None:
        """Try every model on the card with a blank picture; a failing or slower one stays on the processor."""
        blank = np.zeros((TRIAL_SIZE, TRIAL_SIZE, 3), np.uint8)
        box = np.array([[TRIAL_SIZE * .25, TRIAL_SIZE * .25, TRIAL_SIZE * .75, TRIAL_SIZE * .75]], np.float32)
        trials = (("detector", self.detector, lambda: self.detect(blank)),
                  ("outline", self.segmenter, lambda: self.outlines(blank, box)),
                  ("depth", self.depth_model, lambda: self.depth(blank)))
        direct_ml = self.device != "cuda"
        for name, model, trial in trials:
            if direct_ml and name in DIRECTML_UNSAFE:
                continue
            try:
                # A weak card can be slower than the processor, so measure rather than assume.
                processor_time = timed(trial) if direct_ml else None
                model.to(self.device)
                self.devices[name] = self.device
                trial()
                if processor_time is not None and timed(trial) >= processor_time:
                    raise RuntimeError("slower than the processor")
            except Exception:  # noqa: BLE001 - backends raise anything for an unsupported operator.
                self.devices[name] = "cpu"
                model.to("cpu")

    @property
    def on_card(self) -> list:
        return [name for name, device in self.devices.items() if device != "cpu"]

    @property
    def device_label(self) -> str:
        """For the status line: where the models actually run."""
        if self.device == "cpu":
            return "процессор"
        if not self.on_card:
            return f"процессор ({self.device_name} не подошла)"
        return self.device_name + ("" if len(self.on_card) == len(self.devices) else " и процессор")

    def detect(self, bgr) -> tuple:
        """Boxes (x0, y0, x1, y1) and scores of the animals, best first."""
        torch = self.torch
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inputs = self.detector_processor(images=image, return_tensors="pt").to(self.devices["detector"])
            output = self.detector(**inputs)
            found = self.detector_processor.post_process_object_detection(
                SimpleNamespace(logits=output.logits.cpu(), pred_boxes=output.pred_boxes.cpu()),
                target_sizes=torch.tensor([bgr.shape[:2]]), threshold=min(self.thresholds.values()))[0]
        boxes, scores, labels = (found[key].numpy() for key in ("boxes", "scores", "labels"))
        wanted = np.array([score >= self.thresholds.get(int(label), 2.) for score, label in zip(scores, labels)], bool)
        large = (boxes[:, 2] - boxes[:, 0] >= MIN_BOX_PX) & (boxes[:, 3] - boxes[:, 1] >= MIN_BOX_PX)
        boxes, scores = boxes[wanted & large], scores[wanted & large]
        kept = keep_distinct(boxes, scores)[:MAX_ANIMALS]
        return boxes[kept], scores[kept]

    def outlines(self, bgr, boxes) -> list:
        """Body pixels for every box, from one pass of the outline model over the picture."""
        if not len(boxes):
            return []
        torch = self.torch
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.segment_prompts(image, input_boxes=[[[float(value) for value in box] for box in boxes]],
                                      return_tensors="pt")
        device = self.devices["outline"]
        # The processor gives 64-bit coordinates, which DirectML does not have.
        with torch.no_grad():
            embedding = self.segmenter.get_image_embeddings(inputs["pixel_values"].to(device, torch.float32))
            output = self.segmenter(image_embeddings=embedding,
                                    input_boxes=inputs["input_boxes"].to(device, torch.float32),
                                    multimask_output=False)
        masks = self.segment_images.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"],
                                                       inputs["reshaped_input_sizes"])[0]
        return [fill_holes(mask[0].numpy().astype(bool)) for mask in masks]

    def depth(self, bgr) -> np.ndarray:
        """Estimated metres from the camera for every pixel of an ordinary picture."""
        torch = self.torch
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inputs = self.depth_processor(images=image, return_tensors="pt").to(self.devices["depth"])
            predicted = self.depth_model(**inputs).predicted_depth
            full = torch.nn.functional.interpolate(predicted.unsqueeze(1).float(), size=bgr.shape[:2], mode="bilinear",
                                                   align_corners=False)[0, 0]
        return full.cpu().numpy().astype(np.float32)
