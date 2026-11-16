"""Find pigs in a picture and trace their outlines, with models downloaded on request.

D-FINE trained on Objects365 knows pigs, which the people detector does not. SlimSAM, shared with
the people tab, traces every found animal from one pass over the picture.
"""
import os
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from .body_measure import fill_holes
from .people_detection import WEIGHTS_FILE, ModelSource, SEGMENTER, model_path

DETECTOR = ModelSource("dfine-medium-obj365", "ustc-community/dfine-medium-obj365",
                       "f0fa3695138f3886d0da3f8dd17697e2ee8fc00f",
                       80462640, "0612b5b23f1438eae4d0eaeca07e0efef0b9e44cb23f14610eb7a563144094e8")
SOURCES = (DETECTOR, SEGMENTER)
PIG_LABEL = "Pig"
# Seen from above, a pig is often taken for another four-legged animal. On this tab they are all pigs.
LOOKALIKE_LABELS = ("Sheep", "Cow", "Dog", "Bear", "Horse", "Donkey", "Yak", "Elephant", "Antelope", "Deer", "Cat")
PIG_THRESHOLD = .3
LOOKALIKE_THRESHOLD = .45
MAX_OVERLAP = .6
MIN_BOX_PX = 24
MAX_ANIMALS = 40


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


class PigDetector:
    def __init__(self, root, status=lambda message: None):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from transformers import DFineForObjectDetection, RTDetrImageProcessor, SamImageProcessor, SamModel, SamProcessor
        self.torch = torch
        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        self.device_label = torch.cuda.get_device_name(0) if cuda else "процессор"
        if not cuda:
            torch.set_num_threads(max(2, (os.cpu_count() or 4) - 2))
        options = dict(local_files_only=True)
        status("Загружаю модели поиска свиней…")
        detector, segmenter = model_path(root, DETECTOR), model_path(root, SEGMENTER)
        self.detector_processor = RTDetrImageProcessor.from_pretrained(detector, **options)
        self.detector = DFineForObjectDetection.from_pretrained(detector, use_safetensors=True, **options).eval().to(self.device)
        self.segment_images = SamImageProcessor.from_pretrained(segmenter, **options)
        self.segment_prompts = SamProcessor(self.segment_images)
        self.segmenter = SamModel.from_pretrained(segmenter, use_safetensors=True, **options).eval().to(self.device)
        names = self.detector.config.label2id
        self.thresholds = {names[PIG_LABEL]: PIG_THRESHOLD}
        self.thresholds.update({names[name]: LOOKALIKE_THRESHOLD for name in LOOKALIKE_LABELS if name in names})

    def detect(self, bgr) -> tuple:
        """Boxes (x0, y0, x1, y1) and scores of the animals, best first."""
        torch = self.torch
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inputs = self.detector_processor(images=image, return_tensors="pt").to(self.device)
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
        """Exact body pixels for every box, from one pass of the outline model over the picture."""
        if not len(boxes):
            return []
        torch = self.torch
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.segment_prompts(image, input_boxes=[[[float(value) for value in box] for box in boxes]],
                                      return_tensors="pt")
        with torch.no_grad():
            embedding = self.segmenter.get_image_embeddings(inputs["pixel_values"].to(self.device, torch.float32))
            output = self.segmenter(image_embeddings=embedding,
                                    input_boxes=inputs["input_boxes"].to(self.device, torch.float32),
                                    multimask_output=False)
        masks = self.segment_images.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"],
                                                       inputs["reshaped_input_sizes"])[0]
        return [fill_holes(mask[0].numpy().astype(bool)) for mask in masks]
