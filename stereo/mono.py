"""Local relative depth from one photograph; never interpreted as metres."""
from pathlib import Path
import os
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image

from .depth import DepthFrame


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


def model_directory(root, status=lambda message: None):
    directory = Path(root) / "models" / "depth-anything-v2-small"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        target = directory / name
        minimum = 90_000_000 if name.endswith("safetensors") else 100
        if target.exists() and target.stat().st_size > minimum:
            continue
        status("Скачиваю модель глубины для локальной обработки (около 100 МБ)…")
        kind = "resolve" if name.endswith("safetensors") else "raw"
        url = f"https://huggingface.co/{MODEL_ID}/{kind}/{MODEL_REVISION}/{name}"
        temporary = directory / (name + ".part")
        last_error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "StereoMapper/2"})
                with urllib.request.urlopen(request, timeout=45) as source, temporary.open("wb") as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
                if temporary.stat().st_size < minimum:
                    raise ValueError("Модель загрузилась не полностью.")
                temporary.replace(target)
                last_error = None
                break
            except (OSError, ValueError) as error:
                last_error = error
                time.sleep(.5 * (attempt + 1))
        if last_error:
            raise RuntimeError(f"Не удалось скачать {name}. Проверьте интернет и повторите. {last_error}")
    return directory


class MonoDepth:
    def __init__(self, root, status=lambda message: None):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError("Для одного фото запустите setup_photo.ps1: нужны PyTorch и Transformers.") from error
        self.torch = torch
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        directory = model_directory(root, status)
        status("Загружаю модель оценки глубины…")
        self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True,
                                                            use_fast=False, trust_remote_code=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            directory, local_files_only=True, use_safetensors=True, trust_remote_code=False).eval()

    def compute(self, bgr):
        h, w = bgr.shape[:2]
        if w > 1600:
            bgr = cv2.resize(bgr, (1600, round(h * 1600 / w)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            output = self.torch.nn.functional.interpolate(output.unsqueeze(1), size=(h, w),
                                                          mode="bicubic", align_corners=False)[0, 0]
        inverse = output.cpu().numpy().astype(np.float32)
        return relative_frame(bgr, inverse), inverse


def relative_frame(image, inverse):
    valid = np.isfinite(inverse)
    if not valid.any():
        raise ValueError("Модель не вернула глубину.")
    low, high = np.percentile(inverse[valid], [2, 98])
    score = np.clip((inverse - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)
    score = cv2.bilateralFilter(score, 5, .08, 3)
    relative_z = 1 / (.15 + score * .85)
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w))
    # Assumed projection for an illustrative 2.5D surface; scale is arbitrary.
    focal = w * .9
    points = np.stack(((xx - w / 2) / focal * relative_z,
                       (yy - h / 2) / focal * relative_z, relative_z), axis=2).astype(np.float32)
    points[~valid] = np.nan
    color = cv2.applyColorMap(np.uint8((1 - score) * 255), cv2.COLORMAP_TURBO)
    color[~valid] = (18, 22, 30)
    return DepthFrame(image.copy(), image.copy(), inverse.copy(), points, valid, color,
                      units="relative", method="Depth-Anything-V2-Small")
