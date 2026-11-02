from dataclasses import dataclass
import cv2
import numpy as np

from .depth import DepthEngine, DepthFrame, depth_colors, scene_color_limits, measure


@dataclass
class PrecisionSnapshot:
    frame: DepthFrame
    frames: list
    raw: tuple
    calibration: object
    source_calibration: object

    def length(self, first, second):
        a, b = measure(self.frame, *first, radius=2), measure(self.frame, *second, radius=2)
        if a is None or b is None:
            raise ValueError("На одной из точек нет принятой глубины. Увеличьте изображение и выберите резкие детали.")
        samples = []
        for frame in self.frames:
            x, y = measure(frame, *first, radius=2), measure(frame, *second, radius=2)
            if x is not None and y is not None:
                samples.append(float(np.linalg.norm(np.array(x["point"]) - y["point"])) * 1000)
        length = float(np.linalg.norm(np.array(a["point"]) - b["point"])) * 1000
        repeatability = float(np.std(samples, ddof=1)) if len(samples) >= 3 else None
        return {"length_mm": length, "repeatability_mm": repeatability, "samples": len(samples),
                "first": list(first), "second": list(second)}


def verify_stationary(first, last):
    size = (640, round(first.shape[0] * 640 / first.shape[1]))
    a, b = [cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2GRAY) for image in (first, last)]
    features = cv2.goodFeaturesToTrack(a, 200, .02, 12)
    if features is None or len(features) < 20:
        raise ValueError("Недостаточно деталей для проверки неподвижности. Добавьте свет или видимый фон с рисунком.")
    current, status, _ = cv2.calcOpticalFlowPyrLK(a, b, features, None)
    if current is None:
        raise ValueError("Не удалось проверить неподвижность снимка.")
    flow = np.linalg.norm(current - features, axis=2).ravel()[status.ravel() == 1]
    if len(flow) < 20 or np.median(flow) > .25 or np.percentile(flow, 90) > .7:
        raise ValueError("Во время точного снимка было движение. Остановите камеру и растение и повторите.")


def make_precision_snapshot(calibration, pairs, wide_view=True, progress=None):
    if len(pairs) < 5:
        raise ValueError("Для точного снимка нужны 5 отдельных стереопар.")
    for pair in pairs[1:]:
        verify_stationary(pairs[0][0], pair[0])
        verify_stationary(pairs[0][1], pair[1])
    engine = DepthEngine(calibration, max_depth=3., disparities=128, quality="high",
                         processing_width=None, temporal=False, near_priority=True,
                         wide_view=wide_view, detail=True,estimate_surfaces=False,parallel=True)
    frames = []
    for i, pair in enumerate(pairs):
        if progress:
            progress(f"Подробный снимок: расчёт {i + 1} из {len(pairs)} в полном разрешении…")
        frame = engine.compute(*pair)
        if frame.quality_warning:
            raise ValueError(frame.quality_warning)
        frames.append(frame)
    values = np.stack([frame.disparity for frame in frames])
    masked = np.ma.masked_invalid(values)
    disparity = np.ma.median(masked, axis=0).filled(np.nan).astype(np.float32)
    spread = np.ma.median(np.ma.abs(masked - disparity), axis=0).filled(np.inf)
    valid = np.isfinite(values).sum(axis=0) >= 4
    valid &= np.isfinite(disparity) & (spread <= .5)
    points = cv2.reprojectImageTo3D(disparity, engine.calibration.Q)
    valid &= np.isfinite(points).all(axis=2) & (points[:, :, 2] >= .05) & (points[:, :, 2] <= 3.)
    points[~valid], disparity[~valid] = np.nan, np.nan
    if valid.sum() < 100:
        raise ValueError("Слишком мало устойчивой глубины. Проверьте расстояние, резкость, свет и калибровку.")
    limits = scene_color_limits(points[:, :, 2], valid, 3.)
    frame = DepthFrame(frames[-1].left, frames[-1].right, disparity, points, valid,
                       depth_colors(points[:, :, 2], valid, *limits), color_limits=limits)
    return PrecisionSnapshot(frame, frames, pairs[-1], engine.calibration, calibration)
