"""Measure a person from an outline, metric depth and joint positions, independent of the camera.

The outline gives the shape and depth gives the scale. A body part is modelled as an elliptic
cylinder, so its front and side widths give its volume, and volume gives mass. Turning on the
spot is what provides the side widths. The result is an estimate that has not been validated
against scales; WeightEstimate carries how much of it was observed rather than assumed.
"""
from dataclasses import dataclass, field
import math
from typing import NamedTuple

import cv2
import numpy as np

# COCO keypoints, as returned by the pose model.
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
KEYPOINT_COUNT = 17
LIMBS = {"upper_arm": ((L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW)),
         "forearm": ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)),
         "thigh": ((L_HIP, L_KNEE), (R_HIP, R_KNEE)),
         "shin": ((L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE))}
SKELETON = ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
            *(pair for sides in LIMBS.values() for pair in sides))

MIN_KEYPOINT_SCORE = .35
MAX_GAP_PX = 2
DEPTH_EROSION_PX = 9
DEPTH_BIN_M = .04
MIN_DEPTH_PIXELS = 30
# Stereo distance to one person jumped by a metre between frames at 3 m, and every size scales
# with it. A person turning on the spot stays put, so the recent median is far steadier.
DEPTH_WINDOW = 15
SAMPLES_ALONG_PART = 7
# Part of a limb next to a joint is widened by the neighbouring part, so sample its middle.
PART_START, PART_END = .25, .75

# Ankle height is about 3.9% of stature (Drillis and Contini, 1966).
ANKLE_HEIGHT_FRACTION = .039
BODY_DENSITY_KG_M3 = 1020.
# Hands and feet are about 4% of body mass (Winter, Biomechanics) and are not measured here.
HANDS_FEET_FACTOR = 1.04
# Side width over front width, used only when turning has not shown the side.
ASSUMED_DEPTH_RATIO = {"torso": .62, "head": .85, "upper_arm": 1., "forearm": 1., "thigh": 1., "shin": 1.}
LIMB_COUNT = {"torso": 1, "head": 1, "upper_arm": 2, "forearm": 2, "thigh": 2, "shin": 2}

MIN_SAMPLES = 15
MIN_BIN_SAMPLES = 5
MAX_SAMPLES = 400
FRONT_MAX_DEG = 30.
SIDE_MIN_DEG = 60.
# Unvalidated model: never present a tighter range than this, whatever the sample spread.
MIN_RELATIVE_RANGE = .15


class PersonObservation(NamedTuple):
    box: tuple
    score: float
    keypoints: np.ndarray
    keypoint_scores: np.ndarray


class BodySample(NamedTuple):
    depth: float
    height: float | None
    shoulder_span: float
    widths: dict
    lengths: dict


@dataclass(frozen=True)
class WeightEstimate:
    samples: int
    height_m: float | None
    widths_m: dict
    side_widths_m: dict
    lengths_m: dict
    volume_l: float | None
    mass_kg: float | None
    mass_low_kg: float | None
    mass_high_kg: float | None
    observed_side: bool
    view_coverage: dict
    message: str


def run_length(mask, x, y, dx, dy, limit_px) -> float:
    """How far a straight walk stays on the outline, ignoring tiny gaps."""
    height, width = mask.shape
    gap, reached = 0, 0.
    for step in range(1, int(limit_px) + 1):
        px, py = int(round(x + dx * step)), int(round(y + dy * step))
        if not (0 <= px < width and 0 <= py < height):
            break
        if mask[py, px]:
            gap, reached = 0, float(step)
        else:
            gap += 1
            if gap > MAX_GAP_PX:
                break
    return reached


def part_width(mask, start, end, scale) -> float | None:
    """Median width across the middle of a body part, in metres.

    Twice the shorter side from the bone: legs held together or an arm against the
    body would otherwise count as one wide part.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 3:
        return None
    ux, uy = axis / length
    limit = max(8., length * 1.5)
    widths = []
    for t in np.linspace(PART_START, PART_END, SAMPLES_ALONG_PART):
        x, y = start + axis * t
        px, py = int(round(x)), int(round(y))
        if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px]:
            widths.append(2 * min(run_length(mask, x, y, -uy, ux, limit), run_length(mask, x, y, uy, -ux, limit)) + 1)
    return float(np.median(widths)) * scale if len(widths) >= 3 else None


def ray_point(u, v, depth, K):
    return np.array(((u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth))


def outline_depth(z, valid, mask) -> float | None:
    """Distance of the body: measured pixels well inside the outline, free of its edges."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((DEPTH_EROSION_PX, DEPTH_EROSION_PX), np.uint8)) > 0
    values = z[inner & valid]
    if len(values) < MIN_DEPTH_PIXELS:
        return None
    # The most common distance, not the median: stereo fills plain clothing with background,
    # which on a live recording pushed the median 10% too far. A body is one peak; that is not.
    edges = np.arange(values.min(), values.max() + DEPTH_BIN_M, DEPTH_BIN_M)
    if len(edges) < 3:
        return float(np.median(values))
    counts = np.convolve(np.histogram(values, edges)[0], np.ones(3), "same")
    peak = int(np.argmax(counts))
    near = values[(values >= edges[max(0, peak - 1)]) & (values < edges[min(len(edges) - 1, peak + 2)])]
    return float(np.median(near))


def measure_person(points, valid, K, observation, mask) -> BodySample | None:
    """One frame's measurements from the body's outline, or None if the torso is not reliably seen."""
    z = np.nan_to_num(points[:, :, 2]).astype(np.float32)
    valid = valid & (z > 0)
    kp, scores = observation.keypoints, observation.keypoint_scores
    seen = scores >= MIN_KEYPOINT_SCORE
    if mask is None or not (seen[L_SHOULDER] and seen[R_SHOULDER] and seen[L_HIP] and seen[R_HIP]):
        return None
    neck = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2
    pelvis = (kp[L_HIP] + kp[R_HIP]) / 2
    torso_axis = pelvis - neck
    torso_px = float(np.linalg.norm(torso_axis))
    if torso_px < 8:
        return None
    body_depth = outline_depth(z, valid, mask)
    if body_depth is None:
        return None
    focal = float(K[0, 0] + K[1, 1]) / 2
    scale = body_depth / focal
    widths, lengths = {}, {}
    width = part_width(mask, neck, pelvis, scale)
    if width is not None:
        widths["torso"] = width
    lengths["torso"] = torso_px * scale
    for name, sides in LIMBS.items():
        for side, (a, b) in zip("lr", sides):
            if not (seen[a] and seen[b]):
                continue
            width = part_width(mask, kp[a], kp[b], scale)
            if width is not None:
                widths[f"{name}_{side}"] = width
            lengths[f"{name}_{side}"] = float(np.linalg.norm(kp[a] - kp[b])) * scale
    ux, uy = -torso_axis / torso_px
    head_px = run_length(mask, *neck, ux, uy, torso_px)
    height = None
    if head_px >= 3:
        top = neck + np.array((ux, uy)) * head_px
        lengths["head"] = head_px * scale
        center = neck + np.array((ux, uy)) * head_px * .65
        widths["head"] = (run_length(mask, *center, -uy, ux, head_px) + run_length(mask, *center, uy, -ux, head_px) + 1) * scale
        ankles = [kp[i] for i in (L_ANKLE, R_ANKLE) if seen[i]]
        if ankles:
            ankle = ray_point(*np.mean(ankles, axis=0), body_depth, K)
            height = float(np.linalg.norm(ray_point(*top, body_depth, K) - ankle)) / (1 - ANKLE_HEIGHT_FRACTION)
    shoulder_span = float(np.linalg.norm(kp[L_SHOULDER] - kp[R_SHOULDER])) * scale
    return BodySample(body_depth, height, shoulder_span, widths, lengths)


def rescaled(sample: BodySample, depth: float) -> BodySample:
    """The same measurement at a steadier distance: every size is proportional to depth."""
    ratio = depth / sample.depth
    return BodySample(depth, sample.height * ratio if sample.height is not None else None,
                      sample.shoulder_span * ratio,
                      {key: value * ratio for key, value in sample.widths.items()},
                      {key: value * ratio for key, value in sample.lengths.items()})


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


@dataclass
class BodyAccumulator:
    """Combine one person's frames: front and side views need separating as they turn."""
    samples: list = field(default_factory=list)
    recent_depths: list = field(default_factory=list)

    def add(self, sample: BodySample) -> None:
        self.recent_depths = (self.recent_depths + [sample.depth])[-DEPTH_WINDOW:]
        self.samples.append(rescaled(sample, float(np.median(self.recent_depths))))
        if len(self.samples) > MAX_SAMPLES:
            del self.samples[0]

    def views(self) -> list:
        """Apparent turn of each sample, from how narrow the shoulders look."""
        widest = percentile([s.shoulder_span for s in self.samples], 95) or 1.
        return [math.degrees(math.acos(min(1., s.shoulder_span / max(widest, 1e-6)))) for s in self.samples]

    def part_values(self, part, selected) -> list:
        return [sample.widths[key] for sample in selected for key in (part, f"{part}_l", f"{part}_r")
                if key in sample.widths]

    def part_length(self, part) -> float | None:
        # Foreshortening only ever shortens a part, so favour the longer views.
        values = [sample.lengths[key] for sample in self.samples for key in (part, f"{part}_l", f"{part}_r")
                  if key in sample.lengths]
        return percentile(values, 80)

    def estimate(self) -> WeightEstimate:
        count = len(self.samples)
        views = self.views()
        front = [s for s, view in zip(self.samples, views) if view <= FRONT_MAX_DEG]
        side = [s for s, view in zip(self.samples, views) if view >= SIDE_MIN_DEG]
        coverage = dict(front=len(front), oblique=count - len(front) - len(side), side=len(side))
        heights = [s.height for s in self.samples if s.height is not None]
        height = percentile(heights, 50) if len(heights) >= MIN_BIN_SAMPLES else None
        if count < MIN_SAMPLES:
            return WeightEstimate(count, height, {}, {}, {}, None, None, None, None, False, coverage,
                                  f"Идёт сбор: {count} из {MIN_SAMPLES}")
        widths, side_widths, lengths = {}, {}, {}
        for part in ASSUMED_DEPTH_RATIO:
            front_values = self.part_values(part, front)
            values = front_values if len(front_values) >= MIN_BIN_SAMPLES else self.part_values(part, self.samples)
            width = percentile(values, 50)
            length = self.part_length(part)
            if width is None or length is None:
                continue
            widths[part], lengths[part] = width, length
            side_values = self.part_values(part, side)
            if len(side_values) >= MIN_BIN_SAMPLES:
                side_widths[part] = percentile(side_values, 50)
        needed = {"torso", "head", "thigh", "shin"}
        if not needed <= set(widths):
            missing = "ноги" if not {"thigh", "shin"} <= set(widths) else "туловище и голову"
            return WeightEstimate(count, height, widths, side_widths, lengths, None, None, None, None, False,
                                  coverage, f"Не видно: {missing}. Нужен человек в кадре целиком.")
        observed_side = "torso" in side_widths
        volume = 0.
        for part, width in widths.items():
            depth = side_widths.get(part, width * ASSUMED_DEPTH_RATIO[part])
            area = math.pi / 4 * width * depth
            volume += area * lengths[part] * (2 / 3 if part == "head" else 1) * LIMB_COUNT[part]
        mass = volume * BODY_DENSITY_KG_M3 * HANDS_FEET_FACTOR
        spread = self.relative_spread(front or self.samples)
        margin = max(MIN_RELATIVE_RANGE, spread) * (1 if observed_side else 1.5)
        message = ("Оценка готова; бок и перед измерены." if observed_side else
                   "Оценка предварительная: бок не измерен. Медленно повернитесь на месте.")
        return WeightEstimate(count, height, widths, side_widths, lengths, volume * 1000, mass,
                              mass * (1 - margin), mass * (1 + margin), observed_side, coverage, message)

    def relative_spread(self, selected) -> float:
        values = self.part_values("torso", selected)
        if len(values) < 3:
            return 0.
        low, middle, high = np.percentile(values, [16, 50, 84])
        return float((high - low) / (2 * middle)) if middle > 0 else 0.
