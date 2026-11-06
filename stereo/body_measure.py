"""Measure a person from an outline, metric depth and joint positions, independent of the camera.

The outline gives the shape and depth gives the scale. The body from shoulders to ankles is cut
into height slices, each an ellipse from its front and side widths, so that a bust, belly, pelvis
and buttocks count at their own height. Arms and head are simple shapes. Turning on the spot
provides the side widths. The result is an estimate that has not been validated against scales;
WeightEstimate carries how much of it was observed rather than assumed.
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
# Stereo distance to one person jumped by a metre between frames at 3 m while their height in
# pixels moved by under 2%. Distance times pixel height is constant for one upright body, so a
# single scale per person removes that noise, and the person may even walk.
MIN_SCALE_SAMPLES = 5
MAX_SCALE_CORRECTION = 1.6
SAMPLES_ALONG_PART = 7
# Part of a limb next to a joint is widened by the neighbouring part, so sample its middle.
PART_START, PART_END = .25, .75

# Ankle height is about 3.9% of stature (Drillis and Contini, 1966).
ANKLE_HEIGHT_FRACTION = .039
BODY_DENSITY_KG_M3 = 1020.
# Hands and feet are about 4% of body mass (Winter, Biomechanics) and are not measured here.
HANDS_FEET_FACTOR = 1.04
# Side width over front width, used only when turning has not shown the side.
ASSUMED_DEPTH_RATIO = {"head": .85, "upper_arm": 1., "forearm": 1.}
LIMB_COUNT = {"head": 1, "upper_arm": 2, "forearm": 2}
# Reported as sizes only; their volume comes from the slices.
REPORTED_PARTS = ("torso", "thigh", "shin", *ASSUMED_DEPTH_RATIO)

SLICES = 32
# Slices reach this far to each side, in torso lengths: past the body, short of a raised hand.
SLICE_REACH = .75
MIN_RUN_PX = 2
SHOULDER_SLICES = 4
MAX_SHOULDER_RATIO = 1.35
# Side width over front width, for slices when turning has not shown the side.
ASSUMED_SLICE_RATIO, ASSUMED_LEG_RATIO = .62, 1.

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
    # Width of each slice from shoulders to ankles, NaN where an arm hides it, and whether
    # that slice shows two separate legs. Empty if the ankles are not visible.
    slice_widths: tuple = ()
    slice_split: tuple = ()
    slice_length: float | None = None


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
    # Widths from shoulders to ankles, for checking where the volume comes from.
    slices_front_m: tuple = ()
    slices_side_m: tuple = ()


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


def body_slices(mask, neck, ankle, reach_px, scale) -> tuple:
    """Each slice's width across the body and whether it shows two legs, from neck to ankles.

    A slice the outline fills to both ends of its reach is an outstretched arm, not body.
    """
    neck, ankle = np.asarray(neck, float), np.asarray(ankle, float)
    axis = ankle - neck
    length = float(np.linalg.norm(axis))
    if length < 8:
        return (), ()
    ux, uy = axis / length
    offsets = np.arange(-int(reach_px), int(reach_px) + 1)
    height, width = mask.shape
    widths, split = [], []
    for index in range(SLICES):
        cx, cy = neck + axis * (index + .5) / SLICES
        xs = np.clip(np.rint(cx - uy * offsets).astype(int), 0, width - 1)
        ys = np.clip(np.rint(cy + ux * offsets).astype(int), 0, height - 1)
        line = mask[ys, xs]
        if len(line) and line[0] and line[-1]:
            widths.append(math.nan)
            split.append(False)
            continue
        changes = np.flatnonzero(np.diff(np.concatenate(((0,), line.astype(np.int8), (0,)))))
        runs = changes[1::2] - changes[::2]
        total = float(runs[runs >= MIN_RUN_PX].sum()) * scale
        # A body has no empty level between shoulders and ankles: the outline missed this one.
        widths.append(total if total > 0 else math.nan)
        split.append(int((runs >= MIN_RUN_PX).sum()) >= 2)
    return tuple(widths), tuple(split)


def span_width(mask, start, end, scale) -> float | None:
    """Median full width across a part, for the torso: its bones are not at its centre from the side."""
    return _across(mask, start, end, scale, lambda first, second: first + second + 1)


def part_width(mask, start, end, scale) -> float | None:
    """Median width across a limb, in metres.

    Twice the shorter side from the bone: legs held together or an arm against the
    body would otherwise count as one wide part.
    """
    return _across(mask, start, end, scale, lambda first, second: 2 * min(first, second) + 1)


def _across(mask, start, end, scale, combine) -> float | None:
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
            widths.append(combine(run_length(mask, x, y, -uy, ux, limit), run_length(mask, x, y, uy, -ux, limit)))
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
    width = span_width(mask, neck, pelvis, scale)
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
    slices, split, slice_length = (), (), None
    visible_ankles = [kp[i] for i in (L_ANKLE, R_ANKLE) if seen[i]]
    if visible_ankles:
        ankle = np.mean(visible_ankles, axis=0)
        slices, split = body_slices(mask, neck, ankle, torso_px * SLICE_REACH, scale)
        slice_length = float(np.linalg.norm(ankle - neck)) * scale
    return BodySample(body_depth, height, shoulder_span, widths, lengths, slices, split, slice_length)


def rescaled(sample: BodySample, depth: float) -> BodySample:
    """The same measurement at a steadier distance: every size is proportional to depth."""
    ratio = depth / sample.depth
    return BodySample(depth, sample.height * ratio if sample.height is not None else None,
                      sample.shoulder_span * ratio,
                      {key: value * ratio for key, value in sample.widths.items()},
                      {key: value * ratio for key, value in sample.lengths.items()},
                      tuple(value * ratio for value in sample.slice_widths), sample.slice_split,
                      sample.slice_length * ratio if sample.slice_length is not None else None)


def slice_profile(samples) -> np.ndarray | None:
    """Typical width of each slice over these frames, with gaps from arms filled by neighbours."""
    rows = [sample.slice_widths for sample in samples if sample.slice_widths]
    if len(rows) < MIN_BIN_SAMPLES:
        return None
    stacked = np.array(rows, float)
    usable = np.isfinite(stacked).sum(axis=0) >= max(2, len(rows) // 4)
    if usable.sum() < SLICES // 2:
        return None
    profile = np.full(SLICES, math.nan)
    profile[usable] = [float(np.median(column[np.isfinite(column)])) for column in stacked.T[usable]]
    # Arms held slightly down join the shoulders' slices without reaching the slice ends.
    chest = profile[SHOULDER_SLICES:SHOULDER_SLICES * 3]
    if np.isfinite(chest).any():
        limit = float(np.nanmedian(chest)) * MAX_SHOULDER_RATIO
        shoulders = np.zeros(SLICES, bool)
        shoulders[:SHOULDER_SLICES] = True
        usable = usable & ~(shoulders & (np.nan_to_num(profile) > limit))
    positions = np.arange(SLICES)
    return np.interp(positions, positions[usable], profile[usable])


def slices_volume(front, side, split, length) -> float:
    """Sum of elliptic slices. Where legs are apart each is half the front width and the full side."""
    step = length / SLICES
    area = np.where(split, 2 * math.pi / 4 * (front / 2) * side, math.pi / 4 * front * side)
    return float(area.sum() * step)


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


@dataclass
class BodyAccumulator:
    """Combine one person's frames: front and side views need separating as they turn."""
    samples: list = field(default_factory=list)

    def add(self, sample: BodySample) -> None:
        self.samples.append(sample)
        if len(self.samples) > MAX_SAMPLES:
            del self.samples[0]

    def steady(self) -> list:
        """All frames at one scale for this body, as if depth had been free of noise."""
        heights = [sample.height for sample in self.samples if sample.height]
        depths = [sample.depth for sample in self.samples]
        if len(heights) < MIN_SCALE_SAMPLES:
            middle = float(np.median(depths)) if depths else 0.
            return [rescaled(sample, middle) for sample in self.samples]
        height = float(np.median(heights))
        steady = []
        for sample in self.samples:
            if not sample.height:
                continue
            # This frame's outline can be cut short: then its pixels say nothing about scale.
            correction = height / sample.height
            if 1 / MAX_SCALE_CORRECTION <= correction <= MAX_SCALE_CORRECTION:
                steady.append(rescaled(sample, sample.depth * correction))
        return steady or list(self.samples)

    def views(self, samples) -> list:
        """Apparent turn of each sample, from how narrow the shoulders look."""
        widest = percentile([s.shoulder_span for s in samples], 95) or 1.
        return [math.degrees(math.acos(min(1., s.shoulder_span / max(widest, 1e-6)))) for s in samples]

    def part_values(self, part, selected) -> list:
        return [sample.widths[key] for sample in selected for key in (part, f"{part}_l", f"{part}_r")
                if key in sample.widths]

    def part_length(self, part, samples) -> float | None:
        # Foreshortening only ever shortens a part, so favour the longer views.
        values = [sample.lengths[key] for sample in samples for key in (part, f"{part}_l", f"{part}_r")
                  if key in sample.lengths]
        return percentile(values, 80)

    def estimate(self) -> WeightEstimate:
        count = len(self.samples)
        samples = self.steady()
        views = self.views(samples)
        front = [s for s, view in zip(samples, views) if view <= FRONT_MAX_DEG]
        side = [s for s, view in zip(samples, views) if view >= SIDE_MIN_DEG]
        coverage = dict(front=len(front), oblique=len(samples) - len(front) - len(side), side=len(side))
        heights = [s.height for s in self.samples if s.height is not None]
        height = percentile(heights, 50) if len(heights) >= MIN_BIN_SAMPLES else None
        if count < MIN_SAMPLES:
            return WeightEstimate(count, height, {}, {}, {}, None, None, None, None, False, coverage,
                                  f"Идёт сбор: {count} из {MIN_SAMPLES}")
        widths, side_widths, lengths = {}, {}, {}
        for part in REPORTED_PARTS:
            front_values = self.part_values(part, front)
            values = front_values if len(front_values) >= MIN_BIN_SAMPLES else self.part_values(part, samples)
            width = percentile(values, 50)
            length = self.part_length(part, samples)
            if width is None or length is None:
                continue
            widths[part], lengths[part] = width, length
            side_values = self.part_values(part, side)
            if len(side_values) >= MIN_BIN_SAMPLES:
                side_widths[part] = percentile(side_values, 50)
        front_slices = slice_profile(front) if len(front) >= MIN_BIN_SAMPLES else None
        front_slices = front_slices if front_slices is not None else slice_profile(samples)
        if not {"torso", "head", "thigh", "shin"} <= set(widths) or front_slices is None:
            missing = "ноги" if not {"thigh", "shin"} <= set(widths) or front_slices is None else "туловище и голову"
            return WeightEstimate(count, height, widths, side_widths, lengths, None, None, None, None, False,
                                  coverage, f"Не видно: {missing}. Нужен человек в кадре целиком.")
        lengths_seen = [sample.slice_length for sample in samples if sample.slice_length]
        split_rows = [sample.slice_split for sample in (front or samples) if sample.slice_split]
        split = np.mean(np.array(split_rows, float), axis=0) >= .5
        side_slices = slice_profile(side) if len(side) >= MIN_BIN_SAMPLES else None
        observed_side = side_slices is not None
        if side_slices is None:
            side_slices = front_slices * np.where(split, ASSUMED_LEG_RATIO / 2, ASSUMED_SLICE_RATIO)
        # Legs seen from the side overlap, so a slice's side width belongs to one leg at most.
        volume = slices_volume(front_slices, side_slices, split, percentile(lengths_seen, 80))
        for part in ASSUMED_DEPTH_RATIO:
            if part in widths:
                depth = side_widths.get(part, widths[part] * ASSUMED_DEPTH_RATIO[part])
                area = math.pi / 4 * widths[part] * depth
                volume += area * lengths[part] * (2 / 3 if part == "head" else 1) * LIMB_COUNT[part]
        mass = volume * BODY_DENSITY_KG_M3 * HANDS_FEET_FACTOR
        spread = self.relative_spread(front or self.samples)
        margin = max(MIN_RELATIVE_RANGE, spread) * (1 if observed_side else 1.5)
        message = ("Оценка готова; бок и перед измерены." if observed_side else
                   "Оценка предварительная: бок не измерен. Медленно повернитесь на месте.")
        return WeightEstimate(count, height, widths, side_widths, lengths, volume * 1000, mass,
                              mass * (1 - margin), mass * (1 + margin), observed_side, coverage, message,
                              tuple(float(value) for value in front_slices),
                              tuple(float(value) for value in side_slices))

    def relative_spread(self, selected) -> float:
        values = self.part_values("torso", selected)
        if len(values) < 3:
            return 0.
        low, middle, high = np.percentile(values, [16, 50, 84])
        return float((high - low) / (2 * middle)) if middle > 0 else 0.
