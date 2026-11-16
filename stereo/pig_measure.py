"""Size and weight of a pig from its outline, seen from above or from the side.

Geometry only. The outline and the scale in metres per pixel come from elsewhere. The weight is
the farm tape formula: heart girth squared times body length. One camera sees only one thickness
of the body, so the other is assumed from it.
"""
from typing import NamedTuple

import cv2
import numpy as np

MIN_AREA_PX = 400
# Ears, the tail and, from the side, the legs are thinner than this share of the trunk and are cut away.
THIN_PART_RATIO = .45
SMOOTH_SHARE = .05
CENTRELINE_SMOOTH_SHARE = .15
END_WIDTH_SHARE = .2
HEAD_COMPARE_SHARE = .18
# Where the neck may be, as a share of the full length from the snout; the fallback when no dip is found.
HEAD_SEARCH = (.10, .38)
DEFAULT_HEAD_SHARE = .22
MIN_NECK_DIP = .06
# Just behind the shoulders, as a share of the trunk length from the neck.
GIRTH_ZONE = (.12, .35)
# Chest depth to chest width of a growing pig.
DEPTH_TO_WIDTH = 1.2
# Weight, kg = heart girth, m, squared x length from ears to tail root, m x 69.3
TAPE_FACTOR = 69.3
# The widest level of the body above the floor, as a share of the trunk length.
BODY_LEVEL_TO_LENGTH = .4
MIN_STRAIGHTNESS = .96
LENGTH_TO_THICKNESS = (1.8, 5.5)
FORMULA_SPREAD = .1
MAX_SAMPLES = 120
VIEWS = ("top", "side")


class PigShape(NamedTuple):
    """One outline measured in pixels; points are (x, y) in the image."""
    length_px: float
    full_length_px: float
    girth_px: float
    straightness: float
    head: tuple
    neck: tuple
    rump: tuple


class PigEstimate(NamedTuple):
    length_m: float
    thickness_m: float
    girth_m: float
    mass_kg: float
    mass_low_kg: float
    mass_high_kg: float
    samples: int


def largest_part(mask) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count < 2:
        return np.zeros(mask.shape, bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def trunk_mask(mask) -> np.ndarray | None:
    """The body without thin parts: ears, tail, legs."""
    body = largest_part(mask)
    if body.sum() < MIN_AREA_PX:
        return None
    thickness = 2 * cv2.distanceTransform(body.astype(np.uint8), cv2.DIST_L2, 5).max()
    size = max(3, int(thickness * THIN_PART_RATIO)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    trunk = largest_part(cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_OPEN, kernel))
    return trunk if trunk.sum() >= MIN_AREA_PX else None


def smooth(values, share) -> np.ndarray:
    window = max(3, int(len(values) * share)) | 1
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def neck_index(profile) -> int:
    """Where the head ends, counted from the snout: the dip before the shoulders, or the usual share."""
    count = len(profile)
    first, last = int(count * HEAD_SEARCH[0]), int(count * HEAD_SEARCH[1])
    usual = int(count * DEFAULT_HEAD_SHARE)
    if last - first < 2 or first < 1:
        return usual
    dip = first + int(np.argmin(profile[first:last]))
    is_dip = (profile[dip] <= profile[:dip].max() * (1 - MIN_NECK_DIP)
              and profile[dip] <= profile[dip:].max() * (1 - MIN_NECK_DIP))
    return dip if is_dip else usual


def pig_shape(mask) -> PigShape | None:
    """Body length along the spine and thickness behind the shoulders, in pixels."""
    rows, columns = np.nonzero(mask)
    if len(rows) < MIN_AREA_PX:
        return None
    top, left = max(0, rows.min() - 2), max(0, columns.min() - 2)
    trunk = trunk_mask(mask[top:rows.max() + 3, left:columns.max() + 3])
    if trunk is None:
        return None
    ys, xs = np.nonzero(trunk)
    points = np.stack((xs + left, ys + top), 1).astype(np.float64)
    centre = points.mean(0)
    direction = np.linalg.eigh(np.cov((points - centre).T))[1][:, 1]
    across = np.array((-direction[1], direction[0]))
    along, side = (points - centre) @ direction, (points - centre) @ across
    start = along.min()
    bins = np.floor(along - start).astype(int)
    counts = np.bincount(bins).astype(np.float64)
    offsets = np.bincount(bins, weights=side) / np.maximum(counts, 1)
    thickness = smooth(counts, SMOOTH_SHARE)
    kept = np.nonzero(thickness >= np.median(thickness) * END_WIDTH_SHARE)[0]
    if len(kept) < 10:
        return None
    first, last = int(kept[0]), int(kept[-1])
    profile = thickness[first:last + 1]
    spine = smooth(offsets, CENTRELINE_SMOOTH_SHARE)[first:last + 1]
    compare = max(1, int(len(profile) * HEAD_COMPARE_SHARE))
    head_first = profile[:compare].mean() <= profile[-compare:].mean()
    if not head_first:
        profile, spine = profile[::-1], spine[::-1]
    steps = np.hypot(1., np.diff(spine))
    neck = neck_index(profile)
    trunk_count = len(profile) - neck
    zone = profile[neck + int(trunk_count * GIRTH_ZONE[0]):neck + max(int(trunk_count * GIRTH_ZONE[1]), int(trunk_count * GIRTH_ZONE[0]) + 1)]
    full_length = float(steps.sum())
    chord = float(np.hypot(len(profile) - 1, spine[-1] - spine[0]))

    def point(index) -> tuple:
        position = first + index if head_first else last - index
        x, y = centre + direction * (start + position + .5) + across * spine[index]
        return float(x), float(y)
    return PigShape(float(steps[neck:].sum()), full_length, float(np.median(zone)),
                    chord / full_length if full_length else 0., point(0), point(neck), point(len(profile) - 1))


def plausible(shape) -> bool:
    """A straight body of pig proportions; a turning pig or two merged outlines are skipped."""
    ratio = shape.length_px / shape.girth_px if shape.girth_px else 0.
    return shape.straightness >= MIN_STRAIGHTNESS and LENGTH_TO_THICKNESS[0] <= ratio <= LENGTH_TO_THICKNESS[1]


def body_scale(floor_scale, camera_height, length_px) -> float:
    """Metres per pixel at the body, from the scale measured on the floor under a camera looking down.

    The body is closer to the camera than the floor, by a height that grows with the pig, so the
    same pixels mean fewer metres. Without the camera height the floor scale is returned as it is.
    """
    if not camera_height:
        return floor_scale
    return floor_scale / (1 + floor_scale * BODY_LEVEL_TO_LENGTH * length_px / camera_height)


def ellipse_perimeter(width, depth) -> float:
    a, b = width / 2, depth / 2
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def tape_mass(length_m, thickness_m, view="top") -> tuple:
    """(heart girth in metres, weight in kg) from the body length and the one thickness the camera sees."""
    if view == "side":
        width, depth = thickness_m / DEPTH_TO_WIDTH, thickness_m
    else:
        width, depth = thickness_m, thickness_m * DEPTH_TO_WIDTH
    girth = ellipse_perimeter(width, depth)
    return girth, TAPE_FACTOR * girth * girth * length_m


class PigAccumulator:
    """Measurements of one animal over the frames, kept in pixels so the scale can be set later."""

    def __init__(self) -> None:
        self.samples = []

    def add(self, shape) -> bool:
        if not plausible(shape):
            return False
        self.samples = [*self.samples[-(MAX_SAMPLES - 1):], (shape.length_px, shape.girth_px)]
        return True

    def estimate(self, floor_scale, camera_height=None, view="top") -> PigEstimate | None:
        if not self.samples or not floor_scale:
            return None
        lengths, thicknesses = (np.array(values) for values in zip(*self.samples))
        length_px = float(np.median(lengths))
        scale = body_scale(floor_scale, camera_height, length_px) if view == "top" else floor_scale
        length, thickness = length_px * scale, float(np.median(thicknesses)) * scale
        girth, mass = tape_mass(length, thickness, view)
        each = [tape_mass(a * scale, b * scale, view)[1] for a, b in self.samples]
        low = min(float(np.percentile(each, 25)), mass * (1 - FORMULA_SPREAD))
        high = max(float(np.percentile(each, 75)), mass * (1 + FORMULA_SPREAD))
        return PigEstimate(length, thickness, girth, mass, low, high, len(self.samples))
