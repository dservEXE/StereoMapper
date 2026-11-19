"""Size and weight of a pig from its outline and the position in space of every pixel in it.

The outline gives the shape in the picture: where the head, the neck and the rump are, how thick
the body is along its whole length. The positions turn that into metres, each part at its own
distance from the camera, and tell from which side the body is seen, so no view has to be chosen.

The weight is found as for people: the body from the snout to the rump is cut into slices across
it, each an ellipse of the width and the depth of the body there, so a narrow head, a deep belly
and broad hams each count at their own place. The volume times the density of the body is the
mass. A camera sees one thickness of the body; the other is assumed from it. Length and heart
girth, the farm tape measurements, are kept as well, for comparison with a tape.
"""
from typing import NamedTuple

import cv2
import numpy as np

from .pig_scene import CAMERA_UP, surface_point

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
SPINE_FRACTIONS = tuple(float(value) for value in np.linspace(0., 1., 11))
# The ends of the trunk lie on the edge of the outline, where depth is least reliable.
INNER_SPINE = (.1, .9)
MIN_SPINE_POINTS = 5
SURFACE_WINDOW = .12
# Chest depth to chest width of a growing pig.
DEPTH_TO_WIDTH = 1.2
# Weight, kg = heart girth, m, squared x length from ears to tail root, m x 69.3
TAPE_FACTOR = 69.3
SLICES = 24
# Not measured on pigs: lean tissue is about 1.06 kg/l and fat about 0.92, and carcass density is
# known to follow the fat content; a live body also holds air. The pig database corrects the rest.
BODY_DENSITY_KG_M3 = 1030.
# The legs below the belly are cut away with the other thin parts and are not measured; assumed.
LEGS_FACTOR = 1.05
MIN_STRAIGHTNESS = .96
LENGTH_TO_WIDTH = (1.7, 5.5)
# Beyond this the body points at the camera and its length is mostly hidden.
MAX_TOWARD_CAMERA = .75
FORMULA_SPREAD = .1
UNCALIBRATED_SPREAD = .35
MAX_SAMPLES = 120
BAD_POSE = "поза не подходит для замера"
END_ON = "стоит к камере головой или хвостом"
NO_DEPTH = "расстояние не определено"


class PigShape(NamedTuple):
    """One outline measured in pixels; points are (x, y) in the image.

    slices are the thickness of the body across it, from the snout to the rump, and slice_points
    where each is taken, on the body line.
    """
    length_px: float
    full_length_px: float
    girth_px: float
    straightness: float
    head: tuple
    neck: tuple
    rump: tuple
    spine: tuple
    girth_point: tuple
    slices: tuple = ()
    slice_points: tuple = ()


class PigSample(NamedTuple):
    """One frame of one animal, in metres. side_share is 0 seen from above and 1 seen from the side."""
    length_m: float
    girth_m: float
    thickness_m: float
    side_share: float
    volume_m3: float | None = None

    @property
    def mass_kg(self) -> float:
        """From the volume when the slices are known, else by the tape formula."""
        if self.volume_m3:
            return self.volume_m3 * BODY_DENSITY_KG_M3 * LEGS_FACTOR
        return tape_mass(self.length_m, self.girth_m)


class PigEstimate(NamedTuple):
    length_m: float
    girth_m: float
    mass_kg: float
    mass_low_kg: float
    mass_high_kg: float
    samples: int
    volume_l: float | None = None

    def scaled(self, factor) -> "PigEstimate":
        """The same animal with every size `factor` times larger, so the weight is its cube times heavier."""
        cube = factor ** 3
        return PigEstimate(self.length_m * factor, self.girth_m * factor, self.mass_kg * cube, self.mass_low_kg * cube,
                           self.mass_high_kg * cube, self.samples, None if self.volume_l is None else self.volume_l * cube)


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
    end = len(profile) - 1
    trunk_count = end - neck
    zone = profile[neck + int(trunk_count * GIRTH_ZONE[0]):neck + max(int(trunk_count * GIRTH_ZONE[1]), int(trunk_count * GIRTH_ZONE[0]) + 1)]
    full_length = float(steps.sum())
    chord = float(np.hypot(end, spine[-1] - spine[0]))
    bounds = np.linspace(0, end + 1, SLICES + 1)
    parts = [profile[int(bounds[index]):max(int(bounds[index]) + 1, int(bounds[index + 1]))] for index in range(SLICES)]

    def point(index) -> tuple:
        index = int(round(index))
        position = first + index if head_first else last - index
        x, y = centre + direction * (start + position + .5) + across * spine[index]
        return float(x), float(y)
    return PigShape(float(steps[neck:].sum()), full_length, float(np.median(zone)),
                    chord / full_length if full_length else 0., point(0), point(neck), point(end),
                    tuple(point(neck + fraction * trunk_count) for fraction in SPINE_FRACTIONS),
                    point(neck + trunk_count * sum(GIRTH_ZONE) / 2),
                    tuple(float(np.median(part)) for part in parts),
                    tuple(point(min(end, (bounds[index] + bounds[index + 1]) / 2)) for index in range(SLICES)))


def ellipse_perimeter(width, depth) -> float:
    a, b = width / 2, depth / 2
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def tape_mass(length_m, girth_m) -> float:
    return TAPE_FACTOR * girth_m * girth_m * length_m


def measure_pig(shape, points, mask, focal, floor=None) -> tuple:
    """(sample, "") in metres, or (None, why not).

    The length is measured along the body in space, so an animal standing at an angle to the camera
    is not shortened. The thickness seen in the picture is the width from above, the depth of the
    body from the side, and a mix of the two in between; the floor tells which.
    """
    if shape.straightness < MIN_STRAIGHTNESS:
        return None, BAD_POSE
    radius = max(2, int(shape.girth_px * SURFACE_WINDOW))
    located = [(fraction, surface_point(points, x, y, radius, mask)) for fraction, (x, y) in zip(SPINE_FRACTIONS, shape.spine)]
    inner = [(fraction, spot) for fraction, spot in located if spot is not None and INNER_SPINE[0] <= fraction <= INNER_SPINE[1]]
    if len(inner) < MIN_SPINE_POINTS:
        return None, NO_DEPTH
    fractions, spots = np.array([fraction for fraction, _ in inner]), np.array([spot for _, spot in inner])
    centre = spots.mean(0)
    axis = np.linalg.svd(spots - centre, full_matrices=False)[2][0]
    slope = float(np.polyfit(fractions, (spots - centre) @ axis, 1)[0])
    length, axis = abs(slope), axis if slope > 0 else -axis
    view = centre / np.linalg.norm(centre)
    if abs(float(axis @ view)) > MAX_TOWARD_CAMERA:
        return None, END_ON
    chest = surface_point(points, *shape.girth_point, radius, mask)
    thickness = shape.girth_px * float(centre[2] if chest is None else chest[2]) / focal
    side_share = seen_from_side(axis, view, CAMERA_UP if floor is None else floor.normal)
    width = width_of(thickness, side_share)
    # Compared by width, so that the deeper body seen from the side passes the same test as from above.
    if not width or not LENGTH_TO_WIDTH[0] <= length / width <= LENGTH_TO_WIDTH[1]:
        return None, BAD_POSE
    volume = slices_volume(shape, points, mask, focal, length / shape.length_px, side_share, float(centre[2]), radius)
    return PigSample(length, ellipse_perimeter(width, width * DEPTH_TO_WIDTH), thickness, side_share, volume), ""


def width_of(thickness, side_share) -> float:
    """The width of the body from the thickness seen, by how much of the side is seen."""
    return thickness / float(np.sqrt(DEPTH_TO_WIDTH ** 2 * side_share + 1 - side_share))


def slices_volume(shape, points, mask, focal, metres_per_pixel, side_share, distance, radius) -> float | None:
    """Sum of elliptic slices from the snout to the rump, each at its own distance from the camera.

    metres_per_pixel is along the body, so an animal standing at an angle is not shortened.
    """
    if not shape.slices:
        return None
    step = shape.full_length_px / len(shape.slices) * metres_per_pixel
    volume = 0.
    for thickness_px, (x, y) in zip(shape.slices, shape.slice_points):
        spot = surface_point(points, x, y, radius, mask)
        width = width_of(thickness_px * (distance if spot is None else float(spot[2])) / focal, side_share)
        volume += np.pi / 4 * width * width * DEPTH_TO_WIDTH * step
    return float(volume)


def seen_from_side(axis, view, up) -> float:
    """0 when the thickness in the picture is the width of the body, 1 when it is its depth."""
    vertical = up - (up @ axis) * axis
    if np.linalg.norm(vertical) < .2:
        return .5
    vertical = vertical / np.linalg.norm(vertical)
    seen = np.cross(axis, view)
    seen = seen / np.linalg.norm(seen)
    return float(min(1., (seen @ vertical) ** 2))


class PigAccumulator:
    """Measurements of one animal over the frames."""

    def __init__(self) -> None:
        self.samples = []

    def add(self, sample, factor=1.) -> None:
        volume = None if sample.volume_m3 is None else sample.volume_m3 * factor ** 3
        self.samples = [*self.samples[-(MAX_SAMPLES - 1):],
                        (sample.length_m * factor, sample.girth_m * factor, sample.mass_kg * factor ** 3, volume)]

    def estimate(self, calibrated=True) -> PigEstimate | None:
        if not self.samples:
            return None
        lengths, girths, masses, volumes = zip(*self.samples)
        each = np.array(masses)
        mass = float(np.median(each))
        known = [volume for volume in volumes if volume is not None]
        spread = FORMULA_SPREAD if calibrated else UNCALIBRATED_SPREAD
        low = min(float(np.percentile(each, 25)), mass * (1 - spread))
        high = max(float(np.percentile(each, 75)), mass * (1 + spread))
        return PigEstimate(float(np.median(lengths)), float(np.median(girths)), mass, low, high, len(self.samples),
                           float(np.median(known)) * 1000 if known else None)
