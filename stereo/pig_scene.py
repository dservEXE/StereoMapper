"""The scene behind one picture: where every pixel is in space, where the floor is, what the scale is.

A depth model stands in for the second camera of a stereo pair and gives metres from the camera
for every pixel. Its absolute scale is rough, so a real stereo pair, or one segment of known
length, corrects it. Proportions inside the picture, a far pig against a near one, hold much better.
"""
from typing import NamedTuple

import cv2
import numpy as np

# The field of view of the pictures the depth model learned from, as focal length over the image diagonal.
FOCAL_TO_DIAGONAL = .693
POINT_WINDOW_PX = 3
SEGMENT_INNER = (.05, .95)
SEGMENT_SAMPLES = 25
MIN_SEGMENT_SAMPLES = 8
MIN_SCALE_PIXELS = 500
SCALE_LIMITS = (.2, 5.)
FLOOR_BAND_SHARE = .15
FLOOR_BAND_MIN_PX = 6
FLOOR_SAMPLE = 3000
FLOOR_ITERATIONS = 150
FLOOR_TOLERANCE = .03
MIN_FLOOR_POINTS = 200
MIN_FLOOR_SHARE = .35
MAX_ANIMAL_HEIGHT_M = 1.5
CONTACT_REACH_PX = 9
# Touching along more than this share of the outline, with the neighbour nearer by this share of the distance.
CONTACT_SHARE = .12
CONTACT_DEPTH_STEP = .02
CAMERA_UP = np.array((0., -1., 0.))


class Floor(NamedTuple):
    """Height above the floor of a point p is normal @ p + offset; the camera is above the floor."""
    normal: np.ndarray
    offset: float

    def height(self, points) -> np.ndarray:
        return points @ self.normal + self.offset


def assumed_focal(shape) -> float:
    """Focal length in pixels for a picture whose camera is unknown."""
    return FOCAL_TO_DIAGONAL * float(np.hypot(shape[0], shape[1]))


def scene_points(depth, focal, centre=None) -> np.ndarray:
    """The position in metres of every pixel: x to the right, y down, z away from the camera."""
    height, width = depth.shape
    centre_x, centre_y = centre or ((width - 1) / 2, (height - 1) / 2)
    xs = (np.arange(width, dtype=np.float32) - centre_x) / focal
    ys = (np.arange(height, dtype=np.float32) - centre_y) / focal
    return np.dstack((xs[None, :] * depth, ys[:, None] * depth, depth)).astype(np.float32)


def surface_point(points, x, y, radius=POINT_WINDOW_PX, mask=None) -> np.ndarray | None:
    """A steady position near a pixel: the median over a small window, inside the outline if given."""
    height, width = points.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < width and 0 <= y < height):
        return None
    window = (slice(max(0, y - radius), min(height, y + radius + 1)), slice(max(0, x - radius), min(width, x + radius + 1)))
    near = points[window]
    usable = np.isfinite(near[:, :, 2]) & (near[:, :, 2] > 0)
    if mask is not None:
        usable &= mask[window]
    if usable.sum() < 3:
        return None
    return np.median(near[usable], axis=0).astype(np.float64)


def ray_through(points, x, y) -> np.ndarray | None:
    """The direction from the camera through a pixel, scaled so that its z is 1."""
    height, width = points.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < width and 0 <= y < height):
        return None
    spot = points[y, x].astype(np.float64)
    return spot / spot[2] if np.isfinite(spot).all() and spot[2] > 0 else None


def segment_length(points, segment) -> float | None:
    """Metres between two marked pixels, each at its own distance from the camera.

    The ends of a marked thing lie on its edge, where the distance jumps to whatever is behind.
    So the distance is read along the whole segment and a straight line is put through it; an
    animal standing across the segment, or the background at its ends, does not move the result.
    """
    (x0, y0), (x1, y1) = segment
    rays = [ray_through(points, x, y) for x, y in segment]
    if rays[0] is None or rays[1] is None:
        return None
    shares = np.linspace(SEGMENT_INNER[0], SEGMENT_INNER[1], SEGMENT_SAMPLES)
    seen = [(share, surface_point(points, x0 + (x1 - x0) * share, y0 + (y1 - y0) * share, 1)) for share in shares]
    seen = [(share, spot[2]) for share, spot in seen if spot is not None]
    if len(seen) < MIN_SEGMENT_SAMPLES:
        return None
    share, depth = (np.array(values) for values in zip(*seen))
    first, second = np.triu_indices(len(share), 1)
    slope = float(np.median((depth[second] - depth[first]) / (share[second] - share[first])))
    start = float(np.median(depth - slope * share))
    if start <= 0 or start + slope <= 0:
        return None
    return float(np.linalg.norm(rays[1] * (start + slope) - rays[0] * start))


def stereo_scale(estimated, measured, valid) -> float | None:
    """How many times the estimated depth must be stretched to agree with a real stereo pair."""
    usable = valid & np.isfinite(measured) & (measured > 0) & (estimated > 0)
    if usable.sum() < MIN_SCALE_PIXELS:
        return None
    scale = float(np.median(measured[usable] / estimated[usable]))
    return scale if SCALE_LIMITS[0] <= scale <= SCALE_LIMITS[1] else None


def hidden_behind(outlines, points) -> list:
    """For every animal, whether another one stands in front of it and covers a part of its body.

    Two outlines that touch along a good stretch belong to animals side by side or one behind the
    other; the distance at the contact tells which. The covered one cannot be measured by what is left.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CONTACT_REACH_PX, CONTACT_REACH_PX))
    grown = [cv2.dilate(outline.astype(np.uint8), kernel).astype(bool) for outline in outlines]
    depth = points[:, :, 2]
    hidden = []
    for index, outline in enumerate(outlines):
        rim = grown[index] & ~outline
        limit = CONTACT_SHARE * max(1, int(rim.sum()))
        covered = False
        for other, neighbour in enumerate(outlines):
            contact = rim & neighbour
            if other == index or contact.sum() <= limit:
                continue
            own = outline & grown[other]
            if own.any() and float(np.median(depth[contact])) < float(np.median(depth[own])) * (1 - CONTACT_DEPTH_STEP):
                covered = True
                break
        hidden.append(covered)
    return hidden


def floor_candidates(outlines) -> np.ndarray:
    """Pixels just under every animal in the picture: mostly the floor it stands on."""
    if not outlines:
        return np.zeros((0, 0), bool)
    everything = np.zeros(outlines[0].shape, bool)
    for outline in outlines:
        everything |= outline
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blocked = cv2.dilate(everything.astype(np.uint8), kernel).astype(bool)
    band = np.zeros(everything.shape, bool)
    for outline in outlines:
        rows = np.nonzero(outline.any(axis=1))[0]
        if not len(rows):
            continue
        shift = max(FLOOR_BAND_MIN_PX, int((rows[-1] - rows[0] + 1) * FLOOR_BAND_SHARE))
        band[shift:] |= outline[:-shift]
    return band & ~blocked


def plane_through(points) -> Floor | None:
    centre = points.mean(0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][-1]
    return oriented(normal, -float(normal @ centre))


def oriented(normal, offset) -> Floor | None:
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    normal, offset = normal / length, offset / length
    # The camera sits at the origin, above the floor.
    return Floor(-normal, -offset) if offset < 0 else Floor(normal, offset)


def fit_floor(points, candidates, animals=(), seed=0) -> Floor | None:
    """The floor plane from pixels likely to lie on it, such that the animals stand on it."""
    if not candidates.size:
        return None
    sample = points[candidates].astype(np.float64)
    sample = sample[np.isfinite(sample).all(1) & (sample[:, 2] > 0)]
    if len(sample) < MIN_FLOOR_POINTS:
        return None
    rng = np.random.default_rng(seed)
    if len(sample) > FLOOR_SAMPLE:
        sample = sample[rng.choice(len(sample), FLOOR_SAMPLE, replace=False)]
    tolerance = FLOOR_TOLERANCE * float(np.median(sample[:, 2]))
    best, best_count = None, 0
    for _ in range(FLOOR_ITERATIONS):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        floor = oriented(normal, -float(normal @ a))
        if floor is None:
            continue
        count = int((np.abs(floor.height(sample)) < tolerance).sum())
        if count > best_count and stands_on(floor, animals, tolerance):
            best, best_count = floor, count
    if best is None or best_count < MIN_FLOOR_SHARE * len(sample):
        return None
    return plane_through(sample[np.abs(best.height(sample)) < tolerance]) or best


def stands_on(floor, animals, tolerance) -> bool:
    """Most animals are a little above the plane: this tells the floor from a wall behind them."""
    if not len(animals):
        return True
    heights = [float(np.median(floor.height(animal))) for animal in animals if len(animal)]
    good = sum(-tolerance <= height <= MAX_ANIMAL_HEIGHT_M for height in heights)
    return good >= .7 * len(heights)
