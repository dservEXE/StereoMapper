"""Carry every outline from frame to frame between two passes of the neural networks.

The networks take a second or more per frame on a processor, a video brings 25 or 30 frames a
second. Between their passes each animal's outline is moved along with its body: optical flow
follows small corners inside the outline from one frame to the next, and the turn, growth and
shift that fit them best move the outline, the body line and the box. A pig moves as one piece
over a second, so the outline stays on it until the networks look again.
"""
from typing import NamedTuple

import cv2
import numpy as np

from .pig_measure import PigShape

TRACK_WIDTH = 640
MAX_POINTS = 60
MIN_POINTS = 6
REFILL_BELOW = 20
CORNER_QUALITY = .01
CORNER_DISTANCE_PX = 5
EDGE_CLEARANCE_PX = 3
LK_WINDOW = (21, 21)
LK_LEVELS = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03)
# A point is trusted only if following it back lands where it started.
MAX_ROUND_TRIP_PX = 1.
RANSAC_PX = 2.
IDENTITY = np.array(((1., 0., 0.), (0., 1., 0.)))


def tracking_gray(image) -> tuple:
    """A small grey copy of the frame for following points, and how many times smaller it is."""
    height, width = image.shape[:2]
    scale = min(1., TRACK_WIDTH / width)
    small = image if scale == 1 else cv2.resize(image, (round(width * scale), round(height * scale)),
                                                interpolation=cv2.INTER_AREA)
    gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray, scale


def compose(later, earlier) -> np.ndarray:
    """One 2x3 motion doing `earlier`, then `later`."""
    return np.hstack((later[:, :2] @ earlier[:, :2], later[:, :2] @ earlier[:, 2:] + later[:, 2:]))


def moved(points, matrix) -> np.ndarray:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    return points @ matrix[:, :2].T + matrix[:, 2]


def motion_scale(matrix) -> float:
    return float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))


def moved_shape(shape, matrix) -> PigShape | None:
    """The body line and the slices where the motion puts them; widths grow with the motion."""
    if shape is None:
        return None

    def one(point) -> tuple:
        x, y = moved(point, matrix)[0]
        return float(x), float(y)
    grow = motion_scale(matrix)
    return shape._replace(length_px=shape.length_px * grow, full_length_px=shape.full_length_px * grow,
                          girth_px=shape.girth_px * grow, head=one(shape.head), neck=one(shape.neck),
                          rump=one(shape.rump), spine=tuple(one(point) for point in shape.spine),
                          girth_point=one(shape.girth_point),
                          slices=tuple(width * grow for width in shape.slices),
                          slice_points=tuple(one(point) for point in shape.slice_points))


def moved_contours(contours, matrix) -> tuple:
    return tuple(np.round(moved(contour, matrix)).astype(np.int32).reshape(-1, 1, 2) for contour in contours)


def contour_box(contours, fallback) -> tuple:
    if not contours:
        return fallback
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
    return (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max()))


def corners(gray, contours, scale) -> np.ndarray:
    """Points worth following inside an outline, away from its edge, in the small frame."""
    mask = np.zeros(gray.shape, np.uint8)
    polygons = [np.round(contour.reshape(-1, 2) * scale).astype(np.int32) for contour in contours]
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    mask = cv2.erode(mask, np.ones((2 * EDGE_CLEARANCE_PX + 1,) * 2, np.uint8))
    if not mask.any():
        return np.zeros((0, 2), np.float32)
    found = cv2.goodFeaturesToTrack(gray, MAX_POINTS, CORNER_QUALITY, CORNER_DISTANCE_PX, mask=mask)
    return np.zeros((0, 2), np.float32) if found is None else found.reshape(-1, 2).astype(np.float32)


class Follower(NamedTuple):
    """One outline as the networks found it, and the motion since then, in analysed pixels."""
    contours: tuple
    shape: PigShape | None
    box: tuple
    matrix: np.ndarray
    points: np.ndarray

    @property
    def now_contours(self) -> tuple:
        return moved_contours(self.contours, self.matrix)

    @property
    def now_shape(self) -> PigShape | None:
        return moved_shape(self.shape, self.matrix)

    @property
    def now_box(self) -> tuple:
        return contour_box(self.now_contours, self.box)


def follower(contours, shape, box, gray, scale) -> Follower:
    return Follower(contours, shape, tuple(box), IDENTITY.copy(), corners(gray, contours, scale))


def follow(previous, gray, followers, scale) -> list:
    """Every follower one frame on, from the previous small grey frame to this one."""
    counts = [len(item.points) for item in followers]
    if not sum(counts):
        return [refilled(item, gray, scale) for item in followers]
    start = np.concatenate([item.points for item in followers]).reshape(-1, 1, 2).astype(np.float32)
    ahead, found, _ = cv2.calcOpticalFlowPyrLK(previous, gray, start, None, winSize=LK_WINDOW, maxLevel=LK_LEVELS,
                                               criteria=LK_CRITERIA)
    back, returned, _ = cv2.calcOpticalFlowPyrLK(gray, previous, ahead, None, winSize=LK_WINDOW, maxLevel=LK_LEVELS,
                                                 criteria=LK_CRITERIA)
    good = (found.ravel() == 1) & (returned.ravel() == 1) & (np.linalg.norm((back - start).reshape(-1, 2), axis=1)
                                                            <= MAX_ROUND_TRIP_PX)
    start, ahead = start.reshape(-1, 2), ahead.reshape(-1, 2)
    result, offset = [], 0
    for item, count in zip(followers, counts):
        keep = good[offset:offset + count]
        old, new = start[offset:offset + count][keep], ahead[offset:offset + count][keep]
        offset += count
        step = step_motion(old, new, scale)
        if step is None:
            result.append(refilled(item._replace(points=new.astype(np.float32)), gray, scale))
            continue
        result.append(refilled(item._replace(matrix=compose(step, item.matrix), points=new.astype(np.float32)),
                               gray, scale))
    return result


def step_motion(old, new, scale) -> np.ndarray | None:
    """The turn, growth and shift of one body between two frames, in analysed pixels."""
    if len(old) < MIN_POINTS:
        if not len(old):
            return None
        shift = np.median(new - old, axis=0) / scale
        return np.array(((1., 0., shift[0]), (0., 1., shift[1])))
    matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if matrix is None:
        return None
    matrix = matrix.astype(np.float64)
    matrix[:, 2] /= scale
    return matrix


def refilled(item, gray, scale) -> Follower:
    """New corners when too few are left, found inside the outline where it is now."""
    if len(item.points) >= REFILL_BELOW:
        return item
    fresh = corners(gray, item.now_contours, scale)
    if not len(fresh):
        return item
    points = np.concatenate((item.points.reshape(-1, 2), fresh))[:MAX_POINTS].astype(np.float32)
    return item._replace(points=points)


def carried(found, grays, scale) -> list:
    """Outlines found in the first of `grays` carried through the others: the followers in every one of them.

    `found` holds (contours, shape, box) from the networks; the frames shown meanwhile are in grays.
    """
    followers = [follower(contours, shape, box, grays[0], scale) for contours, shape, box in found]
    steps = [followers]
    for previous, gray in zip(grays, grays[1:]):
        followers = follow(previous, gray, followers, scale)
        steps.append(followers)
    return steps
