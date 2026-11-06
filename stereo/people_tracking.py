"""Keep one label on each person within a session, by position, depth and clothing colour.

Labels are numbers that last only for the session. Nothing here recognises who a person is.
"""
from dataclasses import dataclass
import itertools

import cv2
import numpy as np

from .body_measure import L_HIP, L_SHOULDER, MIN_KEYPOINT_SCORE, R_HIP, R_SHOULDER

# A track keeps its label this long out of sight, and only appearance can bring it back then.
MEMORY_S = 30.
RECENT_S = 1.5
MIN_MATCH_SCORE = .25
MIN_REIDENTIFY_SIMILARITY = .75
HISTOGRAM_ALPHA = .2
MAX_DEPTH_STEP_M = 1.
CLEAR_OVERLAP = .3
PALETTE = ((80, 200, 255), (120, 230, 120), (255, 160, 90), (200, 130, 255), (90, 220, 220), (130, 130, 255))


def box_iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.


def clothing_histogram(image, observation) -> np.ndarray:
    """Hue and saturation of the torso, or of the upper box when joints are unreliable."""
    x0, y0, x1, y1 = (int(value) for value in observation.box)
    points = observation.keypoints
    scores = observation.keypoint_scores
    torso = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if all(scores[i] >= MIN_KEYPOINT_SCORE for i in torso):
        xs, ys = points[list(torso), 0], points[list(torso), 1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
    else:
        y1 = y0 + (y1 - y0) // 2
    height, width = image.shape[:2]
    x0, x1 = max(0, x0), min(width, max(x1, x0 + 2))
    y0, y1 = max(0, y0), min(height, max(y1, y0 + 2))
    region = image[y0:y1, x0:x1]
    if not region.size:
        return np.zeros((16, 8), np.float32)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(histogram, None, 1, 0, cv2.NORM_L1)


def similarity(a, b) -> float:
    return 1. - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


@dataclass
class Track:
    track_id: int
    box: tuple
    histogram: np.ndarray
    depth: float | None
    last_seen: float

    @property
    def color(self):
        return PALETTE[(self.track_id - 1) % len(PALETTE)]


class PersonTracker:
    def __init__(self):
        self.tracks = {}
        self.ids = itertools.count(1)

    def match_score(self, track, observation, histogram, depth, now) -> float:
        look = similarity(track.histogram, histogram)
        if now - track.last_seen > RECENT_S:
            # Out of sight for a while: it may have moved anywhere, so only appearance counts.
            return look if look >= MIN_REIDENTIFY_SIMILARITY else -1.
        overlap = box_iou(track.box, observation.box)
        # Stereo depth on one small patch can jump by metres, so it only rules out what the
        # picture does not already support: somebody else passing in front of the camera.
        if (overlap < CLEAR_OVERLAP and depth is not None and track.depth is not None
                and abs(depth - track.depth) > MAX_DEPTH_STEP_M):
            return -1.
        box = observation.box
        size = max(box[2] - box[0], box[3] - box[1], 1)
        shift = np.hypot((track.box[0] + track.box[2] - box[0] - box[2]) / 2,
                         (track.box[1] + track.box[3] - box[1] - box[3]) / 2)
        nearness = max(0., 1 - shift / size)
        return .45 * overlap + .25 * nearness + .3 * look

    def update(self, image, observations, depths, now) -> list:
        """Return (track, observation, index) for each observation, in input order."""
        for track_id in [key for key, track in self.tracks.items() if now - track.last_seen > MEMORY_S]:
            del self.tracks[track_id]
        histograms = [clothing_histogram(image, observation) for observation in observations]
        pairs = []
        for index, observation in enumerate(observations):
            for track in self.tracks.values():
                score = self.match_score(track, observation, histograms[index], depths[index], now)
                if score >= MIN_MATCH_SCORE:
                    pairs.append((score, track.track_id, index))
        assigned, used = {}, set()
        for score, track_id, index in sorted(pairs, key=lambda pair: -pair[0]):
            if track_id not in used and index not in assigned:
                assigned[index] = track_id
                used.add(track_id)
        results = []
        for index, observation in enumerate(observations):
            track = self.tracks.get(assigned.get(index))
            if track is None:
                track = Track(next(self.ids), observation.box, histograms[index], depths[index], now)
                self.tracks[track.track_id] = track
            else:
                track.box, track.last_seen = observation.box, now
                track.depth = depths[index] if depths[index] is not None else track.depth
                track.histogram = cv2.addWeighted(track.histogram, 1 - HISTOGRAM_ALPHA,
                                                  histograms[index], HISTOGRAM_ALPHA, 0)
            results.append((track, observation, index))
        return results
