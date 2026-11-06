"""Background person weighing: never slows capture or depth, always works on the newest frame."""
from dataclasses import dataclass
import json
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time

import numpy as np

from .body_measure import (BodyAccumulator, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, measure_person, outline_depth,
                           outline_disparity)
from .people_tracking import PersonTracker
from .version import VERSION


DEFAULT_DISTANCE_M = 3.
MAX_MEASURE_FAILURES = 5


@dataclass(frozen=True)
class PersonView:
    """What the interface needs about one person in the latest analysed frame."""
    track_id: int
    color: tuple
    box: tuple
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    depth: float | None
    estimate: object
    # The body pixels counted at the latest measurement, for checking what went into the sum.
    outline: np.ndarray | None = None


@dataclass(frozen=True)
class WeighingState:
    people: tuple
    known: dict
    fps: float
    message: str
    # The analysed image itself, so labels never lag behind the picture they describe.
    image: np.ndarray | None = None


def torso_depth(frame, observation) -> float | None:
    points = observation.keypoints[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
    x, y = np.round(points.mean(axis=0)).astype(int)
    height, width = frame.valid.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    window = (slice(max(0, y - 4), y + 5), slice(max(0, x - 4), x + 5))
    values = frame.points[window][frame.valid[window]]
    return float(np.median(values[:, 2])) if len(values) else None


def person_result(track_id, estimate) -> dict:
    """Numbers for one person, as saved in experiments and in the people database."""
    return dict(label=track_id, samples=estimate.samples, height_m=estimate.height_m, widths_m=estimate.widths_m,
                side_widths_m=estimate.side_widths_m, lengths_m=estimate.lengths_m, volume_l=estimate.volume_l,
                mass_kg=estimate.mass_kg, mass_low_kg=estimate.mass_low_kg, mass_high_kg=estimate.mass_high_kg,
                side_observed=estimate.observed_side, views=estimate.view_coverage,
                slices_front_m=estimate.slices_front_m, slices_side_m=estimate.slices_side_m)


class WeighingSession(Thread):
    def __init__(self, detector_factory, status=lambda message: None):
        super().__init__(daemon=True, name="person-weighing")
        self.detector_factory, self.status = detector_factory, status
        self.frames = Queue(maxsize=1)
        self.outlines = Queue(maxsize=1)
        self.stop_event = Event()
        self.lock = Lock()
        self.tracker = PersonTracker()
        self.bodies = {}
        self.state = WeighingState((), {}, 0., "Загружаю модели поиска людей…")
        self.error = None
        self.reset_requested = False
        self.measured_at = {}
        self.outlines_seen = {}
        # Numbers restart after a reset: a measurement still in progress must not reach the new person.
        self.generation = 0

    def submit(self, frame, K, baseline=None) -> None:
        """Newest frame for analysis. A baseline in metres enables distance from both outlines."""
        try:
            self.frames.put_nowait((frame, K, baseline))
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            self.frames.put_nowait((frame, K, baseline))

    def reset(self) -> None:
        self.reset_requested = True

    def outline_distance(self, detector, frame, K, baseline, mask, observation) -> float | None:
        """Distance from the same body traced in the second camera, or None to use the depth map."""
        right = getattr(frame, "right", None)
        if not baseline or right is None or mask is None:
            return None
        focal = float(K[0, 0] + K[1, 1]) / 2
        z = np.nan_to_num(frame.points[:, :, 2]).astype(np.float32)
        rough = outline_depth(z, frame.valid & (z > 0), mask) or DEFAULT_DISTANCE_M
        shift = focal * baseline / rough
        x0, y0, x1, y1 = observation.box
        moved = observation._replace(box=(x0 - shift, y0, x1 - shift, y1),
                                     keypoints=observation.keypoints - np.array((shift, 0.), np.float32))
        other = detector.outline(right, moved)
        disparity = outline_disparity(mask, other) if other is not None else None
        return focal * baseline / disparity if disparity else None

    def analyse(self, detector, frame, K, fps, baseline=None) -> WeighingState:
        now = time.perf_counter()
        observations = detector.detect(frame.left)
        depths = [torso_depth(frame, observation) for observation in observations]
        views, waiting = [], []
        matched = self.tracker.update(frame.left, observations, depths, now)
        self.retire_forgotten()
        for track, observation, index in matched:
            with self.lock:
                estimate = self.bodies.setdefault(track.track_id, BodyAccumulator()).estimate()
                outline = self.outlines_seen.get(track.track_id)
            views.append(PersonView(track.track_id, track.color, observation.box, observation.keypoints,
                                    observation.keypoint_scores, depths[index], estimate, outline))
            waiting.append((track.track_id, observation))
        self.request_outline(frame, K, baseline, waiting)
        people = tuple(sorted(views, key=lambda view: view.track_id))
        with self.lock:
            known = {track_id: body.estimate() for track_id, body in self.bodies.items()}
        message = f"В кадре людей: {len(people)}" if people else "Людей в кадре не видно"
        if getattr(detector, "device_label", None):
            message += " · " + detector.device_label
        return WeighingState(people, known, fps, message, frame.left)

    def retire_forgotten(self) -> None:
        """A label that left the tracker never returns: keep its result and free its memory."""
        with self.lock:
            for track_id, body in self.bodies.items():
                if track_id not in self.tracker.tracks and body.samples:
                    body.retire()
                    self.outlines_seen.pop(track_id, None)
                    self.measured_at.pop(track_id, None)

    def request_outline(self, frame, K, baseline, waiting) -> None:
        """Hand the least recently measured person to the slower outline thread, if it is free."""
        if not waiting or self.outlines.full():
            return
        track_id, observation = min(waiting, key=lambda item: self.measured_at.get(item[0], 0.))
        self.measured_at[track_id] = time.perf_counter()
        self.outlines.put((frame, K, baseline, track_id, observation, self.generation))

    def measure_loop(self, detector) -> None:
        """Outlines take a while, so labels must not wait for them."""
        failures = 0
        while not self.stop_event.is_set():
            try:
                frame, K, baseline, track_id, observation, generation = self.outlines.get(timeout=.1)
            except Empty:
                continue
            try:
                mask = detector.outline(frame.left, observation)
                depth = self.outline_distance(detector, frame, K, baseline, mask, observation)
                sample = measure_person(frame.points, frame.valid, K, observation, mask, depth)
                failures = 0
            except Exception as error:
                # One bad frame costs one measurement; repeating failures stop the session visibly.
                failures += 1
                if failures >= MAX_MEASURE_FAILURES:
                    self.fail(f"Ошибка измерения: {error}")
                    self.stop_event.set()
                    return
                self.status(f"Замер пропущен: {error}")
                continue
            with self.lock:
                body = self.bodies.get(track_id) if generation == self.generation else None
                if sample is not None and body is not None:
                    body.add(sample)
                    self.outlines_seen[track_id] = mask

    def run(self) -> None:
        try:
            detector = self.detector_factory(self.status)
        except Exception as error:
            self.fail(f"Не удалось загрузить модели поиска людей: {error}")
            return
        measuring = Thread(target=self.measure_loop, args=(detector,), daemon=True, name="person-measuring")
        measuring.start()
        with self.lock:
            self.state = WeighingState((), {}, 0., "Модели готовы, жду кадр глубины…")
        fps, previous = 0., None
        while not self.stop_event.is_set():
            try:
                frame, K, baseline = self.frames.get(timeout=.1)
            except Empty:
                continue
            if self.reset_requested:
                self.reset_requested = False
                with self.lock:
                    self.tracker, self.bodies, self.measured_at = PersonTracker(), {}, {}
                    self.outlines_seen = {}
                    self.generation += 1
                try:
                    self.outlines.get_nowait()
                except Empty:
                    pass
            try:
                state = self.analyse(detector, frame, K, fps, baseline)
            except Exception as error:
                self.fail(f"Ошибка анализа кадра: {error}")
                break
            now = time.perf_counter()
            if previous is not None:
                fps = .8 * fps + .2 / max(now - previous, 1e-3) if fps else 1 / max(now - previous, 1e-3)
            previous = now
            with self.lock:
                self.state = state
        self.stop_event.set()
        measuring.join(timeout=5)

    def fail(self, message: str) -> None:
        self.error = message
        with self.lock:
            known = {track_id: body.estimate() for track_id, body in self.bodies.items()}
            self.state = WeighingState((), known, 0., message)
        self.status(message)

    def snapshot(self) -> WeighingState:
        with self.lock:
            return self.state

    def save(self, path) -> int:
        """Write numbers only: no images and nothing that identifies a person."""
        state = self.snapshot()
        people = [person_result(track_id, estimate) for track_id, estimate in sorted(state.known.items())
                  if estimate.samples]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(units="metres, litres, kilograms", validated_against_scales=False,
                                        app_version=VERSION, people=people), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return len(people)
