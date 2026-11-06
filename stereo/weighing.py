"""Background person weighing: never slows capture or depth, always works on the newest frame."""
from dataclasses import dataclass
import json
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time

import numpy as np

from .body_measure import BodyAccumulator, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, measure_person
from .people_tracking import PersonTracker


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

    def submit(self, frame, K) -> None:
        try:
            self.frames.put_nowait((frame, K))
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            self.frames.put_nowait((frame, K))

    def reset(self) -> None:
        self.reset_requested = True

    def analyse(self, detector, frame, K, fps) -> WeighingState:
        now = time.perf_counter()
        observations = detector.detect(frame.left)
        depths = [torso_depth(frame, observation) for observation in observations]
        views, waiting = [], []
        for track, observation, index in self.tracker.update(frame.left, observations, depths, now):
            with self.lock:
                estimate = self.bodies.setdefault(track.track_id, BodyAccumulator()).estimate()
                outline = self.outlines_seen.get(track.track_id)
            views.append(PersonView(track.track_id, track.color, observation.box, observation.keypoints,
                                    observation.keypoint_scores, depths[index], estimate, outline))
            waiting.append((track.track_id, observation))
        self.request_outline(frame, K, waiting)
        people = tuple(sorted(views, key=lambda view: view.track_id))
        with self.lock:
            known = {track_id: body.estimate() for track_id, body in self.bodies.items()}
        message = f"В кадре людей: {len(people)}" if people else "Людей в кадре не видно"
        return WeighingState(people, known, fps, message, frame.left)

    def request_outline(self, frame, K, waiting) -> None:
        """Hand the least recently measured person to the slower outline thread, if it is free."""
        if not waiting or self.outlines.full():
            return
        track_id, observation = min(waiting, key=lambda item: self.measured_at.get(item[0], 0.))
        self.measured_at[track_id] = time.perf_counter()
        self.outlines.put((frame, K, track_id, observation))

    def measure_loop(self, detector) -> None:
        """Outlines take about a second each, so labels must not wait for them."""
        while not self.stop_event.is_set():
            try:
                frame, K, track_id, observation = self.outlines.get(timeout=.1)
            except Empty:
                continue
            try:
                mask = detector.outline(frame.left, observation)
                sample = measure_person(frame.points, frame.valid, K, observation, mask)
            except Exception as error:
                self.fail(f"Ошибка измерения: {error}")
                return
            with self.lock:
                body = self.bodies.get(track_id)
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
                frame, K = self.frames.get(timeout=.1)
            except Empty:
                continue
            if self.reset_requested:
                self.reset_requested = False
                with self.lock:
                    self.tracker, self.bodies, self.measured_at = PersonTracker(), {}, {}
                    self.outlines_seen = {}
            try:
                state = self.analyse(detector, frame, K, fps)
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
            self.state = WeighingState((), {}, 0., message)
        self.status(message)

    def snapshot(self) -> WeighingState:
        with self.lock:
            return self.state

    def save(self, path) -> int:
        """Write numbers only: no images and nothing that identifies a person."""
        state = self.snapshot()
        people = [dict(label=track_id, samples=e.samples, height_m=e.height_m, widths_m=e.widths_m,
                       side_widths_m=e.side_widths_m, lengths_m=e.lengths_m, volume_l=e.volume_l,
                       mass_kg=e.mass_kg, mass_low_kg=e.mass_low_kg, mass_high_kg=e.mass_high_kg,
                       side_observed=e.observed_side, views=e.view_coverage,
                       slices_front_m=e.slices_front_m, slices_side_m=e.slices_side_m)
                  for track_id, e in sorted(state.known.items())]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(units="metres, litres, kilograms", validated_against_scales=False,
                                        people=people), ensure_ascii=False, indent=2), encoding="utf-8")
        return len(people)
