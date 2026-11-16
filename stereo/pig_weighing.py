"""Pig weighing from a photograph or a video file: find the animals, number them, measure each.

Runs in its own thread and works only while its tab is open. A photograph is analysed once. A
video plays at its own speed and is analysed as fast as the computer manages, dropping frames.
"""
from dataclasses import dataclass, field, replace
import itertools
import json
from pathlib import Path
from threading import Condition, Thread
import time

import cv2
import numpy as np

from .pig_detection import overlap
from .pig_measure import PigAccumulator, PigEstimate, PigShape, pig_shape

MAX_SIDE = 1280
MEMORY_S = 5.
MIN_MATCH = .15
EDGE_MARGIN_PX = 3
IDLE_WAIT_S = .5
DEFAULT_FPS = 25.
READING_ROWS = 3
PALETTE = ((80, 200, 255), (120, 230, 120), (255, 160, 90), (200, 130, 255), (90, 220, 220), (130, 130, 255))
START_HINT = "Откройте фото или видео со свиньями."
NO_SCALE_HINT = "Отметьте на кадре отрезок известной длины, чтобы получить размеры и вес."
CUT, NO_OUTLINE, BAD_POSE = "обрезана краем кадра", "контур не найден", "поза не подходит для замера"


def read_image(path) -> np.ndarray | None:
    """cv2.imread cannot open a path with Cyrillic letters on Windows."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def fitted(image, limit=MAX_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    factor = limit / max(height, width)
    if factor >= 1:
        return image
    return cv2.resize(image, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)


def reading_order(boxes, height) -> np.ndarray:
    """Left to right in a few rows, so that numbers on a photograph are easy to find."""
    if not len(boxes):
        return np.zeros(0, int)
    centre_x, centre_y = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    return np.lexsort((centre_x, (centre_y * READING_ROWS // max(1, height)).astype(int)))


@dataclass
class PigTrack:
    track_id: int
    box: tuple
    last_seen: float
    accumulator: PigAccumulator = field(default_factory=PigAccumulator)

    @property
    def color(self) -> tuple:
        return PALETTE[(self.track_id - 1) % len(PALETTE)]


def match_score(known, box) -> float:
    size = max(box[2] - box[0], box[3] - box[1], 1.)
    shift = np.hypot((known[0] + known[2] - box[0] - box[2]) / 2, (known[1] + known[3] - box[1] - box[3]) / 2)
    return overlap(known, box) + .5 * max(0., 1 - shift / size)


class PigTracker:
    """Keeps one number on each animal while it stays in view, by where it is."""

    def __init__(self) -> None:
        self.tracks = {}
        self.ids = itertools.count(1)

    def update(self, boxes, now) -> list:
        self.tracks = {key: track for key, track in self.tracks.items() if now - track.last_seen <= MEMORY_S}
        pairs = sorted(((match_score(track.box, box), track.track_id, index)
                        for index, box in enumerate(boxes) for track in self.tracks.values()), reverse=True)
        assigned, used = {}, set()
        for score, track_id, index in pairs:
            if score < MIN_MATCH:
                break
            if track_id not in used and index not in assigned:
                assigned[index] = track_id
                used.add(track_id)
        result = []
        for index, box in enumerate(boxes):
            track = self.tracks.get(assigned.get(index))
            if track is None:
                track = PigTrack(next(self.ids), tuple(box), now)
                self.tracks[track.track_id] = track
            else:
                track.box, track.last_seen = tuple(box), now
            result.append(track)
        return result


@dataclass(frozen=True)
class PigView:
    track_id: int
    box: tuple
    color: tuple
    outline: np.ndarray | None
    shape: PigShape | None
    estimate: PigEstimate | None
    note: str


@dataclass(frozen=True)
class PigState:
    image: np.ndarray | None = None
    pigs: tuple = ()
    message: str = ""
    kind: str | None = None
    playing: bool = False
    finished: bool = False
    position_s: float = 0.
    duration_s: float = 0.
    fps: float = 0.
    ready: bool = False
    failed: bool = False
    serial: int = 0


def pig_result(pig) -> dict:
    estimate = pig.estimate
    return dict(label=pig.track_id, mass_kg=round(estimate.mass_kg, 1), mass_low_kg=round(estimate.mass_low_kg, 1),
                mass_high_kg=round(estimate.mass_high_kg, 1), length_m=round(estimate.length_m, 3),
                thickness_m=round(estimate.thickness_m, 3), girth_m=round(estimate.girth_m, 3), samples=estimate.samples)


def write_result(path, state, setup) -> int:
    """Write numbers only, no pictures. Returns how many animals were saved."""
    pigs = [pig_result(pig) for pig in state.pigs if pig.estimate is not None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(setup, pigs=pigs), ensure_ascii=False, indent=2), encoding="utf-8")
    return len(pigs)


class PigSession(Thread):
    def __init__(self, detector_factory):
        super().__init__(daemon=True, name="pig-weighing")
        self.detector_factory = detector_factory
        self.wake = Condition()
        self.pending = []
        self.active = False
        self.stopping = False
        self.state = PigState(message="Загружаю модели поиска свиней…")
        # Everything below belongs to the thread.
        self.scale, self.camera_height, self.view = None, None, "top"
        self.tracker = PigTracker()
        self.source = self.frame = self.video = self.clock = None
        self.entries = []
        self.kind = None
        self.playing = self.finished = False
        self.index, self.fps, self.frames, self.rate = 0, DEFAULT_FPS, 0, 0.

    # Called from the interface.
    def post(self, *command) -> None:
        with self.wake:
            self.pending.append(command)
            self.wake.notify()

    def set_active(self, active) -> None:
        with self.wake:
            self.active = bool(active)
            self.wake.notify()

    def stop(self) -> None:
        with self.wake:
            self.stopping = True
            self.wake.notify()

    # The thread.
    def publish(self, **changes) -> None:
        self.state = replace(self.state, serial=self.state.serial + 1, **changes)

    def running(self) -> bool:
        return self.active and self.playing and self.video is not None

    def due(self) -> bool:
        return self.running() and (self.clock is None or time.perf_counter() >= self.clock + self.index / self.fps)

    def run(self) -> None:
        try:
            detector = self.detector_factory(lambda text: self.publish(message=text))
        except Exception as error:  # noqa: BLE001 - a missing or damaged model must reach the interface.
            self.publish(message=f"Не удалось загрузить модели: {error}", failed=True)
            return
        self.publish(message=f"{START_HINT} Считает {detector.device_label}.", ready=True)
        try:
            while self.step(detector):
                pass
        finally:
            self.close_video()

    def step(self, detector) -> bool:
        with self.wake:
            while not self.stopping and not self.pending and not self.due():
                if not self.running():
                    # After a pause the video continues from where it stopped, not from where the clock went.
                    self.clock = None
                    self.wake.wait(IDLE_WAIT_S)
                else:
                    self.wake.wait(max(.005, self.clock + self.index / self.fps - time.perf_counter()))
            if self.stopping:
                return False
            commands, self.pending = self.pending, []
        fresh = None
        for command in commands:
            image = self.handle(*command)
            fresh = image if image is not None else fresh
        try:
            if fresh is not None:
                self.analyse(detector, fresh, 0.)
            elif self.running():
                frame = self.read_frame()
                if frame is not None:
                    self.analyse(detector, frame, self.index / self.fps)
        except Exception as error:  # noqa: BLE001 - one bad frame must not end the session.
            self.publish(message=f"Кадр не обработан: {error}")
        return True

    def handle(self, name, *values) -> np.ndarray | None:
        """Apply one command; return a still picture to analyse now, if it calls for one."""
        if name == "scale":
            self.scale, self.camera_height, self.view = values
            self.show()
        elif name == "play":
            if values[0] and self.finished:
                self.rewind()
            self.playing = bool(values[0]) and self.video is not None
            self.publish(playing=self.playing)
        elif name == "rewind" and self.video is not None:
            self.rewind()
        elif name == "reset":
            self.tracker = PigTracker()
            if self.kind == "photo":
                return self.source
            self.entries = []
            self.show()
        elif name == "photo":
            self.close_video()
            image = read_image(values[0])
            if image is None:
                self.publish(message="Не удалось открыть фото: файл повреждён или это не изображение.")
                return None
            self.kind, self.source, self.tracker = "photo", fitted(image), PigTracker()
            return self.source
        elif name == "video":
            self.open_video(values[0])
        return None

    def open_video(self, path) -> None:
        self.close_video()
        video = cv2.VideoCapture(str(path))
        if not video.isOpened():
            video.release()
            self.publish(message="Не удалось открыть видео: формат не поддерживается или файл повреждён.")
            return
        fps = video.get(cv2.CAP_PROP_FPS)
        self.video, self.kind, self.source = video, "video", None
        self.fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else DEFAULT_FPS
        self.frames = max(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.tracker, self.entries, self.frame = PigTracker(), [], None
        self.index, self.clock, self.playing, self.finished = 0, None, True, False
        self.publish(kind="video", playing=True, finished=False, image=None, pigs=(), position_s=0.,
                     duration_s=self.frames / self.fps, message="Видео открыто, ищу свиней…")

    def rewind(self) -> None:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.tracker, self.entries = PigTracker(), []
        self.index, self.clock, self.playing, self.finished = 0, None, True, False
        self.publish(playing=True, finished=False, position_s=0.)

    def close_video(self) -> None:
        if self.video is not None:
            self.video.release()
        self.video, self.playing, self.finished = None, False, False

    def read_frame(self) -> np.ndarray | None:
        """The frame that is due now; the ones the analysis had no time for are skipped."""
        now = time.perf_counter()
        if self.clock is None:
            self.clock = now - self.index / self.fps
        target = max(self.index, int((now - self.clock) * self.fps))
        while self.index < target and self.video.grab():
            self.index += 1
        ok, frame = self.video.read()
        if not ok:
            self.playing, self.finished = False, True
            self.publish(playing=False, finished=True, message="Видео закончилось. Нажмите «Сначала», чтобы повторить.")
            return None
        self.index += 1
        return fitted(frame)

    def analyse(self, detector, image, now) -> None:
        started = time.perf_counter()
        boxes, _ = detector.detect(image)
        boxes = boxes[reading_order(boxes, image.shape[0])]
        masks = detector.outlines(image, boxes)
        tracks = self.tracker.update(boxes, now)
        height, width = image.shape[:2]
        self.entries = []
        for track, box, mask in zip(tracks, boxes, masks):
            cut = (box[0] <= EDGE_MARGIN_PX or box[1] <= EDGE_MARGIN_PX
                   or box[2] >= width - EDGE_MARGIN_PX or box[3] >= height - EDGE_MARGIN_PX)
            shape = pig_shape(mask)
            if cut:
                note = CUT
            elif shape is None:
                note = NO_OUTLINE
            else:
                note = "" if track.accumulator.add(shape) else BAD_POSE
            self.entries.append((track, tuple(float(value) for value in box), mask, shape, note))
        self.frame = image
        self.rate = 1 / max(1e-3, time.perf_counter() - started)
        self.show()

    def show(self) -> None:
        """Publish the last analysed frame with estimates at the current scale."""
        pigs = tuple(PigView(track.track_id, box, track.color, mask, shape,
                             track.accumulator.estimate(self.scale, self.camera_height, self.view), note)
                     for track, box, mask, shape, note in self.entries)
        if self.frame is None:
            message = self.state.message
        elif not pigs:
            message = "Свиней на кадре не найдено."
        elif not self.scale:
            message = f"Найдено свиней: {len(pigs)}. {NO_SCALE_HINT}"
        else:
            measured = sum(pig.estimate is not None for pig in pigs)
            message = f"Найдено свиней: {len(pigs)}, измерено: {measured}."
        if self.frame is not None and self.finished:
            message += " Видео закончилось."
        self.publish(image=self.frame, pigs=pigs, message=message, kind=self.kind, playing=self.playing,
                     finished=self.finished, position_s=self.index / self.fps if self.kind == "video" else 0.,
                     duration_s=self.frames / self.fps if self.kind == "video" else 0.,
                     fps=self.rate if self.kind == "video" else 0.)
