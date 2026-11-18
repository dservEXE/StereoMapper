"""Pig weighing from a photograph, a video file or the connected camera.

Each frame goes the same way: estimate the distance to every pixel, find the animals, number them,
measure each in space. Runs in its own thread and works only while its tab is open. A photograph
is analysed once. A video is analysed frame by frame, every one of them, as fast as the computer
manages; then it is written again with every frame marked: outlines, body lines and the final
weight of each animal. The camera is asked for a frame only when the previous one is done.

The depth model only guesses how large things are in an ordinary picture, and the guess can be
wrong several times over. A stereo pair, a segment of known length or one animal weighed on
scales fixes the scale; the last two apply to every animal seen, before and after.
"""
from dataclasses import dataclass, field, replace
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import Condition, Thread
import time
from typing import NamedTuple

import cv2
import numpy as np

from .pig_detection import distinct_outlines, overlap
from .pig_drawing import draw_marks, label_text, outline_contours
from .pig_measure import PigAccumulator, PigEstimate, PigShape, measure_pig, pig_shape
from .pig_scene import (Floor, assumed_focal, fit_floor, floor_candidates, hidden_behind, scene_points, segment_length,
                        stereo_scale)

MAX_SIDE = 1280
MEMORY_S = 5.
MIN_MATCH = .15
EDGE_MARGIN_PX = 3
IDLE_WAIT_S = .5
DEFAULT_FPS = 25.
READING_ROWS = 3
FLOOR_CHECK_POINTS = 300
FACTOR_LIMITS = (.1, 10.)
RATE_SMOOTHING = .2
RENDER_BATCH = 8
MARKED_SUFFIX = "_разметка"
PALETTE = ((80, 200, 255), (120, 230, 120), (255, 160, 90), (200, 130, 255), (90, 220, 220), (130, 130, 255))
START_HINT = "Откройте фото или видео или включите камеру."
CUT, NO_OUTLINE, HIDDEN = "обрезана краем кадра", "контур не найден", "закрыта другой свиньёй"
WEIGHED = "взвешена на весах"
# Where the scale comes from.
PICTURE, SEGMENT, STEREO, SEGMENT_FAILED = "picture", "segment", "stereo", "segment_failed"
WEIGHT, WEIGHT_FAILED = "weight", "weight_failed"
CALIBRATED = (SEGMENT, STEREO, WEIGHT)


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


class CameraFrame(NamedTuple):
    """A frame from the connected camera; z is real depth from a calibrated stereo pair, if there is one."""
    image: np.ndarray
    z: np.ndarray | None = None
    valid: np.ndarray | None = None
    focal: float | None = None
    centre: tuple | None = None


class Scene(NamedTuple):
    """What is known about the last analysed frame, kept so the scale can change without a new pass."""
    image: np.ndarray
    points: np.ndarray
    focal: float
    floor: Floor | None
    source: str


@dataclass
class Entry:
    """One animal in the current frame."""
    track: PigTrack
    box: tuple
    mask: np.ndarray
    shape: PigShape | None
    blocked: str
    contours: tuple
    note: str = ""


class Mark(NamedTuple):
    """What is drawn over one animal in one frame of the marked-up video, in analysed pixels."""
    track_id: int
    box: tuple
    color: tuple
    contours: tuple
    shape: PigShape | None


@dataclass(frozen=True)
class PigView:
    """One animal in the frame on screen."""
    track_id: int
    box: tuple
    color: tuple
    contours: tuple
    shape: PigShape | None
    estimate: PigEstimate | None
    note: str


@dataclass(frozen=True)
class PigRow:
    """One animal seen since the source was opened, for the table and the saved result."""
    track_id: int
    estimate: PigEstimate | None
    note: str
    in_view: bool


@dataclass(frozen=True)
class PigState:
    image: np.ndarray | None = None
    pigs: tuple = ()
    animals: tuple = ()
    message: str = ""
    kind: str | None = None
    playing: bool = False
    finished: bool = False
    position_s: float = 0.
    duration_s: float = 0.
    frame: int = 0
    frames: int = 0
    fps: float = 0.
    scale_source: str = PICTURE
    camera_height_m: float | None = None
    weighed_pig: int | None = None
    device: str = ""
    rendering: float | None = None
    video_file: str = ""
    ready: bool = False
    failed: bool = False
    serial: int = 0


def pig_result(pig) -> dict:
    estimate = pig.estimate
    return dict(label=pig.track_id, mass_kg=round(estimate.mass_kg, 1), mass_low_kg=round(estimate.mass_low_kg, 1),
                mass_high_kg=round(estimate.mass_high_kg, 1), length_m=round(estimate.length_m, 3),
                girth_m=round(estimate.girth_m, 3), samples=estimate.samples)


def write_result(path, animals, setup) -> int:
    """Write numbers only, no pictures. Returns how many animals were saved."""
    pigs = [pig_result(pig) for pig in animals if pig.estimate is not None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(setup, pigs=pigs), ensure_ascii=False, indent=2), encoding="utf-8")
    return len(pigs)


def result_folder(output) -> Path:
    return Path(output) / f"pigs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def open_writer(path, fps, width, height) -> tuple:
    """A video writer and the file it really writes to.

    The writer is asked for the path itself; should the codec library refuse its letters, a plain
    temporary name is used and the file is moved into place when done.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    spare = Path(tempfile.gettempdir()) / f"pigs_{os.getpid()}_{time.monotonic_ns()}.mp4"
    for candidate in (Path(path), spare):
        writer = cv2.VideoWriter(str(candidate), fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer, candidate
        writer.release()
        candidate.unlink(missing_ok=True)
    raise OSError("не удалось создать файл видео")


class PigSession(Thread):
    def __init__(self, detector_factory, output=None):
        super().__init__(daemon=True, name="pig-weighing")
        self.detector_factory = detector_factory
        self.output = None if output is None else Path(output)
        self.wake = Condition()
        self.pending = []
        self.submitted = None
        self.active = False
        self.stopping = False
        self.state = PigState(message="Загружаю модели поиска свиней…")
        # Everything below belongs to the thread.
        self.reference = None
        self.segment_factor = None
        self.known = None
        self.tracker = PigTracker()
        self.source = self.video = self.scene = self.path = None
        self.entries = []
        self.seen, self.notes, self.marks = {}, {}, {}
        self.frame_source = PICTURE
        self.kind = None
        self.playing = self.finished = False
        self.index, self.fps, self.frames, self.rate = 0, DEFAULT_FPS, 0, 0.
        self.shrink = 1.
        self.render = None
        self.video_file = ""

    # Called from the interface and from the camera thread.
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

    def wants_frame(self) -> bool:
        """The camera thread asks before preparing a frame, so nothing is computed in vain."""
        return self.active and self.kind == "camera" and self.submitted is None and not self.stopping

    def submit(self, image, z=None, valid=None, focal=None, centre=None) -> None:
        with self.wake:
            self.submitted = CameraFrame(image, z, valid, focal, centre)
            self.wake.notify()

    # The thread.
    def publish(self, **changes) -> None:
        self.state = replace(self.state, serial=self.state.serial + 1, **changes)

    def running(self) -> bool:
        return self.active and self.playing and self.video is not None

    def rendering(self) -> bool:
        return self.active and self.render is not None

    def frame_waiting(self) -> bool:
        return self.active and self.kind == "camera" and self.submitted is not None

    def run(self) -> None:
        try:
            detector = self.detector_factory(lambda text: self.publish(message=text))
        except Exception as error:  # noqa: BLE001 - a missing or damaged model must reach the interface.
            self.publish(message=f"Не удалось загрузить модели: {error}", failed=True)
            return
        label = getattr(detector, "device_label", "")
        self.publish(message=START_HINT, device=label, ready=True)
        try:
            while self.step(detector):
                pass
        finally:
            self.stop_render()
            self.close_video()

    def step(self, detector) -> bool:
        with self.wake:
            while (not self.stopping and not self.pending and not self.running() and not self.frame_waiting()
                   and not self.rendering()):
                self.wake.wait(IDLE_WAIT_S)
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
                    self.analyse(detector, frame, (self.index - 1) / self.fps)
            elif self.frame_waiting():
                with self.wake:
                    camera, self.submitted = self.submitted, None
                self.analyse(detector, camera.image, time.monotonic(), camera)
            elif self.rendering():
                self.render_step()
        except Exception as error:  # noqa: BLE001 - one bad frame must not end the session.
            self.publish(message=f"Кадр не обработан: {error}")
        return True

    def handle(self, name, *values) -> np.ndarray | None:
        """Apply one command; return a still picture to analyse now, if it calls for one."""
        if name == "reference":
            self.reference, self.segment_factor = values[0], None
            self.calibrate_segment()
            self.show()
        elif name == "known":
            self.known = values[0]
            self.show()
        elif name == "play":
            if values[0] and self.finished:
                self.rewind()
            self.playing = bool(values[0]) and self.video is not None
            self.publish(playing=self.playing)
        elif name == "rewind" and self.video is not None:
            self.rewind()
        elif name == "render":
            self.start_render()
            self.show()
        elif name == "reset":
            if self.kind == "video" and self.video is not None:
                self.rewind()
                return None
            self.forget_animals()
            if self.kind == "photo":
                return self.source
            self.show()
        elif name == "photo":
            self.close_video()
            image = read_image(values[0])
            if image is None:
                self.publish(message="Не удалось открыть фото: файл повреждён или это не изображение.")
                return None
            self.kind, self.source = "photo", fitted(image)
            self.forget_animals()
            return self.source
        elif name == "video":
            self.open_video(values[0])
        elif name == "camera":
            self.close_video()
            self.kind, self.source, self.scene = "camera", None, None
            self.forget_animals()
            with self.wake:
                self.submitted = None
            self.publish(kind="camera", playing=False, finished=False, image=None, pigs=(), animals=(), position_s=0.,
                         duration_s=0., frame=0, frames=0, rendering=None, video_file="", message="Жду кадр с камеры…")
        return None

    def forget_animals(self) -> None:
        """New numbers from 1, nothing measured."""
        self.tracker = PigTracker()
        self.entries, self.seen, self.notes, self.marks = [], {}, {}, {}
        self.rate = 0.

    def calibrate_segment(self) -> None:
        """How many times the marked segment is longer than it is seen in the frame on screen now."""
        if self.reference is None or self.scene is None:
            return
        segment, true_length = self.reference
        seen = segment_length(self.scene.points, segment)
        factor = true_length / seen if seen else 0.
        self.segment_factor = factor if FACTOR_LIMITS[0] <= factor <= FACTOR_LIMITS[1] else 0.

    def open_video(self, path) -> None:
        self.close_video()
        video = cv2.VideoCapture(str(path))
        if not video.isOpened():
            video.release()
            self.publish(message="Не удалось открыть видео: формат не поддерживается или файл повреждён.")
            return
        fps = video.get(cv2.CAP_PROP_FPS)
        self.video, self.path, self.kind, self.source, self.scene = video, Path(path), "video", None, None
        self.fps = float(fps) if fps and np.isfinite(fps) and fps > 0 else DEFAULT_FPS
        self.frames = max(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.forget_animals()
        self.index, self.playing, self.finished, self.shrink, self.video_file = 0, True, False, 1., ""
        self.publish(kind="video", playing=True, finished=False, image=None, pigs=(), animals=(), position_s=0.,
                     duration_s=self.frames / self.fps, frame=0, frames=self.frames, rendering=None, video_file="",
                     message="Видео открыто, ищу свиней…")

    def rewind(self) -> None:
        self.stop_render()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.forget_animals()
        self.index, self.playing, self.finished, self.video_file = 0, True, False, ""
        self.publish(playing=True, finished=False, position_s=0., frame=0, rendering=None, video_file="")

    def close_video(self) -> None:
        self.stop_render()
        if self.video is not None:
            self.video.release()
        self.video, self.playing, self.finished = None, False, False

    def read_frame(self) -> np.ndarray | None:
        """The next frame of the video: every one is analysed, none is skipped."""
        ok, frame = self.video.read()
        if not ok:
            self.playing, self.finished = False, True
            self.frames = self.index
            self.start_render()
            self.show()
            return None
        self.index += 1
        small = fitted(frame)
        self.shrink = frame.shape[1] / small.shape[1]
        return small

    def analyse(self, detector, image, now, camera=None) -> None:
        started = time.perf_counter()
        depth = detector.depth(image)
        focal, centre, source = assumed_focal(image.shape), None, PICTURE
        if camera is not None and camera.z is not None and camera.focal:
            focal, centre = camera.focal, camera.centre
            scale = stereo_scale(depth, camera.z, camera.valid)
            if scale is not None:
                depth, source = depth * scale, STEREO
        points = scene_points(depth, focal, centre)
        boxes, scores = detector.detect(image)
        order = reading_order(boxes, image.shape[0])
        boxes, scores = boxes[order], scores[order]
        found = distinct_outlines(detector.outlines(image, boxes), scores)
        boxes, masks = boxes[[index for index, _ in found]], [mask for _, mask in found]
        tracks = self.tracker.update(boxes, now)
        animals = [points[mask][::max(1, int(mask.sum()) // FLOOR_CHECK_POINTS)].astype(np.float64) for mask in masks]
        floor = fit_floor(points, floor_candidates(masks), animals)
        height, width = image.shape[:2]
        self.entries = []
        for track, box, mask, hidden in zip(tracks, boxes, masks, hidden_behind(masks, points)):
            cut = bool(box[0] <= EDGE_MARGIN_PX or box[1] <= EDGE_MARGIN_PX
                       or box[2] >= width - EDGE_MARGIN_PX or box[3] >= height - EDGE_MARGIN_PX)
            self.entries.append(Entry(track, tuple(float(value) for value in box), mask, pig_shape(mask),
                                      CUT if cut else HIDDEN if hidden else "", outline_contours(mask)))
            self.seen[track.track_id] = track
        self.scene = Scene(image, points, focal, floor, source)
        self.frame_source = source
        if self.reference is not None and self.segment_factor is None:
            self.calibrate_segment()
        self.measure()
        if self.kind == "video":
            self.marks[self.index - 1] = tuple(Mark(entry.track.track_id, entry.box, entry.track.color, entry.contours,
                                                    entry.shape) for entry in self.entries)
        rate = 1 / max(1e-3, time.perf_counter() - started)
        self.rate = rate if not self.rate else self.rate + RATE_SMOOTHING * (rate - self.rate)
        self.show()

    def measure(self) -> None:
        """Add one measurement of the current frame to every animal that can be measured in it."""
        scene = self.scene
        for entry in self.entries:
            if entry.blocked:
                entry.note = entry.blocked
            elif entry.shape is None:
                entry.note = NO_OUTLINE
            else:
                sample, entry.note = measure_pig(entry.shape, scene.points, entry.mask, scene.focal, scene.floor)
                if sample is not None:
                    entry.track.accumulator.add(sample)
            self.notes[entry.track.track_id] = entry.note

    def scale(self) -> tuple:
        """(length factor, where the scale comes from, the animal weighed on scales if it set the scale)."""
        factor, source = 1., self.frame_source
        if self.reference is not None and self.segment_factor is not None:
            factor, source = (self.segment_factor, SEGMENT) if self.segment_factor else (1., SEGMENT_FAILED)
        if self.known is not None:
            track_id, mass = self.known
            track = self.seen.get(track_id)
            weighed = track.accumulator.estimate() if track is not None else None
            if weighed is None or weighed.mass_kg <= 0:
                return factor, WEIGHT_FAILED, None
            return float((mass / weighed.mass_kg) ** (1 / 3)), WEIGHT, track_id
        return factor, source, None

    def estimates(self) -> tuple:
        """Every animal seen, at the current scale: ({number: estimate or None}, scale source, factor, weighed one)."""
        factor, source, weighed = self.scale()
        found = {}
        for key, track in self.seen.items():
            estimate = track.accumulator.estimate(source in CALIBRATED)
            found[key] = None if estimate is None else estimate.scaled(factor)
        return found, source, factor, weighed

    def show(self) -> None:
        """Publish the last analysed frame with what is known about every animal in it."""
        estimates, source, factor, weighed = self.estimates()
        pigs = tuple(PigView(entry.track.track_id, entry.box, entry.track.color, entry.contours, entry.shape,
                             estimates.get(entry.track.track_id), entry.note) for entry in self.entries)
        in_view = {pig.track_id for pig in pigs}
        animals = tuple(PigRow(key, estimate, WEIGHED if key == weighed else self.notes.get(key, ""), key in in_view)
                        for key, estimate in estimates.items())
        measured = sum(estimate is not None for estimate in estimates.values())
        video = self.kind == "video"
        if self.scene is None:
            message = self.state.message
        elif video and self.finished:
            message = f"Видео обработано целиком, кадров: {self.frames}. Взвешено свиней: {measured}."
        elif video:
            message = f"Обрабатываю видео: кадр {self.index} из {max(self.frames, self.index)}."
        elif not pigs:
            message = "Свиней на кадре не найдено."
        else:
            message = f"Найдено свиней: {len(pigs)}, измерено: {sum(pig.estimate is not None for pig in pigs)}."
        floor = None if self.scene is None else self.scene.floor
        self.publish(image=None if self.scene is None else self.scene.image, pigs=pigs, animals=animals, message=message,
                     kind=self.kind, playing=self.playing, finished=self.finished,
                     position_s=self.index / self.fps if video else 0., duration_s=self.frames / self.fps if video else 0.,
                     frame=self.index if video else 0, frames=self.frames if video else 0, scale_source=source,
                     camera_height_m=None if floor is None else floor.offset * factor, weighed_pig=weighed,
                     fps=self.rate if self.kind in ("video", "camera") else 0., video_file=self.video_file,
                     rendering=None if self.render is None else self.state.rendering or 0.)

    # The marked-up video.
    def start_render(self) -> None:
        """Write the video again, every frame marked, with the weights as they are now."""
        self.stop_render()
        if self.kind == "video" and self.finished and self.marks and self.path is not None and self.output is not None:
            self.render = self.render_frames()
            self.state = replace(self.state, rendering=0.)

    def stop_render(self) -> None:
        if self.render is not None:
            self.render.close()
            self.render = None

    def render_step(self) -> None:
        try:
            done = next(self.render)
        except StopIteration as finished:
            self.render = None
            self.video_file = str(finished.value or "")
            self.show()
        except Exception as error:  # noqa: BLE001 - a full disk must not end the session.
            self.stop_render()
            self.publish(rendering=None, message=f"Не удалось записать видео с разметкой: {error}")
        else:
            self.publish(rendering=min(1., done / max(1, self.frames)))

    def render_frames(self):
        """A generator that writes a batch of frames each time it is advanced and returns the file."""
        estimates, source, _, _ = self.estimates()
        texts = {key: label_text(key, estimate) for key, estimate in estimates.items()}
        folder = result_folder(self.output)
        target = folder / f"{self.path.stem}{MARKED_SUFFIX}.mp4"
        capture = cv2.VideoCapture(str(self.path))
        writer = written = None
        complete = False
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    folder.mkdir(parents=True, exist_ok=True)
                    writer, written = open_writer(target, self.fps, frame.shape[1], frame.shape[0])
                marks = self.marks.get(index, ())
                draw_marks(frame, marks, [texts.get(mark.track_id, f"#{mark.track_id}") for mark in marks], self.shrink)
                writer.write(frame)
                index += 1
                if index % RENDER_BATCH == 0:
                    yield index
            complete = writer is not None
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if not complete and written is not None:
                Path(written).unlink(missing_ok=True)
        if written != target:
            shutil.move(str(written), str(target))
        from .version import VERSION
        rows = [PigRow(key, estimate, "", False) for key, estimate in estimates.items()]
        write_result(folder / "result.json", rows, dict(version=VERSION, source="video", scale=source,
                                                        video=self.path.name, frames=index))
        return target
