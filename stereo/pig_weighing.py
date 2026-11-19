"""Pig weighing from a photograph, a video file or the connected camera.

Each analysed frame goes the same way: estimate the distance to every pixel, find the animals,
measure each in space. The neural networks run in a thread of their own and work only while the
tab is open. A photograph is analysed once; the camera is asked for a frame only when the
previous one is done.

A video plays frame by frame, every one of them, at its own rate. The networks take the newest
frame whenever they are free; meanwhile the outlines are carried along with the bodies by optical
flow, so every frame shown has its outlines, and a pass that ends later is carried forward to the
frame on screen. When the computer is slow, the video slows down rather than let the outlines run
more than a second ahead of the networks. At the end it is written again with every frame marked.

The depth model only guesses how large things are in an ordinary picture, and the guess can be
wrong several times over. A stereo pair, a segment of known length or one animal weighed on
scales fixes the scale; the last two apply to every animal seen, before and after. Pigs weighed
earlier and kept in the pig database correct the guess of later videos.
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

from .pig_detection import WORKERS, distinct_outlines, overlap
from .pig_drawing import draw_marks, label_text, outline_contours
from .pig_measure import PigAccumulator, PigEstimate, PigSample, PigShape, measure_pig, pig_shape
from .pig_scene import (Floor, assumed_focal, fit_floor, floor_candidates, hidden_behind, scene_points, segment_length,
                        stereo_scale)
from .pig_tracking import IDENTITY, Follower, carried, contour_box, follow, moved_contours, moved_shape, tracking_gray

MAX_SIDE = 1280
MEMORY_S = 5.
MIN_MATCH = .15
EDGE_MARGIN_PX = 3
IDLE_WAIT_S = .5
PLAY_WAIT_S = .05
DEFAULT_FPS = 25.
# How far, in seconds of video, the outlines may be carried before the networks look again.
MAX_LAG_S = 1.
# Below the networks' pace the video goes evenly slower, rather than run a second and stand still.
PACE_MARGIN = .9
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
WEIGHT, WEIGHT_FAILED, LEARNED = "weight", "weight_failed", "learned"
CALIBRATED = (SEGMENT, STEREO, WEIGHT, LEARNED)


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


class Found(NamedTuple):
    """One animal as the networks found it in one frame, measured there if it could be."""
    box: tuple
    contours: tuple
    shape: PigShape | None
    sample: PigSample | None
    note: str


class Analysis(NamedTuple):
    scene: Scene
    animals: tuple
    seconds: float


def analyse_frame(detector, image, camera=None) -> Analysis:
    """Distances, animals, outlines and one measurement of each: everything the networks give for one frame."""
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
    animals = [points[mask][::max(1, int(mask.sum()) // FLOOR_CHECK_POINTS)].astype(np.float64) for mask in masks]
    floor = fit_floor(points, floor_candidates(masks), animals)
    height, width = image.shape[:2]
    result = []
    for box, mask, hidden in zip(boxes, masks, hidden_behind(masks, points)):
        cut = bool(box[0] <= EDGE_MARGIN_PX or box[1] <= EDGE_MARGIN_PX
                   or box[2] >= width - EDGE_MARGIN_PX or box[3] >= height - EDGE_MARGIN_PX)
        shape, sample = pig_shape(mask), None
        if cut or hidden:
            note = CUT if cut else HIDDEN
        elif shape is None:
            note = NO_OUTLINE
        else:
            sample, note = measure_pig(shape, points, mask, focal, floor)
        result.append(Found(tuple(float(value) for value in box), outline_contours(mask), shape, sample, note))
    return Analysis(Scene(image, points, focal, floor, source), tuple(result), time.perf_counter() - started)


class Job(NamedTuple):
    generation: int
    index: int
    image: np.ndarray
    camera: CameraFrame | None = None

    @property
    def key(self) -> tuple:
        return self.generation, self.index


class Networks:
    """Threads running the neural networks on several frames at once, so the card and every core stay busy.

    Results are kept by frame; the session takes them in the order it asked, so the numbers and
    the outlines go forward in time even when a later frame is done first.
    """

    def __init__(self, detector, count, notify):
        self.detector, self.notify = detector, notify
        self.lock = Condition()
        self.jobs, self.results = [], {}
        self.count, self.running = max(1, int(count)), 0
        self.generation = 0
        self.stopping = False
        self.threads = [Thread(target=self.serve, daemon=True, name=f"pig-networks-{index}") for index in range(self.count)]
        for thread in self.threads:
            thread.start()

    def free(self) -> bool:
        """Whether a thread would start on a frame right away."""
        with self.lock:
            return len(self.jobs) + self.running < self.count

    def submit(self, job) -> None:
        with self.lock:
            self.jobs.append(job)
            self.lock.notify()

    def done(self, key) -> bool:
        with self.lock:
            return key in self.results

    def take(self, key):
        """The analysis or the error it raised, once it is done."""
        with self.lock:
            return self.results.pop(key, None)

    def forget(self, generation) -> None:
        """Drop what was asked for an earlier source: it will never be shown."""
        with self.lock:
            self.generation = generation
            self.jobs = [job for job in self.jobs if job.generation == generation]
            self.results = {key: value for key, value in self.results.items() if key[0] == generation}

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            self.lock.notify_all()

    def serve(self) -> None:
        while True:
            with self.lock:
                while not self.jobs and not self.stopping:
                    self.lock.wait()
                if self.stopping:
                    return
                job = self.jobs.pop(0)
                self.running += 1
            try:
                outcome = analyse_frame(self.detector, job.image, job.camera)
            except Exception as error:  # noqa: BLE001 - one bad frame must not end the session.
                outcome = error
            with self.lock:
                if job.generation == self.generation:
                    self.results[job.key] = outcome
                self.running -= 1
            self.notify()


@dataclass
class Entry:
    """One animal on screen now: as found by the networks, and where the motion since has taken it."""
    track: PigTrack
    found: Found
    follower: Follower | None = None

    @property
    def contours(self) -> tuple:
        return self.found.contours if self.follower is None else self.follower.now_contours

    @property
    def shape(self) -> PigShape | None:
        return self.found.shape if self.follower is None else self.follower.now_shape

    @property
    def box(self) -> tuple:
        return self.found.box if self.follower is None else self.follower.now_box


class Mark(NamedTuple):
    """What is drawn over one animal in one frame of the marked-up video.

    The outline is shared with every frame carried from the same pass; only the motion is kept.
    """
    track_id: int
    color: tuple
    base_contours: tuple
    base_shape: PigShape | None
    base_box: tuple
    matrix: np.ndarray

    @property
    def contours(self) -> tuple:
        return moved_contours(self.base_contours, self.matrix)

    @property
    def shape(self) -> PigShape | None:
        return moved_shape(self.base_shape, self.matrix)

    @property
    def box(self) -> tuple:
        return contour_box(self.contours, self.base_box)


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
    """One animal seen since the source was opened, for the table, the saved result and the pig database.

    raw is the estimate at the scale the pictures themselves give, before any correction.
    """
    track_id: int
    estimate: PigEstimate | None
    note: str
    in_view: bool
    raw: PigEstimate | None = None
    raw_source: str = PICTURE


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
    video_fps: float = 0.
    analysis_fps: float = 0.
    scale_source: str = PICTURE
    camera_height_m: float | None = None
    weighed_pig: int | None = None
    learned_pigs: int = 0
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
                girth_m=round(estimate.girth_m, 3), samples=estimate.samples,
                volume_l=None if estimate.volume_l is None else round(estimate.volume_l, 1))


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


def smoothed(previous, value) -> float:
    return value if not previous else previous + RATE_SMOOTHING * (value - previous)


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
        self.networks = None
        self.asked_for = []
        self.state = PigState(message="Загружаю модели поиска свиней…")
        # Everything below belongs to the thread.
        self.reference = None
        self.segment_factor = None
        self.known = None
        self.learned = None
        self.tracker = PigTracker()
        self.source = self.video = self.scene = self.path = None
        self.entries = []
        self.seen, self.notes, self.marks = {}, {}, {}
        self.frame_source = PICTURE
        self.kind = None
        self.playing = self.finished = self.draining = False
        self.index, self.fps, self.frames = 0, DEFAULT_FPS, 0
        self.rate = self.analysis_rate = self.merge_gap = 0.
        self.last_merge = 0.
        self.generation = 0
        self.asked = -1
        self.camera_frames = 0
        self.current = self.gray = None
        self.grays = {}
        self.track_scale = 1.
        self.next_due = self.last_shown = 0.
        self.shrink = 1.
        self.render = None
        self.video_file = ""

    # Called from the interface and from the camera thread.
    def post(self, *command) -> None:
        with self.wake:
            self.pending.append(command)
            self.wake.notify_all()

    def set_active(self, active) -> None:
        with self.wake:
            self.active = bool(active)
            self.wake.notify_all()

    def stop(self) -> None:
        with self.wake:
            self.stopping = True
            self.wake.notify_all()

    def wants_frame(self) -> bool:
        """The camera thread asks before preparing a frame, so nothing is computed in vain."""
        free = self.networks is not None and self.networks.free()
        return self.active and self.kind == "camera" and self.submitted is None and free and not self.stopping

    def submit(self, image, z=None, valid=None, focal=None, centre=None) -> None:
        with self.wake:
            self.submitted = CameraFrame(image, z, valid, focal, centre)
            self.wake.notify_all()

    def networks_done(self) -> None:
        with self.wake:
            self.wake.notify_all()

    # The thread.
    def publish(self, **changes) -> None:
        self.state = replace(self.state, serial=self.state.serial + 1, **changes)

    def running(self) -> bool:
        return self.active and (self.playing or self.draining) and self.video is not None

    def rendering(self) -> bool:
        return self.active and self.render is not None

    def frame_waiting(self) -> bool:
        return self.active and self.kind == "camera" and self.submitted is not None

    def result_waiting(self) -> bool:
        """The oldest frame asked for is done: results are taken in the order they were asked."""
        return bool(self.asked_for) and self.networks.done(self.asked_for[0].key)

    def ask(self, job) -> None:
        self.asked_for.append(job._replace(image=None, camera=None))
        self.networks.submit(job)

    def run(self) -> None:
        try:
            detector = self.detector_factory(lambda text: self.publish(message=text))
        except Exception as error:  # noqa: BLE001 - a missing or damaged model must reach the interface.
            self.publish(message=f"Не удалось загрузить модели: {error}", failed=True)
            return
        self.networks = Networks(detector, getattr(detector, "workers", WORKERS), self.networks_done)
        self.publish(message=START_HINT, device=getattr(detector, "device_label", ""), ready=True)
        try:
            while self.step():
                pass
        finally:
            self.networks.stop()
            self.stop_render()
            self.close_video()

    def step(self) -> bool:
        with self.wake:
            while (not self.stopping and not self.pending and not self.running() and not self.frame_waiting()
                   and not self.rendering() and not self.result_waiting()):
                self.wake.wait(IDLE_WAIT_S)
            if self.stopping:
                return False
            commands, self.pending = self.pending, []
        for command in commands:
            self.handle(*command)
        try:
            if self.result_waiting():
                self.collect()
            elif self.running():
                self.play()
            elif self.frame_waiting():
                if self.networks.free():
                    with self.wake:
                        camera, self.submitted = self.submitted, None
                    self.camera_frames += 1
                    self.ask(Job(self.generation, self.camera_frames, camera.image, camera))
                else:
                    with self.wake:
                        self.wake.wait(PLAY_WAIT_S)
            elif self.rendering():
                self.render_step()
        except Exception as error:  # noqa: BLE001 - one bad frame must not end the session.
            self.publish(message=f"Кадр не обработан: {error}")
        return True

    def handle(self, name, *values) -> None:
        """Apply one command from the interface."""
        if name == "reference":
            self.reference, self.segment_factor = values[0], None
            self.calibrate_segment()
            self.show()
        elif name == "known":
            self.known = values[0]
            self.show()
        elif name == "learned":
            self.learned = values[0]
            self.show()
        elif name == "play":
            if values[0] and self.finished:
                self.rewind()
            self.playing = bool(values[0]) and self.video is not None and not self.finished
            self.next_due = 0.
            self.publish(playing=self.playing)
        elif name == "rewind" and self.video is not None:
            self.rewind()
        elif name == "render":
            self.start_render()
            self.show()
        elif name == "reset":
            if self.kind == "video" and self.video is not None:
                self.rewind()
                return
            self.forget_animals()
            if self.kind == "photo":
                self.analyse_still(self.source)
            self.show()
        elif name == "photo":
            self.close_video()
            image = read_image(values[0])
            if image is None:
                self.publish(message="Не удалось открыть фото: файл повреждён или это не изображение.")
                return
            self.kind, self.source, self.scene = "photo", fitted(image), None
            self.forget_animals()
            self.analyse_still(self.source)
        elif name == "video":
            self.open_video(values[0])
        elif name == "camera":
            self.close_video()
            self.kind, self.source, self.scene = "camera", None, None
            self.new_generation()
            self.forget_animals()
            with self.wake:
                self.submitted = None
            self.publish(kind="camera", playing=False, finished=False, image=None, pigs=(), animals=(), position_s=0.,
                         duration_s=0., frame=0, frames=0, rendering=None, video_file="", message="Жду кадр с камеры…")

    def analyse_still(self, image) -> None:
        """A photograph goes to the networks like a frame; its result is shown when it comes."""
        self.new_generation()
        self.ask(Job(self.generation, 0, image))

    def new_generation(self) -> None:
        """Whatever the networks still do for the previous source is dropped when it comes."""
        self.generation += 1
        self.asked_for = []
        if self.networks is not None:
            self.networks.forget(self.generation)

    def forget_animals(self) -> None:
        """New numbers from 1, nothing measured."""
        self.tracker = PigTracker()
        self.entries, self.seen, self.notes, self.marks = [], {}, {}, {}
        self.rate = self.analysis_rate = 0.

    def calibrate_segment(self) -> None:
        """How many times the marked segment is longer than it is seen in the last analysed frame.

        On a video that frame can be a little older than the one on screen; the camera is
        expected to stand still, as the segment is marked on things that do not move.
        """
        if self.reference is None or self.scene is None:
            return
        segment, true_length = self.reference
        seen = segment_length(self.scene.points, segment)
        factor = true_length / seen if seen else 0.
        self.segment_factor = factor if FACTOR_LIMITS[0] <= factor <= FACTOR_LIMITS[1] else 0.

    # Results of the networks.
    def collect(self) -> None:
        job = self.asked_for.pop(0)
        outcome = self.networks.take(job.key)
        if isinstance(outcome, Exception):
            self.publish(message=f"Кадр не обработан: {outcome}")
            return
        self.analysis_rate = smoothed(self.analysis_rate, 1 / max(1e-3, outcome.seconds))
        now = time.perf_counter()
        if self.last_merge:
            # Two threads often finish together: the time between results is averaged, not its inverse.
            self.merge_gap = smoothed(self.merge_gap, now - self.last_merge)
        self.last_merge = now
        if self.kind == "video":
            self.merge_video(job, outcome)
        else:
            now = 0. if self.kind == "photo" else time.monotonic()
            self.merge(outcome, now)
            self.rate = self.analysis_rate
        self.show()

    def merge(self, analysis, now, followers=None) -> None:
        """Number the animals found, add their measurements and put them on screen."""
        animals = analysis.animals
        boxes = [item.box for item in animals] if followers is None else [item.now_box for item in followers]
        tracks = self.tracker.update(np.array(boxes, float).reshape(-1, 4), now)
        self.entries = []
        for position, (track, item) in enumerate(zip(tracks, animals)):
            if item.sample is not None:
                track.accumulator.add(item.sample)
            self.notes[track.track_id] = item.note
            self.seen[track.track_id] = track
            self.entries.append(Entry(track, item, None if followers is None else followers[position]))
        self.scene = analysis.scene
        self.frame_source = analysis.scene.source
        if self.reference is not None and self.segment_factor is None:
            self.calibrate_segment()

    def merge_video(self, job, analysis) -> None:
        """Carry what was found in an older frame through the frames shown since, to the one on screen."""
        shown = self.index - 1
        indices = [index for index in range(job.index, shown + 1) if index in self.grays]
        found = [(item.contours, item.shape, item.box) for item in analysis.animals]
        if indices:
            steps = carried(found, [self.grays[index] for index in indices], self.track_scale)
        else:
            indices, steps = [shown], [[Follower(contours, shape, box, IDENTITY.copy(), np.zeros((0, 2), np.float32))
                                        for contours, shape, box in found]]
        self.merge(analysis, shown / self.fps, steps[-1])
        # The frames shown while the networks looked get this pass too: it is nearer to them than the last one.
        for index, followers in zip(indices, steps):
            self.marks[index] = tuple(Mark(entry.track.track_id, entry.track.color, entry.found.contours,
                                           entry.found.shape, entry.found.box, item.matrix)
                                      for entry, item in zip(self.entries, followers))
        self.keep_grays()
        if self.draining and not self.asked_for:
            self.end_of_video()

    # The video.
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
        self.start_playing()
        self.publish(kind="video", playing=True, finished=False, image=None, pigs=(), animals=(), position_s=0.,
                     duration_s=self.frames / self.fps, frame=0, frames=self.frames, video_fps=self.fps, rendering=None,
                     video_file="", message="Видео открыто, ищу свиней…")

    def start_playing(self) -> None:
        self.new_generation()
        self.forget_animals()
        self.index, self.playing, self.finished, self.draining, self.shrink, self.video_file = 0, True, False, False, 1., ""
        self.current = self.gray = None
        self.grays = {}
        self.next_due = self.last_shown = self.last_merge = self.merge_gap = 0.
        self.asked = -1

    def rewind(self) -> None:
        self.stop_render()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.start_playing()
        self.publish(playing=True, finished=False, position_s=0., frame=0, rendering=None, video_file="")

    def close_video(self) -> None:
        self.stop_render()
        if self.video is not None:
            self.video.release()
        self.video, self.playing, self.finished, self.draining = None, False, False, False

    def lag(self) -> int:
        """Frames shown since the oldest frame the networks have not given back."""
        return self.index - 1 - self.asked_for[0].index if self.asked_for else 0

    def keep_grays(self) -> None:
        """The small frames needed to carry outlines from the oldest frame still with the networks."""
        oldest = self.asked_for[0].index if self.asked_for else self.index - 1
        self.grays = {index: value for index, value in self.grays.items() if index >= oldest}

    def play(self) -> None:
        """Show the next frame when it is due, and keep the networks fed with the newest one."""
        if self.current is not None and self.networks.free() and not self.draining and self.asked < self.index - 1:
            self.asked = self.index - 1
            self.ask(Job(self.generation, self.asked, self.current))
            return
        if self.draining and not self.asked_for:
            self.end_of_video()
            return
        # Until the first frame is back the pace of the networks is unknown: the video waits for it.
        first = not self.last_merge and bool(self.asked_for)
        if first or self.draining or self.lag() >= max(1, round(MAX_LAG_S * self.fps)):
            with self.wake:
                self.wake.wait(PLAY_WAIT_S)
            return
        now = time.perf_counter()
        if now < self.next_due:
            with self.wake:
                self.wake.wait(min(PLAY_WAIT_S, self.next_due - now))
            return
        ok, frame = self.video.read()
        if not ok:
            self.frames = self.index
            if self.asked_for:
                self.draining = True
            else:
                self.end_of_video()
            return
        self.next_due = max(now, self.next_due) + self.interval()
        if self.last_shown:
            self.rate = smoothed(self.rate, 1 / max(1e-3, now - self.last_shown))
        self.last_shown = now
        small = fitted(frame)
        self.shrink = frame.shape[1] / small.shape[1]
        gray, self.track_scale = tracking_gray(small)
        followers = [entry.follower for entry in self.entries if entry.follower is not None]
        if self.gray is not None and followers:
            moved = iter(follow(self.gray, gray, followers, self.track_scale))
            for entry in self.entries:
                if entry.follower is not None:
                    entry.follower = next(moved)
                    entry.track.box = entry.follower.now_box
        self.current, self.gray = small, gray
        shown = self.index
        self.index += 1
        self.grays[shown] = gray
        self.keep_grays()
        self.keep_marks(shown)
        self.show()

    def networks_rate(self) -> float:
        """Frames the networks give back per second, all threads together."""
        return 1 / self.merge_gap if self.merge_gap > 1e-3 else self.analysis_rate * self.networks.count

    def interval(self) -> float:
        """Seconds between two frames shown: the video's own, or longer to keep pace with the networks."""
        rate = self.networks_rate()
        lag_limit = max(1, round(MAX_LAG_S * self.fps))
        return max(1 / self.fps, 1 / (lag_limit * rate * PACE_MARGIN)) if rate else 1 / self.fps

    def keep_marks(self, index) -> None:
        self.marks[index] = tuple(Mark(entry.track.track_id, entry.track.color, entry.found.contours, entry.found.shape,
                                       entry.found.box, IDENTITY if entry.follower is None else entry.follower.matrix)
                                  for entry in self.entries)

    def end_of_video(self) -> None:
        self.playing = self.draining = False
        self.finished = True
        self.frames = self.index
        self.start_render()
        self.show()

    # Scale and numbers.
    def scale(self) -> tuple:
        """(length factor, where the scale comes from, the animal weighed on scales if it set the scale)."""
        factor, source = 1., self.frame_source
        if self.reference is not None and self.segment_factor is not None:
            factor, source = (self.segment_factor, SEGMENT) if self.segment_factor else (1., SEGMENT_FAILED)
        elif source == PICTURE and self.learned is not None:
            factor, source = self.learned[0], LEARNED
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
        """Publish the frame on screen with what is known about every animal in it."""
        estimates, source, factor, weighed = self.estimates()
        pigs = tuple(PigView(entry.track.track_id, entry.box, entry.track.color, entry.contours, entry.shape,
                             estimates.get(entry.track.track_id), entry.found.note) for entry in self.entries)
        in_view = {pig.track_id for pig in pigs}
        animals = tuple(PigRow(key, estimate, WEIGHED if key == weighed else self.notes.get(key, ""), key in in_view,
                               self.seen[key].accumulator.estimate(), self.frame_source)
                        for key, estimate in estimates.items())
        measured = sum(estimate is not None for estimate in estimates.values())
        video = self.kind == "video"
        image = self.current if video else None if self.scene is None else self.scene.image
        if image is None:
            message = self.state.message
        elif video and self.finished:
            message = f"Видео обработано целиком, кадров: {self.frames}. Взвешено свиней: {measured}."
        elif video and self.draining:
            message = "Видео закончилось, жду последний разбор кадра…"
        elif video and not self.last_merge:
            message = "Нейросети разбирают первый кадр, видео пойдёт следом…"
        elif video:
            message = f"Кадр {self.index} из {max(self.frames, self.index)}. Взвешено свиней: {measured}."
        elif not pigs:
            message = "Свиней на кадре не найдено."
        else:
            message = f"Найдено свиней: {len(pigs)}, измерено: {sum(pig.estimate is not None for pig in pigs)}."
        floor = None if self.scene is None else self.scene.floor
        self.publish(image=image, pigs=pigs, animals=animals, message=message, kind=self.kind, playing=self.playing,
                     finished=self.finished, position_s=self.index / self.fps if video else 0.,
                     duration_s=self.frames / self.fps if video else 0., frame=self.index if video else 0,
                     frames=self.frames if video else 0, scale_source=source, video_fps=self.fps if video else 0.,
                     camera_height_m=None if floor is None else floor.offset * factor, weighed_pig=weighed,
                     learned_pigs=0 if self.learned is None else self.learned[1],
                     fps=self.rate if self.kind in ("video", "camera") else 0.,
                     analysis_fps=self.networks_rate() if self.networks is not None else 0.,
                     video_file=self.video_file, rendering=None if self.render is None else self.state.rendering or 0.)

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
