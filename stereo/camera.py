from dataclasses import dataclass, asdict, replace
import json
import os
import time
import multiprocessing as mp
import ctypes
from queue import Queue, Empty, Full
from threading import Thread, Event
from collections import deque

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    left: int = 0
    right: int = 1
    width: int = 640
    height: int = 480
    fps: int = 15
    mode: str = "dual"
    swap: bool = False
    backend: str = "AUTO"
    address: str = "http://192.168.233.1"

    def signature(self):
        # Resolution is checked separately against the actual delivered frames.
        # The address belongs to the ToF camera: saved stereo calibrations must still match.
        return json.dumps({k: v for k, v in asdict(self).items()
                           if k not in ("width", "height", "fps", "backend", "address")}, sort_keys=True)


def open_device(index, config, pixel_format="MJPG"):
    backend = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF,
               "AUTO": cv2.CAP_ANY}[config.backend] if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось открыть камеру {index}. Закройте другие приложения камеры "
                           "или попробуйте другой индекс / способ подключения.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width * (2 if config.mode == "sbs" else 1))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if pixel_format != "native":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
    cap.set(cv2.CAP_PROP_FPS, config.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class StereoCamera:
    def __init__(self, config, status=None, cancel_event=None):
        self.config = config
        self.caps = []
        self.initial_pair = None
        self.info = {}
        if config.mode == "dual" and config.left == config.right:
            raise ValueError("Для двух камер нужны разные индексы.")
        failures = []
        for backend, pixel_format, fps in capture_profiles(config):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Подключение отменено.")
            active = replace(config, backend=backend, fps=fps)
            if status:
                status(f"Подключение: {backend}, {pixel_format}, {config.width} × {config.height}, {fps} кадров/с…")
            try:
                self.caps.append(open_device(config.left, active, pixel_format))
                if config.mode == "dual":
                    self.caps.append(open_device(config.right, active, pixel_format))
                deadline = time.perf_counter() + 2.5
                stable_frames = 0
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Подключение отменено.")
                    try:
                        pair = self._read_once()
                        stable_frames += 1
                        if stable_frames >= 6:
                            break
                    except RuntimeError:
                        stable_frames = 0
                        if time.perf_counter() >= deadline:
                            raise
                        time.sleep(.03)
                size = (pair[0].shape[1], pair[0].shape[0])
                if size != (config.width, config.height):
                    raise ValueError(f"Драйвер передал {size[0]} × {size[1]} вместо "
                                     f"{config.width} × {config.height}")
                self.active_config = active
                self.initial_pair = pair
                self.info = {"backend": backend, "pixel_format": pixel_format, "requested_fps": config.fps,
                             "selected_fps": fps, "size": size, "fallback": bool(failures),
                             "attempts": failures.copy()}
                self.info["devices"] = [{"reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
                                         "fourcc": fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))}
                                        for cap in self.caps]
                if status:
                    status(f"Камеры подключены: {backend}, {pixel_format}, {size[0]} × {size[1]}, "
                           f"запрошено {fps} кадров/с" + ("; использован запасной режим" if failures else ""))
                return
            except (RuntimeError, ValueError, cv2.error) as error:
                failures.append({"backend": backend, "format": pixel_format, "fps": fps, "error": str(error)})
                self.close()
        raise RuntimeError("Не удалось получить стабильную стереопару в выбранном разрешении. "
                           "Попробованы разные форматы и способы подключения. "
                           "Выберите 640 × 480 / 15 кадров/с или проверьте USB. "
                           f"Последняя причина: {failures[-1]['error']}")

    def read(self):
        if self.initial_pair is not None:
            pair, self.initial_pair = self.initial_pair, None
            return pair
        for attempt in range(4):
            try:
                return self._read_once()
            except RuntimeError:
                if attempt == 3:
                    raise
                time.sleep(.025)

    def _read_once(self):
        # Grab both before decoding. Host timing does not establish hardware
        # exposure synchronization on two USB devices.
        for cap in self.caps:
            if not cap.grab():
                raise RuntimeError("Нет кадра от камеры. Проверьте USB-подключение.")
        frames = []
        for cap in self.caps:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError("Камера не передала изображение.")
            frames.append(frame)
        if self.config.mode == "sbs":
            frame = frames[0]
            if frame.shape[1] % 2:
                raise ValueError("Ширина общего стереокадра должна быть чётной.")
            middle = frame.shape[1] // 2
            left, right = frame[:, :middle].copy(), frame[:, middle:].copy()
        else:
            left, right = frames
        if left.shape != right.shape:
            raise ValueError(f"Разный размер кадров: {left.shape} и {right.shape}.")
        return (right, left) if self.config.swap else (left, right)

    def close(self):
        for cap in self.caps:
            cap.release()
        self.caps.clear()


def capture_profiles(config):
    if os.name != "nt":
        backends = ["AUTO"]
    elif config.backend == "AUTO":
        backends = ["MSMF", "DSHOW"]
    else:
        backends = [config.backend, "MSMF" if config.backend == "DSHOW" else "DSHOW"]
    profiles = []
    for backend in backends:
        formats = ["MJPG", "native"]
        # Some drivers accept 15 fps but deliver frames only at 30 fps.
        rates = (config.fps, 30, min(config.fps, 15))
        for fps in dict.fromkeys(rates):
            profiles.extend((backend, fmt, fps) for fmt in formats)
    return profiles


def fourcc_name(value):
    value = int(value)
    code = "".join(chr((value >> (8 * i)) & 255) for i in range(4)).strip("\x00")
    return code if code and all(32 <= ord(char) < 127 for char in code) else f"0x{value:08x}"


def _capture_process(config, buffer, lock, sequence, timestamp, notices, stop):
    """Own all driver handles in a process that can be stopped after a hang."""
    camera = None
    cv2.setNumThreads(1)
    array = np.frombuffer(buffer, np.uint8).reshape(2, config.height, config.width, 3)
    try:
        camera = StereoCamera(config, lambda message: notices.put(("status", message)), stop)
        notices.put(("info", camera.info))
        while not stop.is_set():
            pair = camera.read()
            with lock:
                array[0], array[1] = pair
                timestamp.value = time.perf_counter()
                sequence.value += 1
    except Exception as error:
        if not stop.is_set():
            notices.put(("error", str(error)))
    finally:
        if camera:
            camera.close()


class CapturePump(Thread):
    """Supervise an isolated USB reader; keep fresh, bounded consumer queues."""
    def __init__(self, config, status=None, *, process_target=None, stall_timeout=3., startup_timeout=20.):
        super().__init__(daemon=True)
        self.config, self.status = config, status
        self.frames, self.previews = Queue(maxsize=1), Queue(maxsize=1)
        self.stop_event = Event()
        self.error, self.info = None, {}
        self.count, self.fps, self.generation, self.restart_count = 0, 0., 0, 0
        self.stream_generation = 0
        self.state = "starting"
        self.stall_timeout, self.startup_timeout = stall_timeout, startup_timeout
        self.process_target = process_target or _capture_process
        self.child = None
        self.ipc = None

    @staticmethod
    def latest(queue, packet):
        try:
            queue.put_nowait(packet)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            queue.put_nowait(packet)

    def launch(self, config):
        context = mp.get_context("spawn")
        buffer = context.RawArray(ctypes.c_ubyte, 2 * config.width * config.height * 3)
        lock, sequence, timestamp = context.Lock(), context.RawValue("Q", 0), context.RawValue("d", 0.)
        notices, stop = context.Queue(), context.Event()
        self.ipc = (buffer, lock, sequence, timestamp, notices, stop)
        self.child = context.Process(target=self.process_target,
                                     args=(config, *self.ipc), daemon=True, name="Stereo USB capture")
        try:
            self.child.start()
        except Exception:
            notices.close()
            self.child, self.ipc = None, None
            raise
        self.generation += 1

    def dispose(self):
        if self.child is None:
            return
        self.ipc[-1].set()
        self.child.join(.7)
        if self.child.is_alive():
            self.child.terminate()
            self.child.join(1.)
        if not self.child.is_alive():
            self.child.close()
        self.ipc[-2].cancel_join_thread()
        self.ipc[-2].close()
        self.child, self.ipc = None, None

    def run(self):
        restarts, active, failure_streak = deque(), self.config, 0
        try:
            while not self.stop_event.is_set():
                self.launch(active)
                buffer, lock, sequence, timestamp, notices, _ = self.ipc
                array = np.frombuffer(buffer, np.uint8).reshape(2, active.height, active.width, 3)
                last_seen, last_progress, connected = 0, time.perf_counter(), False
                streaming_since = None
                times = deque()
                reason = None
                self.state = "starting" if not self.restart_count else "reconnecting"
                while not self.stop_event.is_set():
                    try:
                        while True:
                            kind, value = notices.get_nowait()
                            if kind == "status":
                                last_progress = time.perf_counter()
                                if self.status:
                                    self.status(value)
                            elif kind == "info":
                                self.info = dict(value, restart_count=self.restart_count)
                                connected = True
                            elif kind == "error":
                                reason = value
                    except Empty:
                        pass
                    now = time.perf_counter()
                    if reason:
                        break
                    if not self.child.is_alive():
                        reason = "Процесс USB-захвата остановился."
                        break
                    if connected and sequence.value > last_seen and lock.acquire(timeout=.03):
                        try:
                            current_sequence = sequence.value
                            captured = timestamp.value
                            snapshot = array.copy()
                        finally:
                            lock.release()
                        self.count += current_sequence - last_seen
                        self.stream_generation = self.generation
                        last_seen, last_progress = current_sequence, now
                        connected = True
                        self.state = "streaming"
                        if streaming_since is None:
                            streaming_since = now
                        times.append((self.count, captured))
                        while times and times[0][1] < captured - 2:
                            times.popleft()
                        self.fps = ((times[-1][0] - times[0][0]) / max(times[-1][1] - times[0][1], .001)
                                    if len(times) > 1 else 0.)
                        packet = ((snapshot[0], snapshot[1]), captured, self.count)
                        self.latest(self.frames, packet)
                        self.latest(self.previews, packet)
                    if streaming_since and now - streaming_since > 8 and 0 < self.fps < 5:
                        reason = "Драйвер передаёт менее 5 стереопар в секунду."
                        break
                    if streaming_since and now - streaming_since > 30:
                        failure_streak = 0
                    timeout = self.stall_timeout if connected else self.startup_timeout
                    if now - last_progress > timeout:
                        reason = "USB-драйвер перестал передавать кадры."
                        break
                    self.stop_event.wait(min(.025,.5/max(active.fps,1)))
                self.dispose()
                if self.stop_event.is_set():
                    break
                now = time.perf_counter()
                restarts.append(now)
                failure_streak += 1
                while restarts and restarts[0] < now - 45:
                    restarts.popleft()
                if len(restarts) > 3 or failure_streak > 3:
                    raise RuntimeError(f"Поток неоднократно обрывается: {reason} "
                                       "Подключите напрямую без удлинителя; попробуйте другой USB-разъём "
                                       "и 640 × 480 / 15 кадров/с.")
                self.restart_count += 1
                self.state, self.fps = "reconnecting", 0.
                for queue in (self.frames, self.previews):
                    try:
                        while True:
                            queue.get_nowait()
                    except Empty:
                        pass
                if self.status:
                    self.status(f"Поток прервался: {reason} Восстанавливаю подключение ({self.restart_count})…")
                backend = "MSMF" if active.backend in ("AUTO", "DSHOW") else "DSHOW"
                active = replace(active, backend=backend, fps=min(active.fps, 15))
                if self.restart_count >= 2 and active.width > 640:
                    active = replace(active, width=640, height=480, backend="MSMF", fps=15)
                    if self.status:
                        self.status("Высокое разрешение нестабильно. Перехожу на 640 × 480; потребуется калибровка этого режима.")
                self.stop_event.wait(.25)
        except Exception as error:
            if not self.stop_event.is_set():
                self.error, self.state = error, "failed"
        finally:
            self.dispose()
            if self.state != "failed":
                self.state = "stopped"

    def take(self, consumer_stop):
        while not consumer_stop.is_set():
            if self.error:
                raise self.error
            try:
                return self.frames.get(timeout=.2)
            except Empty:
                if not self.is_alive():
                    raise RuntimeError("Захват камер остановлен.")
        return None
