"""Sipeed MaixSense A075V: HTTP protocol, factory lens geometry and frame source.

Measured on a live camera, not taken from memory:
- 16-bit depth equals what the vendor's DecodePkg unpacks from its 8-bit mode.
- The vendor's TOF_cali keeps values and only refines status codes.
- Raw depth is millimetres along Z. On a flat wall a Z model fitted a plane to 10.6 mm RMS
  and a radial one to 36.7 mm; the plane lay at 0.97 m for a wall measured at about 1 m.
- The lens model neither helped nor hurt on that wall (10.61 versus 10.68 mm), so its
  coefficient order rests on parsing the device data, not on a measured improvement.
"""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
import struct
from threading import Event, Thread
import time
from typing import Callable, NamedTuple
import urllib.parse
import urllib.request

import cv2
import numpy as np

TOF_MODE = "tof"
TOF_LABEL = "MaixSense A075V (ToF)"
DEFAULT_ADDRESS = "http://192.168.233.1"
TOF_WIDTH, TOF_HEIGHT = 320, 240
HEADER_BYTES = 16
CONFIG_BYTES = 12
SIZES_BYTES = 8
DEPTH_BYTES = IR_BYTES = TOF_WIDTH * TOF_HEIGHT * 2
STATUS_BYTES = TOF_WIDTH * TOF_HEIGHT
STREAM_BYTES = DEPTH_BYTES + IR_BYTES + STATUS_BYTES
MAX_FRAME_BYTES = 4 * 1024 * 1024
INFO_BYTES = 81
LENS_OFFSET = 41

DEPTH_UNIT_M = .001
STATUS_NORMAL = 0
STATUS_OUTSIDE = 255
IR_FULL_SCALE = 4095

# Colour camera relative to the ToF camera, for a 640 x 480 frame. Fitted on live frames of one
# camera by matching image edges: two independent sets of frames agreed.
# The vendor's depth-independent mapping scored half as well and was visibly shifted up close.
RGB_REFERENCE_WIDTH, RGB_REFERENCE_HEIGHT = 640, 480
RGB_FOCAL = 481.4
RGB_CENTER = (320.5, 215.1)
RGB_BASELINE_M = (-.0163, .0349)
UNKNOWN_DEPTH_FOR_COLOR_M = 1.

MAX_STALE_FRAMES = 20
MAX_FAILURES = 4
HEALTHY_FRAMES = 30
REQUEST_TIMEOUT = 3.

_IR_LUT = np.uint8(np.round(np.sqrt(np.arange(IR_FULL_SCALE + 1) / IR_FULL_SCALE) * 255))


class FrameFormatError(ValueError):
    """The camera answered, but not with a complete frame in the requested mode."""


class DeviceInfo(NamedTuple):
    sensor: str
    serial: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple


class MaixFrame(NamedTuple):
    frame_id: int
    stamp_ms: int
    depth: np.ndarray
    ir: np.ndarray
    status: np.ndarray
    rgb: np.ndarray | None


def build_frame_config(trigger_mode=1, deep_mode=0, deep_shift=255, ir_mode=0, status_mode=2,
                       status_mask=7, rgb_mode=1, rgb_res=0, expose_time=0) -> bytes:
    """Default: 16-bit depth and infrared, 8-bit status, JPEG colour at 640 x 480."""
    return struct.pack("<BBBBBBBBi", trigger_mode, deep_mode, deep_shift, ir_mode, status_mode,
                       status_mask, rgb_mode, rgb_res, expose_time)


def normalize_address(value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError("Укажите адрес камеры MaixSense, например 192.168.233.1.")
    if "://" not in text:
        text = "http://" + text
    parts = urllib.parse.urlsplit(text)
    if (parts.scheme != "http" or not parts.hostname or parts.username or parts.password
            or parts.path not in ("", "/") or parts.query or parts.fragment):
        raise ValueError("Адрес камеры MaixSense должен выглядеть как 192.168.233.1 или http://192.168.233.1.")
    return f"http://{parts.netloc}"


def parse_info(data: bytes) -> DeviceInfo:
    if len(data) < INFO_BYTES:
        raise ValueError("Камера вернула неполные заводские параметры.")
    text = lambda chunk: chunk.split(b"\0")[0].decode("ascii", "replace")
    fx, fy, cx, cy, k1, k2, k3, p1, p2 = struct.unpack("<9f", data[LENS_OFFSET:LENS_OFFSET + 36])
    lens = (fx, fy, cx, cy, k1, k2, k3, p1, p2)
    if (not np.isfinite(lens).all() or not 50 < fx < 2000 or not 50 < fy < 2000
            or not 0 < cx < TOF_WIDTH or not 0 < cy < TOF_HEIGHT):
        raise ValueError("Заводские параметры объектива камеры выглядят некорректно.")
    return DeviceInfo(text(data[:13]), text(data[13:33]), fx, fy, cx, cy, (k1, k2, p1, p2, k3))


def decode_frame(body: bytes) -> MaixFrame:
    minimum = HEADER_BYTES + CONFIG_BYTES + SIZES_BYTES
    if len(body) < minimum:
        raise FrameFormatError("Камера вернула неполный кадр.")
    frame_id, stamp = struct.unpack("<QQ", body[:HEADER_BYTES])
    fields = struct.unpack("<BBBBBBBBi", body[HEADER_BYTES:HEADER_BYTES + CONFIG_BYTES])
    # After a mode switch the camera can still send frames in the previous format.
    if fields[1] != 0 or fields[3] != 0 or fields[4] != 2:
        raise FrameFormatError("Камера ещё не переключилась в режим 16 бит.")
    stream_size, rgb_size = struct.unpack("<ii", body[minimum - SIZES_BYTES:minimum])
    if stream_size != STREAM_BYTES or rgb_size < 0 or len(body) != minimum + stream_size + rgb_size:
        raise FrameFormatError("Размер кадра камеры не совпадает с заголовком.")
    start = minimum
    depth = np.frombuffer(body, "<u2", TOF_WIDTH * TOF_HEIGHT, start).reshape(TOF_HEIGHT, TOF_WIDTH)
    start += DEPTH_BYTES
    ir = np.frombuffer(body, "<u2", TOF_WIDTH * TOF_HEIGHT, start).reshape(TOF_HEIGHT, TOF_WIDTH)
    start += IR_BYTES
    status = np.frombuffer(body, "u1", STATUS_BYTES, start).reshape(TOF_HEIGHT, TOF_WIDTH)
    rgb = None
    if rgb_size:
        rgb = cv2.imdecode(np.frombuffer(body, "u1", rgb_size, start + STATUS_BYTES), cv2.IMREAD_COLOR)
    return MaixFrame(frame_id, stamp, depth, ir, status, rgb)


@dataclass
class TofCalibration:
    """Factory intrinsics of the ToF sensor. It has no stereo pair and no baseline."""
    size: tuple
    K: np.ndarray
    D: np.ndarray
    serial: str = ""
    P1: np.ndarray = field(init=False)
    R1: np.ndarray = field(default_factory=lambda: np.eye(3))
    baseline: None = None
    rms: float = 0.
    epipolar_error: float = 0.
    signature: str = "tof"

    def __post_init__(self):
        self.P1 = np.hstack((self.K, np.zeros((3, 1))))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, kind="tof", size=np.asarray(self.size), K=self.K, D=self.D, serial=self.serial)


class TofGeometry:
    """Undistorts each frame so everything downstream can use an ideal pinhole camera."""

    def __init__(self, info: DeviceInfo):
        K = np.array([[info.fx, 0, info.cx], [0, info.fy, info.cy], [0, 0, 1]], np.float64)
        D = np.array(info.distortion, np.float64)
        self.calibration = TofCalibration((TOF_WIDTH, TOF_HEIGHT), K, D, info.serial)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            K, D, None, K, (TOF_WIDTH, TOF_HEIGHT), cv2.CV_32FC1)
        u, v = np.meshgrid(np.arange(TOF_WIDTH, dtype=np.float32), np.arange(TOF_HEIGHT, dtype=np.float32))
        self.ray_x, self.ray_y = (u - info.cx) / info.fx, (v - info.cy) / info.fy

    def color_view(self, rgb: np.ndarray, ir_bgr: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Sample colour for each depth pixel; nearer surfaces shift more between the two lenses."""
        height, width = rgb.shape[:2]
        z = np.where(depth > 0, depth, UNKNOWN_DEPTH_FOR_COLOR_M)
        x = (RGB_CENTER[0] + RGB_FOCAL * (self.ray_x + RGB_BASELINE_M[0] / z)) * width / RGB_REFERENCE_WIDTH
        y = (RGB_CENTER[1] + RGB_FOCAL * (self.ray_y + RGB_BASELINE_M[1] / z)) * height / RGB_REFERENCE_HEIGHT
        warped = cv2.remap(rgb, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
        inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
        return np.where(inside[..., None], warped, ir_bgr)

    def rectify(self, frame: MaixFrame) -> tuple:
        """Return (infrared BGR, colour BGR, depth in metres, status, colour aligned to depth)."""
        ir = _IR_LUT[np.minimum(frame.ir, IR_FULL_SCALE)]
        ir = cv2.remap(ir, self.map_x, self.map_y, cv2.INTER_LINEAR)
        ir_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        # Nearest neighbour only: interpolation would invent depth across object edges.
        depth = cv2.remap(frame.depth.astype(np.float32) * DEPTH_UNIT_M, self.map_x, self.map_y,
                          cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        status = cv2.remap(frame.status, self.map_x, self.map_y, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=STATUS_OUTSIDE)
        if frame.rgb is None:
            return ir_bgr, ir_bgr.copy(), depth, status, ir_bgr.copy()
        return ir_bgr, frame.rgb, depth, status, self.color_view(frame.rgb, ir_bgr, depth)


class MaixSenseClient:
    def __init__(self, address: str = DEFAULT_ADDRESS, timeout: float = REQUEST_TIMEOUT):
        self.address, self.timeout = normalize_address(address), timeout
        # The camera sits on a local USB network and must bypass any system proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path: str, data: bytes | None = None) -> bytes:
        request = urllib.request.Request(self.address + path, data=data,
                                         headers={"Cache-Control": "no-cache"})
        with self.opener.open(request, timeout=self.timeout) as response:
            body = response.read(MAX_FRAME_BYTES + 1)
        if len(body) > MAX_FRAME_BYTES:
            raise FrameFormatError("Ответ камеры слишком большой.")
        return body

    def info(self) -> DeviceInfo:
        return parse_info(self.request("/getinfo"))

    def configure(self) -> None:
        self.request("/set_cfg", build_frame_config())

    def frame(self) -> MaixFrame:
        return decode_frame(self.request("/getdeep"))


class MaixSensePump(Thread):
    """Frame source with the interface of CapturePump, reading over the camera's HTTP API."""

    def __init__(self, config, status: Callable[[str], None] | None = None, *,
                 client=None, retry_delay: float = 1.):
        super().__init__(daemon=True, name="MaixSense capture")
        self.config, self.status = config, status
        self.client = client or MaixSenseClient(config.address)
        self.retry_delay = retry_delay
        self.frames, self.previews = Queue(maxsize=1), Queue(maxsize=1)
        self.stop_event = Event()
        self.error, self.info = None, {}
        self.count, self.fps, self.restart_count = 0, 0., 0
        self.session_frames = 0
        self.stream_generation = 0
        self.geometry = None
        self.state = "starting"

    @staticmethod
    def latest(queue: Queue, packet) -> None:
        try:
            queue.put_nowait(packet)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            queue.put_nowait(packet)

    def notify(self, message: str) -> None:
        if self.status:
            self.status(message)

    def connect(self) -> None:
        device = self.client.info()
        self.geometry = TofGeometry(device)
        self.client.configure()
        self.stream_generation += 1
        self.info = dict(backend="MaixSense A075V", size=(TOF_WIDTH, TOF_HEIGHT),
                         address=getattr(self.client, "address", self.config.address),
                         sensor=device.sensor, serial=device.serial,
                         fx=device.fx, fy=device.fy, cx=device.cx, cy=device.cy,
                         distortion=list(device.distortion), restart_count=self.restart_count)

    def stream(self) -> None:
        """Read until stopped or the link fails, counting frames in session_frames."""
        stale = 0
        last_id = None
        times = deque()
        while not self.stop_event.is_set():
            try:
                frame = self.client.frame()
            except FrameFormatError:
                stale += 1
                if stale > MAX_STALE_FRAMES:
                    raise
                continue
            stale = 0
            if frame.frame_id == last_id:
                self.stop_event.wait(.005)
                continue
            last_id = frame.frame_id
            raw = self.geometry.rectify(frame)
            captured = time.perf_counter()
            self.count += 1
            self.session_frames += 1
            times.append(captured)
            while len(times) > 1 and times[0] < captured - 2:
                times.popleft()
            self.fps = (len(times) - 1) / max(times[-1] - times[0], .001) if len(times) > 1 else 0.
            self.state = "streaming"
            packet = (raw, captured, self.count)
            self.latest(self.frames, packet)
            self.latest(self.previews, packet)

    def run(self) -> None:
        failures = 0
        try:
            while not self.stop_event.is_set():
                self.session_frames = 0
                try:
                    self.connect()
                    self.stream()
                except (OSError, ValueError) as error:
                    if self.stop_event.is_set():
                        break
                    failures = 1 if self.session_frames >= HEALTHY_FRAMES else failures + 1
                    if failures > MAX_FAILURES:
                        raise RuntimeError(
                            f"Камера MaixSense не отвечает по адресу {self.config.address}: {error}. "
                            "Проверьте USB-кабель, драйвер и что страница камеры открывается в браузере.") from error
                    self.state, self.fps = "reconnecting", 0.
                    self.restart_count += 1
                    self.notify(f"Связь с MaixSense прервалась ({error}). Повторное подключение {self.restart_count}…")
                    self.stop_event.wait(self.retry_delay)
        except Exception as error:
            if not self.stop_event.is_set():
                self.error, self.state = error, "failed"
        finally:
            if self.state != "failed":
                self.state = "stopped"

    def take(self, consumer_stop: Event):
        while not consumer_stop.is_set():
            if self.error:
                raise self.error
            try:
                return self.frames.get(timeout=.2)
            except Empty:
                if not self.is_alive():
                    if self.error:
                        raise self.error
                    raise RuntimeError("Захват MaixSense остановлен.")
        return None
