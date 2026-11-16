"""The cameras reopen at the size a calibration was taken at, without a real device."""
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import mock

import cv2
import numpy as np

from stereo import camera as camera_module
from stereo import worker as worker_module
from stereo.camera import CameraConfig, CapturePump, open_device
from stereo.worker import CameraWorker
from test_core import fixture


class FakePump:
    """Stands in for CapturePump: delivers frames only in the sizes the 'driver' supports."""
    opened = []

    def __init__(self, config, status=None, retry_startup=True, supported=((384, 240), (320, 240))):
        self.config, self.status, self.retry_startup = config, status, retry_startup
        self.supported = supported
        self.stop_event = Event()
        self.info = {"size": (config.width, config.height)}
        self.stream_generation = self.count = self.restart_count = 0
        self.fps = 0.
        self.previews = None
        self.stopped = False
        FakePump.opened.append(self)

    def start(self):
        pass

    def join(self, timeout=None):
        self.stopped = self.stop_event.is_set()

    def take(self, consumer_stop):
        size = (self.config.width, self.config.height)
        if size not in self.supported:
            raise RuntimeError(f"Драйвер передал 320 × 240 вместо {size[0]} × {size[1]}")
        self.stream_generation = 1
        image = np.zeros((size[1], size[0], 3), np.uint8)
        return (image, image.copy()), time.perf_counter(), 1


def only_small(*args, **kwargs):
    return FakePump(*args, supported=((320, 240),), **kwargs)


def refusing_driver(config, buffer, lock, sequence, timestamp, notices, stop):
    notices.put(("error", "Драйвер передал 640 × 480 вместо 1280 × 720"))
    time.sleep(10)


class ResolutionSwitchTests(unittest.TestCase):
    def setUp(self):
        FakePump.opened = []
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calibration = fixture()[0]

    def connected_worker(self, pump=FakePump, mode="dual"):
        config = CameraConfig(width=320, height=240, mode=mode)
        self.calibration.signature = config.signature()
        worker = CameraWorker(Path(self.directory.name), config, (9, 6), 25, 8, 64)
        self.addCleanup(lambda: worker.engine and worker.engine.close())
        self.addCleanup(worker.close_maps)
        with mock.patch.object(worker_module, "CapturePump", pump):
            worker.open_capture(config)
        return worker

    def events(self, worker, kind):
        return [value for name, value in list(worker.events.queue) if name == kind]

    def test_mismatched_calibration_asks_before_reopening_cameras(self):
        worker = self.connected_worker()
        self.assertFalse(worker.apply_calibration(self.calibration))
        request = self.events(worker, "resolution_mismatch")[-1]
        self.assertEqual(request["wanted"], (384, 240))
        self.assertEqual(request["current"], (320, 240))
        self.assertIsNone(worker.engine)
        self.assertEqual(len(FakePump.opened), 1)

    def test_switch_reopens_at_calibration_size_and_applies_it(self):
        worker = self.connected_worker()
        first = worker.capture_pump
        with mock.patch.object(worker_module, "CapturePump", FakePump):
            worker.command("switch_resolution", self.calibration)
        self.assertTrue(first.stopped)
        reopened = worker.capture_pump
        self.assertEqual((reopened.config.width, reopened.config.height), (384, 240))
        self.assertFalse(reopened.retry_startup)
        self.assertEqual(worker.calibration_size(), (384, 240))
        self.assertIs(worker.calibration, self.calibration)
        self.assertTrue(worker.depth_enabled)
        self.assertEqual(self.events(worker, "resolution"), [(384, 240)])

    def test_side_by_side_pair_requests_size_of_one_view(self):
        worker = self.connected_worker(mode="sbs")
        with mock.patch.object(worker_module, "CapturePump", FakePump):
            worker.command("switch_resolution", self.calibration)
        config = worker.capture_pump.config
        self.assertEqual((config.mode, config.width, config.height), ("sbs", 384, 240))

    def test_refused_size_returns_previous_resolution_with_clear_message(self):
        worker = self.connected_worker(only_small)
        with mock.patch.object(worker_module, "CapturePump", only_small):
            with self.assertRaises(ValueError) as raised:
                worker.command("switch_resolution", self.calibration)
        message = str(raised.exception)
        self.assertIn("384 × 240", message)
        self.assertIn("320 × 240", message)
        self.assertEqual(worker.calibration_size(), (320, 240))
        self.assertEqual(worker.config.width, 320)
        self.assertIsNone(worker.engine)
        self.assertEqual(len(FakePump.opened), 3)

    def test_matching_calibration_applies_without_reopening(self):
        worker = self.connected_worker()
        worker.raw = tuple(np.zeros((240, 384, 3), np.uint8) for _ in range(2))
        self.assertTrue(worker.apply_calibration(self.calibration))
        self.assertEqual(self.events(worker, "resolution_mismatch"), [])
        self.assertEqual(len(FakePump.opened), 1)

    def test_side_by_side_device_is_asked_for_double_width(self):
        calls = {}
        device = mock.Mock()
        device.isOpened.return_value = True
        device.set.side_effect = lambda prop, value: calls.setdefault(prop, value)
        with mock.patch.object(camera_module.cv2, "VideoCapture", return_value=device):
            open_device(0, CameraConfig(width=384, height=240, mode="sbs"))
        self.assertEqual(calls[cv2.CAP_PROP_FRAME_WIDTH], 768)
        self.assertEqual(calls[cv2.CAP_PROP_FRAME_HEIGHT], 240)

    def test_pump_without_startup_retries_fails_at_once(self):
        pump = CapturePump(CameraConfig(width=80, height=64), process_target=refusing_driver,
                           retry_startup=False, startup_timeout=4.)
        pump.start()
        try:
            with self.assertRaises(RuntimeError) as raised:
                pump.take(Event())
        finally:
            pump.stop_event.set()
            pump.join(4)
        self.assertIn("1280 × 720", str(raised.exception))
        self.assertEqual(pump.restart_count, 0)


if __name__ == "__main__":
    unittest.main()
