import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from stereo.board import BoardSpec, board_image
from stereo.camera import CameraConfig, CapturePump
from stereo.worker import CameraWorker
from test_core import fixture


def hung_driver(config, buffer, lock, sequence, timestamp, notices, stop):
    """Emulate a driver which delivers frames, then blocks inside read forever."""
    array = np.frombuffer(buffer, np.uint8).reshape(2, config.height, config.width, 3)
    notices.put(("info", {"size": (config.width, config.height), "backend": "TEST"}))
    for value in range(1, 4):
        with lock:
            array[0].fill(value)
            array[1].fill(value + 10)
            timestamp.value = time.perf_counter()
            sequence.value += 1
        time.sleep(.04)
    time.sleep(60)


class StreamTests(unittest.TestCase):
    def test_blocked_driver_recovers_and_shutdown_releases_own_process(self):
        pump = CapturePump(CameraConfig(width=80, height=64), process_target=hung_driver,
                           stall_timeout=.2, startup_timeout=4.)
        pump.start()
        try:
            first = pump.frames.get(timeout=5)
            original = first[0][0].copy()
            deadline = time.perf_counter() + 6
            while pump.stream_generation < 2 and time.perf_counter() < deadline:
                time.sleep(.05)
            self.assertGreaterEqual(pump.restart_count, 1)
            self.assertGreaterEqual(pump.stream_generation, 2)
            pair, captured, sequence = pump.frames.get(timeout=1)
            self.assertLess(time.perf_counter() - captured, 1)
            np.testing.assert_array_equal(pair[1], pair[0] + 10)
            np.testing.assert_array_equal(first[0][0], original)
            self.assertLessEqual(pump.previews.qsize(), 1)
            self.assertGreater(sequence, first[2])
        finally:
            started = time.perf_counter()
            pump.stop_event.set()
            pump.join(4)
        self.assertFalse(pump.is_alive())
        self.assertIsNone(pump.child)
        self.assertLess(time.perf_counter() - started, 3)

    def test_board_can_be_detected_and_has_exact_integer_cells(self):
        spec = BoardSpec(9, 6, 25)
        image = board_image(spec, 80)
        self.assertEqual(image.shape, (640, 880))
        found, corners = cv2.findChessboardCorners(image, (9, 6))
        self.assertTrue(found)
        points = corners.reshape(6, 9, 2)
        self.assertAlmostEqual(float(np.linalg.norm(np.diff(points, axis=1), axis=2).mean()), 80, delta=.05)
        self.assertAlmostEqual(float(np.linalg.norm(np.diff(points, axis=0), axis=2).mean()), 80, delta=.05)

    def test_depth_pause_stops_mapping_and_snapshot_measures_displayed_pair(self):
        calibration, left, right = fixture()
        with TemporaryDirectory() as directory:
            config = CameraConfig(width=384, height=240)
            calibration.signature = config.signature()
            worker = CameraWorker(Path(directory), config, (9, 6), 25, 8, 64)
            worker.raw = (left, right)
            worker.apply_calibration(calibration)
            self.assertFalse(worker.depth_enabled)
            worker.command("mapping", True)
            self.assertTrue(worker.depth_enabled)
            worker.command("depth", False)
            self.assertFalse(worker.mapping)
            # Live images may already differ. A click must use its saved pair.
            worker.raw = (np.zeros_like(left), np.zeros_like(right))
            worker.command("measure_snapshot", {"raw": (left, right), "point": (200, 100),
                                               "calibration_token": id(worker.engine)})
            events = list(worker.events.queue)
            result = [value for kind, value in events if kind == "snapshot_measurement"][-1]
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result["z"], .9, delta=.04)
            with self.assertRaises(ValueError):
                worker.command("measure_snapshot", {"calibration_token": -1})
            worker.close_maps()


if __name__ == "__main__":
    unittest.main()
