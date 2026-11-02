from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
import json

import cv2
import numpy as np

from stereo.camera import CameraConfig, CapturePump
from stereo.depth import DepthEngine, clean_disparity, texture_mask
from stereo.model3d import surface_from_frame, export_surface, read_model, render_cloud
from stereo.mono import relative_frame
from stereo.worker import CameraWorker
from test_core import fixture


def continuous_reader(config, buffer, lock, sequence, timestamp, notices, stop):
    array = np.frombuffer(buffer, np.uint8).reshape(2, config.height, config.width, 3)
    notices.put(("info", {"backend": "test", "size": (config.width, config.height)}))
    while not stop.is_set():
        with lock:
            array.fill((sequence.value + 1) % 256)
            timestamp.value = time.perf_counter()
            sequence.value += 1
        stop.wait(.003)


class FeatureTests(unittest.TestCase):
    def test_resize_preserves_metric_depth(self):
        calibration, left, right = fixture()
        f = DepthEngine(calibration, disparities=32, processing_width=192, temporal=False).compute(left, right)
        self.assertEqual(f.left.shape[:2], (120, 192))
        self.assertGreater(f.valid.sum(), 1000)
        self.assertAlmostEqual(float(np.median(f.points[f.valid, 2])), .9, delta=.04)

    def test_near_passes_measure_six_centimetres_when_geometry_permits(self):
        calibration, left, _ = fixture()
        for shift in (96, 240):
            with self.subTest(shift=shift):
                right = np.zeros_like(left)
                right[:, :-shift] = left[:, shift:]
                f = DepthEngine(calibration, disparities=64, quality="high", temporal=False,
                                near_priority=True).compute(left, right)
                self.assertGreater(f.valid.sum(), 500)
                expected = 240 * .06 / shift
                self.assertAlmostEqual(float(np.median(f.points[f.valid, 2])), expected, delta=.005)

    def test_noise_filter_preserves_edges_and_invalid_holes(self):
        rng = np.random.default_rng(9)
        disparity = np.full((100, 140), 16., np.float32)
        disparity[:, 70:] = 32
        disparity += rng.normal(0, .45, disparity.shape).astype(np.float32)
        valid = np.zeros(disparity.shape, bool)
        valid[10:90, 10:130] = True
        valid[40:50, 30:40] = False
        valid[1:3, 1:3] = True
        cleaned, mask = clean_disparity(disparity, valid)
        self.assertFalse(mask[1:3, 1:3].any())
        self.assertFalse(mask[40:50, 30:40].any())
        self.assertLess(np.std(cleaned[20:80, 45:65]), np.std(disparity[20:80, 45:65]) * .6)
        self.assertAlmostEqual(float(np.median(cleaned[20:80, 65:70])), 16., delta=.3)
        self.assertAlmostEqual(float(np.median(cleaned[20:80, 70:75])), 32., delta=.3)

    def test_sensor_grain_is_not_treated_as_scene_texture(self):
        rng = np.random.default_rng(4)
        wall = np.clip(100 + rng.normal(0, 4, (120, 200)), 0, 255).astype(np.uint8)
        self.assertLess(texture_mask(wall).mean(), .01)
        board = np.tile(((np.indices((120, 200))[1] // 10) % 2 * 200).astype(np.uint8), (1, 1))
        self.assertGreater(texture_mask(board).mean(), .8)

    def test_temporal_filter_is_disabled_when_camera_view_moves(self):
        calibration, left, right = fixture()
        engine = DepthEngine(calibration, disparities=64, temporal=True)
        engine.compute(left, right)
        engine.compute(left, right)
        self.assertTrue(engine.last_stationary)
        shifted_left, shifted_right = np.zeros_like(left), np.zeros_like(right)
        shifted_left[:, :-5], shifted_right[:, :-5] = left[:, 5:], right[:, 5:]
        engine.compute(shifted_left, shifted_right)
        self.assertFalse(engine.last_stationary)

    def test_surface_export_roundtrip_and_rotation(self):
        calibration, left, right = fixture()
        f = DepthEngine(calibration, disparities=64).compute(left, right)
        points, colors, faces = surface_from_frame(f)
        self.assertGreater(len(faces), 100)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "поверхность.ply"
            export_surface(path, points, colors, faces)
            loaded, rgb, units = read_model(path)
            self.assertEqual(units, "metres")
            np.testing.assert_allclose(loaded, points, atol=1e-5)
            np.testing.assert_array_equal(rgb, colors)
        first = render_cloud(points, colors, 400, 300, yaw=0)
        second = render_cloud(points, colors, 400, 300, yaw=.6)
        self.assertEqual(first.shape, (300, 400, 3))
        self.assertGreater(np.count_nonzero(first != second), 1000)

    def test_relative_depth_has_correct_direction_and_units(self):
        image = np.zeros((40, 80, 3), np.uint8)
        prediction = np.tile(np.linspace(1, 5, 80, dtype=np.float32), (40, 1))
        f = relative_frame(image, prediction)
        self.assertEqual(f.units, "relative")
        self.assertGreater(float(f.points[20, 5, 2]), float(f.points[20, 75, 2]))
        self.assertTrue(np.isfinite(f.points).all())

    def test_capture_queue_retains_latest_pair_and_releases_camera(self):
        pump = CapturePump(CameraConfig(width=80, height=64), process_target=continuous_reader)
        pump.start()
        try:
            deadline = time.perf_counter() + 4
            while pump.count < 20 and time.perf_counter() < deadline:
                time.sleep(.01)
            pair, _, sequence = pump.take(Event())
            self.assertGreaterEqual(sequence, 19)
            self.assertEqual(int(pair[0][0, 0, 0]), sequence % 256)
            self.assertLessEqual(pump.frames.qsize(), 1)
        finally:
            pump.stop_event.set()
            pump.join(3)
        self.assertFalse(pump.is_alive())
        self.assertIsNone(pump.child)

    def test_worker_export_contains_model_raw_pair_and_counters(self):
        calibration, left, right = fixture()
        with TemporaryDirectory() as directory:
            worker = CameraWorker(directory, CameraConfig(), (9, 6), 25., 8., 64,
                                  processing_width=192, near_priority=False)
            worker.raw = left, right
            worker.apply_calibration(calibration)
            worker.frame = worker.engine.compute(left, right)
            worker.frame_raw = worker.raw
            worker.received_frames, worker.depth_frames, worker.integrated_frames = 10, 4, 3
            worker.export()
            output = next((Path(directory) / "output").iterdir())
            for name in ("visible_surface.ply", "visible_cloud.ply", "raw_left.png", "raw_right.png", "raw_calibration.npz"):
                self.assertTrue((output / name).exists())
            data = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(data["received_stereo_pairs"], 10)
            self.assertEqual(data["depth_frames"], 4)
            self.assertEqual(data["map_frames"], 3)
            self.assertEqual(data["processing_size"], [192, 120])
            worker.close_maps()


if __name__ == "__main__":
    unittest.main()
