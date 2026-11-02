from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from stereo.calibration import Calibration, CalibrationSession, read_image, write_image
from stereo.camera import CameraConfig
from stereo.depth import DepthEngine, consistency_mask, measure
from stereo.mapping import estimate_motion, export_ply, VoxelMap, VisualOdometry


def fixture():
    size = (384, 240)
    K = np.array([[240., 0, 192], [0, 240., 120], [0, 0, 1]])
    D, R, T = np.zeros(5), np.eye(3), np.array([[-.06], [0], [0]])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, D, K, D, size, R, T, alpha=0)
    calibration = Calibration(size, K, D, K, D, R, T, R1, R2, P1, P2, Q,
                              np.array(roi1), np.array(roi2), .1, .1, CameraConfig().signature())
    rng = np.random.default_rng(72)
    gray = rng.integers(0, 256, (240, 384), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), .6)
    left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    right = np.empty_like(left)
    right[:, :-16] = left[:, 16:]
    right[:, -16:] = 0
    return calibration, left, right


class StereoTests(unittest.TestCase):
    def test_metric_depth_on_known_stereo_pair(self):
        calibration, left, right = fixture()
        frame = DepthEngine(calibration, disparities=64).compute(left, right)
        self.assertGreater(frame.valid.mean(), .45)
        self.assertAlmostEqual(float(np.median(frame.points[frame.valid, 2])), .9, delta=.02)
        self.assertTrue(np.isnan(frame.points[~frame.valid]).all())
        ys, xs = np.where(frame.valid)
        measurement = measure(frame, int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement["z"], .9, delta=.02)

    def test_occlusion_and_reciprocity(self):
        a = np.full((5, 30), 4., np.float32)
        b = np.full((5, 30), -4., np.float32)
        valid = consistency_mask(a, b)
        self.assertFalse(valid[:, :4].any())
        self.assertTrue(valid[:, 4:].all())
        b[:, 10] = -8
        self.assertFalse(consistency_mask(a, b)[:, 14].any())

    def test_camera_motion_with_outliers(self):
        rng = np.random.default_rng(21)
        objects = rng.uniform([-1, -.6, 1.5], [1, .6, 4.5], (150, 3)).astype(np.float32)
        K = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]])
        r = np.array([.015, -.025, .01])
        t = np.array([.06, .01, .02])
        pixels, _ = cv2.projectPoints(objects, r, t, K, None)
        pixels = pixels.reshape(-1, 2)
        pixels += rng.normal(0, .15, pixels.shape).astype(np.float32)
        pixels[:25] = rng.uniform([0, 0], [640, 480], (25, 2))
        motion, inliers = estimate_motion(objects, pixels, K)
        self.assertIsNotNone(motion)
        self.assertGreater(inliers, 110)
        np.testing.assert_allclose(motion[:3, 3], t, atol=.005)
        np.testing.assert_allclose(motion[:3, :3], cv2.Rodrigues(r)[0], atol=.005)
        np.testing.assert_allclose(np.linalg.inv(motion) @ motion, np.eye(4), atol=1e-8)

    def test_bad_motion_is_not_integrated(self):
        K = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]])
        motion, _ = estimate_motion(np.zeros((5, 3)), np.zeros((5, 2)), K)
        self.assertIsNone(motion)

    def test_visual_odometry_tracks_and_freezes_on_loss(self):
        calibration, left, right = fixture()
        engine = DepthEngine(calibration, disparities=64)
        odometry = VisualOdometry(calibration.P1[:, :3])
        pose, _ = odometry.update(engine.compute(left, right))
        self.assertIsNotNone(pose)
        shifted_left, shifted_right = np.zeros_like(left), np.zeros_like(right)
        shifted_left[:, :-4], shifted_right[:, :-4] = left[:, 4:], right[:, 4:]
        pose, _ = odometry.update(engine.compute(shifted_left, shifted_right))
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose[0, 3], .015, delta=.008)
        last_pose = pose.copy()
        blank = np.zeros_like(left)
        pose, _ = odometry.update(engine.compute(blank, blank))
        self.assertIsNone(pose)
        np.testing.assert_array_equal(odometry.world_pose, last_pose)

    def test_stereo_calibration_recovers_metric_baseline(self):
        calibration, _, _ = fixture()
        rng = np.random.default_rng(12)
        objects = np.zeros((54, 3), np.float32)
        objects[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * .025
        with TemporaryDirectory() as directory:
            session = CalibrationSession(directory)
            session.size, session.signature = calibration.size, calibration.signature
            for _ in range(20):
                rotation = rng.uniform(-.45, .45, 3)
                translation = np.array([rng.uniform(-.15, -.04), rng.uniform(-.10, -.02), rng.uniform(.55, 1.1)])
                a, _ = cv2.projectPoints(objects, rotation, translation, calibration.K1, calibration.D1)
                b, _ = cv2.projectPoints(objects, rotation, translation + calibration.T.ravel(), calibration.K2, calibration.D2)
                session.points.append((a, b))
            recovered = session.solve()
            self.assertAlmostEqual(recovered.baseline, .06, delta=.0002)
            self.assertLess(recovered.epipolar_error, .01)

    def test_resolution_and_calibration_file(self):
        calibration, left, right = fixture()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "калибровка.npz"
            calibration.save(path)
            loaded = Calibration.load(path)
            np.testing.assert_allclose(loaded.Q, calibration.Q)
            with self.assertRaises(ValueError):
                DepthEngine(loaded).compute(left[:, :300], right[:, :300])
            calibration.T = -calibration.T
            calibration.save(path)
            with self.assertRaises(ValueError):
                Calibration.load(path)

    def test_export_and_unicode_paths(self):
        calibration, left, right = fixture()
        frame = DepthEngine(calibration, disparities=64).compute(left, right)
        cloud = VoxelMap()
        pose = np.eye(4)
        pose[0, 3] = .5
        cloud.add(frame, pose)
        points, colors = cloud.arrays()
        self.assertGreater(len(points), 20)
        self.assertTrue(np.isfinite(points).all())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "снимок.png"
            write_image(path, left)
            np.testing.assert_array_equal(read_image(path), left)
            path = Path(directory) / "карта.ply"
            export_ply(path, points, colors)
            content = path.read_text(encoding="ascii")
            self.assertIn(f"element vertex {len(points)}", content)
            self.assertEqual(content.split("end_header\n")[1].count("\n"), len(points))

    def test_calibration_requires_enough_views(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                CalibrationSession(directory).solve()


if __name__ == "__main__":
    unittest.main()
