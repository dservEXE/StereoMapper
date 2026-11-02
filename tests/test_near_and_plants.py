from copy import deepcopy
import unittest

import cv2
import numpy as np

from stereo.calibration import Calibration
from stereo.camera import CameraConfig
from stereo.depth import DepthEngine
from stereo.mapping import VisualOdometry, top_view
from stereo.precision import make_precision_snapshot, verify_stationary
from stereo.surfaces import plane_estimates
from test_core import fixture


def close_fixture(fx=620., shift=372):
    size = (640, 360)
    K = np.array([[fx, 0, 320], [0, fx, 180], [0, 0, 1.]])
    D, R, T = np.zeros(5), np.eye(3), np.array([[-.06], [0], [0]])
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(K, D, K, D, size, R, T, alpha=0)
    c = Calibration(size, K, D, K, D, R, T, r1, r2, p1, p2, q,
                    np.array(roi1), np.array(roi2), .1, .1, CameraConfig().signature())
    gray = np.random.default_rng(72).integers(0, 256, (360, 640), dtype=np.uint8)
    left = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), .6), cv2.COLOR_GRAY2BGR)
    right = np.zeros_like(left)
    right[:, :-shift] = left[:, shift:]
    return c, left, right


class OneRegion:
    def processImage(self, image):
        return np.zeros(image.shape[:2], np.int32)


class NearAndPlantTests(unittest.TestCase):
    def test_balanced_search_reaches_ten_cm_beyond_first_band(self):
        c, left, right = close_fixture()
        engine = DepthEngine(c, disparities=128, quality="balanced", near_priority=True, temporal=False)
        self.assertGreater(engine.search_limit, 372)
        frame = engine.compute(left, right)
        self.assertGreater(frame.valid.sum(), 2000)
        self.assertAlmostEqual(float(np.median(frame.points[frame.valid, 2])), .10, delta=.002)
        self.assertTrue(engine.distance_report()[0]["search_available"])

    def test_scaling_preserves_wide_rectified_projection_and_metric_coordinates(self):
        c, _, _ = fixture()
        wide = c.scaled_to(c.size, alpha=1)
        scaled = wide.scaled_to((192, 120))
        np.testing.assert_allclose(scaled.P1[:2], wide.P1[:2] * .5)
        point = np.array([.02, .01, .15, 1.])
        for calibration in (wide, scaled):
            a, b = calibration.P1 @ point, calibration.P2 @ point
            u, v, d = a[0]/a[2], a[1]/a[2], a[0]/a[2]-b[0]/b[2]
            recovered = calibration.Q @ np.array([u, v, d, 1.])
            np.testing.assert_allclose(recovered[:3]/recovered[3], point[:3], atol=1e-8)

    def test_large_planar_interior_is_estimated_separately_from_valid_depth(self):
        yy, xx = np.indices((100, 140))
        disparity = (16 + xx*.03 + yy*.01).astype(np.float32)
        valid = np.ones(disparity.shape, bool)
        valid[20:80, 30:110] = False
        original = valid.copy()
        image = np.full((*valid.shape, 3), 100, np.uint8)
        estimate, mask = plane_estimates(disparity, valid, image, np.ones_like(valid), OneRegion())
        self.assertTrue(mask[30:70, 40:100].all())
        np.testing.assert_allclose(estimate[mask], disparity[mask], atol=1e-4)
        np.testing.assert_array_equal(valid, original)
        self.assertFalse((mask & valid).any())
        sparse = np.zeros_like(valid)
        sparse[:, :10] = True
        _, unavailable = plane_estimates(disparity, sparse, image, np.ones_like(valid), OneRegion())
        self.assertFalse(unavailable.any())
        disparity[:, 70:] += 20
        _, mixed = plane_estimates(disparity, valid, image, np.ones_like(valid), OneRegion())
        self.assertFalse(mixed.any())

    def test_surround_map_keeps_points_behind_and_to_both_sides(self):
        points = np.array([[x, 0, z] for x in (-1., 1.) for z in (-1., 1.)])
        image = top_view(points, extent=2., surround=True)
        marked = np.all(image == (220, 197, 82), axis=2)
        self.assertTrue(marked[:250, :250].any())
        self.assertTrue(marked[:250, 250:].any())
        self.assertTrue(marked[250:, :250].any())
        self.assertTrue(marked[250:, 250:].any())

    def test_saved_view_restores_tracking_after_reference_is_lost(self):
        c, left, right = fixture()
        frame = DepthEngine(c, disparities=64, temporal=False).compute(left, right)
        odometry = VisualOdometry(c.P1[:, :3])
        odometry.update(frame)
        unrelated = deepcopy(frame)
        unrelated.left = np.random.default_rng(30).integers(0, 256, left.shape, dtype=np.uint8)
        self.assertTrue(odometry.set_reference(unrelated))
        pose, message = odometry.update(frame)
        self.assertIsNotNone(pose)
        self.assertEqual(odometry.relocalizations, 1)
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-4)

    def test_full_resolution_snapshot_measures_a_one_mm_segment(self):
        c, left, right = close_fixture(fx=800., shift=480)
        result = make_precision_snapshot(c, [(left.copy(), right.copy()) for _ in range(5)], wide_view=False)
        pair = result.length((580, 180), (588, 180))
        self.assertAlmostEqual(pair["length_mm"], 1., delta=.02)
        self.assertEqual(pair["samples"], 5)
        self.assertLess(pair["repeatability_mm"], .01)
        self.assertIsNone(result.frame.estimated_points)
        shifted = np.roll(left, 4, axis=1)
        with self.assertRaises(ValueError):
            verify_stationary(left, shifted)


if __name__ == "__main__":
    unittest.main()
