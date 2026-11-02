import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from stereo.depth import DepthEngine, clean_disparity, smooth_preview, texture_mask, scene_color_limits, measure
from test_core import fixture


class DepthQualityTests(unittest.TestCase):
    def measurement_plane(self):
        yy, xx = np.indices((9, 11))
        points = np.stack((xx * .001, yy * .001, np.ones_like(xx)), axis=2)
        return SimpleNamespace(points=points, valid=np.ones(xx.shape, bool))

    def test_measurement_keeps_clicked_ray_with_asymmetric_edge_support(self):
        frame = self.measurement_plane()
        frame.valid[:, :4] = False
        frame.points[~frame.valid] = np.nan
        result = measure(frame, 4, 4, radius=2)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['point'], [.004, .004, 1.], atol=1e-10)

    def test_measurement_uses_only_accepted_neighbours(self):
        frame = self.measurement_plane()
        frame.valid[:] = False
        frame.valid[4, 3:8] = True
        frame.points[~frame.valid] *= 1.02
        result = measure(frame, 5, 4, radius=2)
        self.assertEqual(result['samples'], 5)
        np.testing.assert_allclose(result['point'], [.005, .004, 1.], atol=1e-10)
        frame.valid[4, 3] = False
        self.assertIsNone(measure(frame, 5, 4, radius=2))

    def test_measurement_rejects_invalid_center_geometry(self):
        for z in (np.nan, np.inf, 0., -1.):
            with self.subTest(z=z):
                frame = self.measurement_plane()
                frame.points[4, 5, 2] = z
                self.assertIsNone(measure(frame, 5, 4, radius=2))

    def test_robust_depth_is_backprojected_on_the_selected_ray(self):
        frame = self.measurement_plane()
        frame.points[4, 5] *= 1.01
        before = frame.points.copy()
        result = measure(frame, 5, 4, radius=2)
        np.testing.assert_allclose(result['point'], [.005, .004, 1.], atol=1e-10)
        np.testing.assert_array_equal(frame.points, before)

    def test_soft_structure_survives_but_sensor_noise_does_not(self):
        ramp = np.tile(np.arange(180, dtype=np.float32) * 1.1 + 25, (100, 1)).astype(np.uint8)
        self.assertGreater(texture_mask(ramp)[15:-15, 15:-15].mean(), .9)
        noise = np.random.default_rng(7).normal(100, 4, ramp.shape).clip(0, 255).astype(np.uint8)
        self.assertLess(texture_mask(noise).mean(), .01)

    def test_preview_fills_small_holes_without_changing_measurements_or_large_unknowns(self):
        disparity = np.full((80, 100), 16., np.float32)
        valid = np.ones(disparity.shape, bool)
        valid[20:23, 20:23] = False
        valid[40:60, 40:60] = False
        disparity[~valid] = np.nan
        original_disp, original_mask = disparity.copy(), valid.copy()
        display, filled = smooth_preview(disparity, valid, np.full(valid.shape, 100, np.uint8))
        self.assertTrue(filled[20:23, 20:23].all())
        self.assertFalse(filled[40:60, 40:60].any())
        np.testing.assert_array_equal(valid, original_mask)
        np.testing.assert_array_equal(disparity, original_disp)
        np.testing.assert_allclose(display[filled], 16.)
        self.assertFalse((filled & valid).any())

    def test_preview_does_not_bridge_a_depth_or_brightness_edge(self):
        disparity = np.full((50, 70), 16., np.float32)
        disparity[:, 35:] = 48
        valid = np.ones(disparity.shape, bool)
        valid[20:23, 34:37] = False
        gray = np.full(disparity.shape, 100, np.uint8)
        _, filled = smooth_preview(disparity, valid, gray)
        self.assertFalse(filled[20:23, 35].any())
        disparity.fill(16.)
        gray[20:23, 34:37] = 240
        _, filled = smooth_preview(disparity, valid, gray)
        self.assertFalse(filled.any())

    def test_small_wrong_depth_speckle_is_removed_even_when_touching_a_surface(self):
        disparity = np.full((60, 80), 16., np.float32)
        disparity[20:23, 30:33] = 48
        _, valid = clean_disparity(disparity, np.ones(disparity.shape, bool), 50)
        self.assertFalse(valid[21, 31])
        self.assertTrue(valid[10:15, 10:15].all())

    def test_vertical_pair_mismatch_cannot_enter_metric_depth(self):
        calibration, left, right = fixture()
        shifted = np.zeros_like(right)
        shifted[4:] = right[:-4]
        engine = DepthEngine(calibration, disparities=64, temporal=False)
        frame = engine.compute(left, shifted)
        self.assertIsNotNone(frame.quality_warning)
        self.assertGreater(frame.alignment_error, 3)
        self.assertFalse(frame.valid.any())
        self.assertTrue(np.isnan(frame.points).all())
        self.assertIsNone(measure(frame, 200, 100))

    def test_color_range_does_not_amplify_flat_surface_noise(self):
        z = np.random.default_rng(1).normal(.8, .001, (50, 80)).astype(np.float32)
        low, high = scene_color_limits(z, np.ones(z.shape, bool), 8.)
        self.assertGreater(high - low, .19)
        self.assertLess(low, .8)
        self.assertGreater(high, .8)


if __name__ == "__main__":
    unittest.main()
