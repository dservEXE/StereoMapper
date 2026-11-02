import unittest

import cv2
import numpy as np

from stereo.registration import refine_icp, rigid_transform, transform_points


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(231)
        self.points = self.rng.uniform([-1., -.6, -.4], [1., .6, .4], (700, 3))
        self.pose = np.eye(4)
        self.pose[:3, :3] = cv2.Rodrigues(np.array([.3, -.7, .2]))[0]
        self.pose[:3, 3] = [.25, -.32, 1.4]

    def test_landmarks_recover_rotation_translation_and_metric_scale(self):
        target = transform_points(self.points, self.pose)
        fitted = rigid_transform(self.points, target)
        np.testing.assert_allclose(fitted, self.pose, atol=1e-12)
        np.testing.assert_allclose(transform_points(self.points, fitted), target, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted[:3, :3]), 1., places=12)
        np.testing.assert_array_equal(self.points.shape, target.shape)

    def test_three_planar_landmarks_are_sufficient(self):
        source = np.array([[0., 0., 0.], [.5, 0., 0.], [.1, .8, 0.]])
        target = transform_points(source, self.pose)
        np.testing.assert_allclose(rigid_transform(source, target), self.pose, atol=1e-12)

    def test_a_reflection_is_never_returned_as_a_rotation(self):
        target = self.points * [-1, 1, 1]
        fitted = rigid_transform(self.points, target)
        self.assertAlmostEqual(np.linalg.det(fitted[:3, :3]), 1., places=12)
        residual = np.linalg.norm(transform_points(self.points, fitted) - target, axis=1)
        self.assertGreater(float(np.sqrt(np.mean(residual ** 2))), .2)

    def test_duplicate_collinear_and_nearly_collinear_landmarks_are_rejected(self):
        for points in (np.zeros((4, 3)), np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]]),
                       np.array([[0., 0., 0.], [1., 0., 0.], [2., 1e-8, 0.]])):
            with self.subTest(points=points.tolist()), self.assertRaises(ValueError):
                rigid_transform(points, points)

    def test_invalid_landmarks_are_rejected(self):
        for points in (np.zeros((2, 3)), np.zeros((4, 2)), [[0, 1, float("nan")]] * 3,
                       [[0, 1, float("inf")]] * 3, np.ones((3, 3), complex) * 1j):
            with self.subTest(points=points), self.assertRaises(ValueError):
                rigid_transform(points, self.points[:3])
        with self.assertRaises(ValueError):
            rigid_transform(self.points[:3], self.points[:4])

    def test_invalid_or_nonrigid_transforms_are_rejected(self):
        scaled, reflected, nonhomogeneous = np.eye(4), np.eye(4), np.eye(4)
        scaled[0, 0], reflected[0, 0], nonhomogeneous[3, 0] = 1.01, -1, .1
        for matrix in (np.eye(3), scaled, reflected, nonhomogeneous, np.full((4, 4), np.nan)):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                transform_points(self.points, matrix)
            with self.subTest(icp_matrix=matrix), self.assertRaises(ValueError):
                refine_icp(self.points, self.points, initial=matrix)

    def test_transform_does_not_mutate_input_and_allows_empty_cloud(self):
        original = self.points.copy()
        transform_points(self.points, self.pose)
        np.testing.assert_array_equal(self.points, original)
        self.assertEqual(transform_points(np.empty((0, 3)), self.pose).shape, (0, 3))

    def test_icp_refines_nearby_scans_without_scaling(self):
        source = self.points
        expected = np.eye(4)
        expected[:3, :3] = cv2.Rodrigues(np.array([.012, -.018, .009]))[0]
        expected[:3, 3] = [.025, -.017, .014]
        target = transform_points(source, expected)
        result = refine_icp(source, target)
        self.assertTrue(result.usable, result.reason)
        self.assertTrue(result.converged)
        self.assertLess(result.after_rmse, result.before_rmse * .01)
        np.testing.assert_allclose(result.transform, expected, atol=1e-7)
        self.assertGreater(result.correspondences, 400)

    def test_noisy_partial_overlap_and_outliers_keep_metric_alignment(self):
        expected = self.pose
        target = transform_points(self.points[:500], expected)
        target += self.rng.normal(0, .001, target.shape)
        target = np.vstack((target, self.rng.uniform([3., 3., 3.], [6., 6., 6.], (200, 3))))
        source = np.vstack((self.points, self.rng.uniform([-6., -6., -6.], [-3., -3., -3.], (200, 3))))
        initial = self.pose.copy()
        initial[:3, 3] += [.018, -.012, .009]
        result = refine_icp(source, target, initial=initial, max_distance=.10)
        self.assertTrue(result.usable, result.reason)
        self.assertLess(result.after_rmse, result.before_rmse * .15)
        np.testing.assert_allclose(result.transform, expected, atol=.0006)
        self.assertGreater(result.correspondences, 250)
        np.testing.assert_allclose(initial[:3, 3], self.pose[:3, 3] + [.018, -.012, .009])

    def test_disjoint_scans_do_not_report_success_or_change_initial_pose(self):
        target = self.points + [20., 0., 0.]
        initial = np.eye(4)
        initial[:3, 3] = [0., .1, 0.]
        result = refine_icp(self.points, target, initial=initial)
        self.assertFalse(result.usable)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "insufficient_overlap")
        np.testing.assert_array_equal(result.transform, initial)

    def test_a_few_accidental_matches_are_insufficient_overlap(self):
        target = np.vstack((self.points[:10], self.points[10:] + [20., 0., 0.]))
        result = refine_icp(self.points, target, max_distance=.01)
        self.assertFalse(result.usable)
        self.assertEqual(result.reason, "insufficient_overlap")
        np.testing.assert_array_equal(result.transform, np.eye(4))

    def test_already_aligned_cloud_does_not_move(self):
        result = refine_icp(self.points, transform_points(self.points, self.pose), initial=self.pose)
        self.assertTrue(result.usable, result.reason)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        np.testing.assert_array_equal(result.transform, self.pose)

    def test_no_progress_keeps_the_initial_pose(self):
        source = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        # The only discrepancy is scale, which rigid ICP must not fit.
        target = source * 1.01
        result = refine_icp(source, target)
        self.assertFalse(result.usable)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "no_improvement")
        np.testing.assert_array_equal(result.transform, np.eye(4))
        self.assertEqual(result.before_rmse, result.after_rmse)

    def test_sampling_and_iteration_limits_are_observed(self):
        source = self.rng.uniform(-1, 1, (3000, 3))
        target = source + [.018, -.012, .01]
        result = refine_icp(source, target, max_iterations=1, sample_limit=400, max_distance=.15)
        self.assertLessEqual(result.iterations, 1)
        self.assertLessEqual(result.correspondences, 400)
        self.assertTrue(np.isfinite(result.transform).all())
        self.assertAlmostEqual(np.linalg.det(result.transform[:3, :3]), 1., places=10)
        if result.usable:
            self.assertLess(result.after_rmse, result.before_rmse)
        else:
            np.testing.assert_array_equal(result.transform, np.eye(4))

    def test_six_point_cloud_can_be_refined(self):
        source = self.points[:6]
        target = source + [.01, -.005, .003]
        result = refine_icp(source, target, max_distance=.06, sample_limit=6)
        self.assertTrue(result.usable, result.reason)
        np.testing.assert_allclose(transform_points(source, result.transform), target, atol=1e-12)

    def test_degenerate_icp_clouds_and_invalid_options(self):
        result = refine_icp(np.zeros((20, 3)), np.zeros((20, 3)))
        self.assertFalse(result.usable)
        self.assertEqual(result.reason, "degenerate_geometry")
        for options in (dict(max_distance=0), dict(max_distance=-1), dict(max_distance=float("nan")),
                        dict(max_distance=float("inf")), dict(max_iterations=0), dict(max_iterations=1.5),
                        dict(sample_limit=5), dict(sample_limit=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                refine_icp(self.points, self.points, **options)


if __name__ == "__main__":
    unittest.main()
