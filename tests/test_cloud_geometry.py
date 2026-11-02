"""Headless tests for the same camera used by drawing, selection and picking."""

import unittest

import numpy as np

from stereo.cloud_geometry import (
    box_mask, cloud_bounds, pick_point, points_in_polygon, project_points,
    render_points, selection_mask,
)
from stereo.trackball import arcball_rotation, axis_rotation


class CloudGeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera = dict(rotation=np.eye(3), zoom=1., pan=(0, 0), center=np.zeros(3), extent=1.)
        self.rectangle = np.array([[95, 95], [105, 95], [105, 105], [95, 105]])

    def test_bounds_ignore_invalid_and_handle_empty_or_single_point(self):
        center, extent = cloud_bounds([[1, 2, 3], [3, 6, 5], [np.nan, 90, 50]])
        np.testing.assert_allclose(center, [2, 4, 4])
        self.assertEqual(extent, 4.)
        center, extent = cloud_bounds([])
        np.testing.assert_array_equal(center, np.zeros(3))
        self.assertEqual(extent, 1.)
        self.assertEqual(cloud_bounds([[3, 3, 3]])[1], 1e-5)

    def test_projection_is_perspective_with_shared_pan_and_zoom(self):
        points = np.array([[.2, .1, 0], [.2, .1, 1], [np.inf, 0, 0], [0, 0, -2]])
        camera = dict(self.camera, zoom=2., pan=(12, -9))
        xy, depth, valid = project_points(points, 200, 100, **camera)
        np.testing.assert_allclose(xy[:2], [[145, 57.5], [134, 52.]])
        np.testing.assert_allclose(depth[[0, 1, 3]], [2, 3, 0])
        np.testing.assert_array_equal(valid, [True, True, False, False])
        self.assertTrue(np.isnan(xy[2:]).all())
        polygon = [[143, 55], [147, 55], [147, 59], [143, 59]]
        np.testing.assert_array_equal(selection_mask(points, polygon, 200, 100, **camera), [True, False, False, False])
        self.assertEqual(pick_point(points, 145, 57.5, 200, 100, radius=1, **camera), 0)

    def test_through_selection_and_visible_surface_selection_differ(self):
        points = np.array([[0, 0, .4], [0, 0, -.4], [.4, 0, 0]])
        through = selection_mask(points, self.rectangle, 200, 200, **self.camera)
        visible = selection_mask(points, self.rectangle, 200, 200, visible_only=True, **self.camera)
        np.testing.assert_array_equal(through, [True, True, False])
        np.testing.assert_array_equal(visible, [False, True, False])
        self.assertEqual(pick_point(points, 100, 100, 200, 200, **self.camera), 1)

    def test_depth_range_is_inclusive_and_does_not_remove_occluders(self):
        points = np.array([[0, 0, .4], [0, 0, -.4]])
        only_rear = selection_mask(points, self.rectangle, 200, 200, depth_range=(2.4, 2.4), **self.camera)
        np.testing.assert_array_equal(only_rear, [True, False])
        rear_visible = selection_mask(points, self.rectangle, 200, 200, depth_range=(2.3, 2.5),
                                      visible_only=True, **self.camera)
        self.assertFalse(rear_visible.any())

    def test_rotation_changes_selection_in_three_dimensions(self):
        points = np.array([[0, 0, .4], [.4, 0, 0], [0, .4, 0]])
        camera = dict(self.camera, rotation=axis_rotation(1, np.pi / 2))
        xy, _, _ = project_points(points, 200, 200, **camera)
        np.testing.assert_allclose(xy[0], [166, 100])
        polygon = [[164, 98], [168, 98], [168, 102], [164, 102]]
        np.testing.assert_array_equal(selection_mask(points, polygon, 200, 200, **camera), [True, False, False])

    def test_polygon_includes_all_edges_and_nonconvex_lasso(self):
        lasso = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4], [0, 0]]
        xy = [[0, 0], [4, .5], [.5, 4], [1, 1], [.5, 2], [2, 2], [5, 0], [np.nan, 0]]
        np.testing.assert_array_equal(points_in_polygon(xy, lasso), [True, True, True, True, True, False, False, False])
        self.assertFalse(points_in_polygon(xy, []).any())
        self.assertFalse(points_in_polygon(xy, [[0, 0], [1, 1], [2, 2]]).any())

    def test_polygon_even_odd_handles_self_intersection(self):
        bowtie = [[0, 0], [4, 4], [0, 4], [4, 0]]
        np.testing.assert_array_equal(points_in_polygon([[2, .5], [2, 3.5], [.5, 2], [2, 2]], bowtie),
                                      [True, True, False, True])

    def test_full_cloud_selection_and_bounds_are_independent_of_render_budget(self):
        points = np.array([[-.4, 0, 0], [0, 0, 0], [.4, 0, 0]])
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        image = render_points(points, colors, 200, 200, max_points=1, point_size=1, axes=False)
        xy, _, _ = project_points(points, 200, 200)
        self.assertEqual(np.count_nonzero(np.any(image != [24, 20, 16], axis=2)), 1)
        np.testing.assert_array_equal(image[int(round(xy[0, 1])), int(round(xy[0, 0]))], [0, 0, 255])
        np.testing.assert_array_equal(selection_mask(points, self.rectangle, 200, 200), [False, True, False])

    def test_render_zbuffer_handles_overlapping_splats_not_just_centers(self):
        # Near and far centers round to adjacent pixels; their splats overlap.
        points = np.array([[0, 0, -.4], [2.4 / 165., 0, .4]])
        colors = np.array([[255, 0, 0], [0, 0, 255]])
        camera = self.camera
        image = render_points(points, colors, 100, 100, point_size=3, axes=False, **camera)
        for column in (49, 50, 51):
            np.testing.assert_array_equal(image[50, column], [0, 0, 255])
        np.testing.assert_array_equal(image[50, 52], [255, 0, 0])
        reverse = render_points(points[::-1], colors[::-1], 100, 100, point_size=3, axes=False, **camera)
        np.testing.assert_array_equal(reverse, image)

    def test_render_selection_highlight_respects_depth(self):
        points = np.array([[0, 0, -.4], [0, 0, .4]])
        colors = np.full((2, 3), 190)
        rear_selected = render_points(points, colors, 200, 200, selection=[False, True], axes=False, **self.camera)
        np.testing.assert_array_equal(rear_selected[100, 100], [190, 190, 190])
        front_selected = render_points(points, colors, 200, 200, selection=[True, False], axes=False, **self.camera)
        np.testing.assert_array_equal(front_selected[100, 100], [25, 220, 255])

    def test_pick_uses_perspective_and_screen_distance_then_depth(self):
        points = np.array([[0, 0, .5], [0, 0, -.5], [.05, 0, -.5]])
        self.assertEqual(pick_point(points, 100, 100, 200, 200, **self.camera), 1)
        self.assertEqual(pick_point(points, 111, 100, 200, 200, radius=2, **self.camera), 2)
        self.assertIsNone(pick_point(points, 150, 150, 200, 200, radius=2, **self.camera))

    def test_box_selection_is_world_space_and_inclusive(self):
        points = [[0, 0, 0], [1, 2, 3], [1.01, 2, 3], [np.nan, 0, 0]]
        np.testing.assert_array_equal(box_mask(points, [1, 2, 3], [0, 0, 0]), [True, True, False, False])

    def test_empty_invalid_and_offscreen_clouds_are_safe(self):
        self.assertEqual(project_points([], 100, 100)[0].shape, (0, 2))
        self.assertEqual(selection_mask([], [], 100, 100).shape, (0,))
        self.assertIsNone(pick_point([], 1, 1, 100, 100))
        blank = render_points([], [], 100, 100, axes=False)
        self.assertEqual(blank.shape, (100, 100, 3))
        bad = [[np.nan, 0, 0], [np.inf, 0, 0], [1e100, 1e100, 0], [0, 0, -3]]
        image = render_points(bad, [[1, 2, 3]] * 4, 100, 100, axes=False, **self.camera)
        np.testing.assert_array_equal(image, blank)
        self.assertFalse(selection_mask(bad, [[0, 0], [100, 0], [100, 100]], 100, 100,
                                        visible_only=True, **self.camera).any())

    def test_invalid_arguments_raise_clear_value_errors(self):
        for kwargs in (dict(zoom=0), dict(extent=-1), dict(rotation=[1, 2, 3]), dict(pan=(np.nan, 0))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                project_points([[0, 0, 0]], 100, 100, **kwargs)
        with self.assertRaises(ValueError):
            project_points([[0, 0]], 100, 100)
        with self.assertRaises(ValueError):
            project_points([[0, 0, 0]], 0, 100)
        with self.assertRaises(ValueError):
            points_in_polygon([[0, 0]], [[0, 0], [1, 0], [0, np.nan]])
        with self.assertRaises(ValueError):
            selection_mask([[0, 0, 0]], self.rectangle, 200, 200, depth_range=(3, 1))
        with self.assertRaises(ValueError):
            render_points([[0, 0, 0]], [[1, 2, 3]], 100, 100, selection=[])

    def test_arcball_center_tilts_and_edge_drags_roll_without_modifier(self):
        tilt = arcball_rotation((400, 300), (450, 300), 800, 600)
        self.assertGreater(abs(tilt[0, 2]), .1)
        edge_roll = arcball_rotation((750, 300), (700, 450), 800, 600)
        self.assertGreater(abs(edge_roll[0, 1]), .1)
        np.testing.assert_allclose(edge_roll[2], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(edge_roll.T @ edge_roll, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
