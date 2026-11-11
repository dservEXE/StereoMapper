"""Distance between two picked cloud points and how it is written."""
import unittest

import numpy as np

from stereo.cloud_measure import format_distance, point_distance


class PointDistanceTests(unittest.TestCase):
    def test_distance_is_straight_line_in_three_dimensions(self):
        self.assertAlmostEqual(point_distance([1, 2, 3], [4, 6, 15]), 13.0)
        self.assertEqual(point_distance([.5, .5, .5], [.5, .5, .5]), 0.0)

    def test_invalid_points_are_rejected(self):
        for first, second in (([0, 0], [1, 1, 1]), ([0, 0, np.nan], [1, 1, 1]), ([0, 0, 0], [np.inf, 0, 0])):
            with self.assertRaises(ValueError):
                point_distance(first, second)


class FormatDistanceTests(unittest.TestCase):
    def test_millimetres_below_one_metre_and_metres_from_one_metre(self):
        self.assertEqual(format_distance(.7211), "721.1 мм")
        self.assertEqual(format_distance(.0004), "0.4 мм")
        self.assertEqual(format_distance(1.0), "1.000 м")
        self.assertEqual(format_distance(2.34567), "2.346 м")

    def test_value_that_rounds_to_one_metre_is_written_in_metres(self):
        self.assertEqual(format_distance(.99996), "1.000 м")

    def test_millimetre_clouds_are_converted(self):
        self.assertEqual(format_distance(721.1, "millimetres"), "721.1 мм")
        self.assertEqual(format_distance(1500, "millimetres"), "1.500 м")

    def test_clouds_without_scale_keep_their_own_units(self):
        self.assertIn("усл. ед.", format_distance(.5, "relative"))
        self.assertNotIn("мм", format_distance(.5, "unknown"))

    def test_negative_or_infinite_distance_is_rejected(self):
        for value in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                format_distance(value)


if __name__ == "__main__":
    unittest.main()
