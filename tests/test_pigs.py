"""Pig weighing: the scene behind a picture, measuring in space, numbering, the session thread and the tab."""
import json
import os
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stereo.app import StereoApp
from stereo.calibration import latest_calibration
from stereo.camera import CameraConfig
from stereo.people_detection import SEGMENTER, WEIGHTS_FILE, model_path
from stereo.pig_database import PigDatabase, correction, export_csv, make_entry, summary_text
from stereo.pig_detection import DEPTH, DETECTOR, PigDetector, distinct_outlines, keep_distinct, missing_size_mb
from stereo.pig_drawing import draw_marks, girth_line, kg_text, label_text, outline_contours
from stereo.pig_measure import (BAD_POSE, BODY_DENSITY_KG_M3, DEPTH_TO_WIDTH, END_ON, LEGS_FACTOR, NO_DEPTH,
                                PigAccumulator, PigEstimate, PigSample, ellipse_perimeter, measure_pig, pig_shape,
                                seen_from_side, tape_mass)
from stereo.pig_scene import (CAMERA_UP, Floor, assumed_focal, fit_floor, floor_candidates, hidden_behind, scene_points,
                              segment_length, stereo_scale)
from stereo.pig_ui import NO_CAMERA, device_text, draw_pigs, parse_number, pig_row, progress_text, scale_text
from stereo.pig_tracking import IDENTITY, carried, compose, follow, follower, moved_shape, tracking_gray
from stereo.pig_weighing import (CUT, LEARNED, MARKED_SUFFIX, MEMORY_S, PICTURE, SEGMENT, SEGMENT_FAILED, STEREO, WEIGHED,
                                 WEIGHT, WEIGHT_FAILED, PigRow, PigSession, PigState, PigTracker, fitted, read_image,
                                 reading_order, write_result)
from stereo.worker import CameraWorker

SIZE = 700
FOCAL = 700.
MIDDLE = (SIZE - 1) / 2


def pig_mask(angle=0., tail=True, ears=True, centre=(350, 350), scale=1.) -> np.ndarray:
    """A pig seen from above: trunk 300 x 90, a narrower head on the left, a neck between them."""
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (350, 350), (150, 45), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (230, 305), (470, 395), 255, -1)
    cv2.ellipse(mask, (150, 350), (55, 30), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (180, 322), (215, 378), 255, -1)
    if ears:
        cv2.line(mask, (165, 330), (150, 290), 255, 7)
        cv2.line(mask, (165, 370), (150, 410), 255, 7)
    if tail:
        cv2.line(mask, (500, 350), (560, 340), 255, 5)
    matrix = cv2.getRotationMatrix2D((350, 350), angle, scale)
    matrix[:, 2] += (centre[0] - 350, centre[1] - 350)
    return cv2.warpAffine(mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0


def bent_mask() -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (350, 500), (220, 220), 0, 200, 340, 255, 80)
    return mask > 0


def from_above(mask, floor=2.5, back=2.05) -> np.ndarray:
    """A camera looking straight down: the floor at one distance, the back of the animal nearer."""
    return scene_points(np.where(mask, back, floor).astype(np.float32), FOCAL)


def from_side(mask, distance=3., drop=.6, wall=6.) -> np.ndarray:
    """A level camera: the floor `drop` metres below it, a wall behind, the flank of the animal at `distance`."""
    rows = np.arange(SIZE, dtype=np.float64)[:, None] - MIDDLE
    floor = np.where(rows > 0, drop * FOCAL / np.maximum(rows, 1e-6), np.inf)
    depth = np.broadcast_to(np.minimum(floor, wall), (SIZE, SIZE)).copy()
    depth[mask] = distance
    return scene_points(depth.astype(np.float32), FOCAL)


def slanted(mask, middle=3., step=.002) -> np.ndarray:
    """Everything on one plane that runs away from the camera to the right."""
    columns = np.arange(SIZE, dtype=np.float64)[None, :] - MIDDLE
    return scene_points(np.broadcast_to(middle + step * columns, (SIZE, SIZE)).astype(np.float32), FOCAL)


def place(point, points) -> np.ndarray:
    return points[int(round(point[1])), int(round(point[0]))].astype(np.float64)


def measured(mask, points, with_floor=True):
    floor = fit_floor(points, floor_candidates([mask]), [points[mask][::50].astype(np.float64)]) if with_floor else None
    return measure_pig(pig_shape(mask), points, mask, FOCAL, floor), floor


class ShapeTests(unittest.TestCase):
    def test_length_and_thickness_do_not_depend_on_how_the_pig_is_turned(self):
        for angle in (0, 30, 90, 137, 200):
            found = pig_shape(pig_mask(angle))
            self.assertAlmostEqual(found.length_px, 280, delta=8, msg=angle)
            self.assertAlmostEqual(found.girth_px, 90, delta=4, msg=angle)
            self.assertGreater(found.straightness, .98)

    def test_head_is_the_narrow_end_and_the_spine_runs_from_neck_to_rump(self):
        found = pig_shape(pig_mask())
        self.assertLess(found.head[0], found.neck[0])
        self.assertTrue(180 <= found.neck[0] <= 235, found.neck)
        xs = [point[0] for point in found.spine]
        self.assertEqual(xs, sorted(xs))
        self.assertAlmostEqual(xs[0], found.neck[0], delta=2)
        self.assertAlmostEqual(xs[-1], found.rump[0], delta=2)
        self.assertTrue(found.neck[0] < found.girth_point[0] < 350)
        self.assertGreater(pig_shape(pig_mask(180)).head[0], pig_shape(pig_mask(180)).rump[0])

    def test_tail_and_ears_do_not_change_the_measurement(self):
        full, bare = pig_shape(pig_mask()), pig_shape(pig_mask(tail=False, ears=False))
        self.assertAlmostEqual(full.length_px, bare.length_px, delta=4)
        self.assertAlmostEqual(full.girth_px, bare.girth_px, delta=2)
        self.assertLess(full.rump[0], 510)

    def test_specks_and_empty_outlines_give_nothing(self):
        self.assertIsNone(pig_shape(np.zeros((100, 100), bool)))
        speck = np.zeros((100, 100), bool)
        speck[10:15, 10:15] = True
        self.assertIsNone(pig_shape(speck))


class SceneTests(unittest.TestCase):
    def test_every_pixel_gets_a_place_in_space_and_goes_back_to_its_pixel(self):
        depth = np.full((480, 640), 2., np.float32)
        points = scene_points(depth, 500., (320., 240.))
        self.assertTrue(np.allclose(points[240, 320], (0, 0, 2)))
        self.assertTrue(np.allclose(points[240, 570], (1, 0, 2)))
        self.assertTrue(np.allclose(points[115, 320], (0, -.5, 2)))
        self.assertAlmostEqual(assumed_focal((768, 1024)), .693 * 1280)

    def test_segment_is_measured_with_each_end_at_its_own_distance(self):
        segment = ((100, 300), (450, 300))
        flat = scene_points(np.full((SIZE, SIZE), 2., np.float32), FOCAL)
        self.assertAlmostEqual(segment_length(flat, segment), 1., places=3)
        away = slanted(None)
        real = float(np.linalg.norm(place(segment[1], away) - place(segment[0], away)))
        self.assertAlmostEqual(segment_length(away, segment), real, delta=real * .01)
        self.assertGreater(real, 350 * 3 / FOCAL * 1.01)
        self.assertIsNone(segment_length(flat, ((-50, -50), (100, 100))))
        self.assertIsNone(segment_length(np.full((SIZE, SIZE, 3), np.nan, np.float32), segment))

    def test_segment_keeps_to_the_marked_thing_at_its_ends_and_ignores_an_animal_across_it(self):
        depth = np.full((SIZE, SIZE), 5., np.float32)
        depth[:, 100:451] = 2.
        segment = ((100, 300), (450, 300))
        self.assertAlmostEqual(segment_length(scene_points(depth, FOCAL), segment), 1., delta=.01)
        depth[:, 200:260] = 1.2
        self.assertAlmostEqual(segment_length(scene_points(depth, FOCAL), segment), 1., delta=.01)

    def test_stereo_pair_sets_the_scale_of_the_estimated_depth(self):
        estimated = np.random.default_rng(0).uniform(1., 4., (120, 160)).astype(np.float32)
        real = estimated * 1.3
        valid = np.ones(real.shape, bool)
        real[:20] = 50.
        valid[:5] = False
        self.assertAlmostEqual(stereo_scale(estimated, real, valid), 1.3, places=3)
        self.assertIsNone(stereo_scale(estimated, real, np.zeros(real.shape, bool)))
        self.assertIsNone(stereo_scale(estimated, estimated * 40, valid))

    def test_floor_is_looked_for_just_under_the_animals(self):
        mask = pig_mask()
        band = floor_candidates([mask])
        self.assertTrue(band.any())
        self.assertFalse((band & mask).any())
        # Under the body, and under the ear that sticks out above it, but nowhere above the animal.
        rows = np.nonzero(band.any(axis=1))[0]
        self.assertGreater(rows.min(), np.nonzero(mask.any(axis=1))[0].min())
        self.assertGreater(rows.max(), 410)
        self.assertEqual(floor_candidates([]).size, 0)

    def test_floor_under_a_camera_looking_down_faces_the_camera(self):
        mask = pig_mask()
        points = from_above(mask)
        floor = fit_floor(points, floor_candidates([mask]), [points[mask][::50].astype(np.float64)])
        self.assertTrue(np.allclose(floor.normal, (0, 0, -1), atol=.02))
        self.assertAlmostEqual(floor.offset, 2.5, delta=.02)
        self.assertAlmostEqual(float(np.median(floor.height(points[mask].astype(np.float64)))), .45, delta=.02)

    def test_a_wall_behind_the_animals_is_not_taken_for_the_floor(self):
        rng = np.random.default_rng(1)
        wall = np.column_stack((rng.uniform(-3, 3, 2000), rng.uniform(-2, 1, 2000), np.full(2000, 4.)))
        ground = np.column_stack((rng.uniform(-3, 3, 1200), np.full(1200, 1.), rng.uniform(2, 6, 1200)))
        points = np.vstack((wall, ground)).reshape(1, -1, 3)
        everything = np.ones(points.shape[:2], bool)
        animal = np.column_stack((rng.uniform(-.5, .5, 100), rng.uniform(.5, .9, 100), np.full(100, 2.)))
        self.assertTrue(np.allclose(fit_floor(points, everything).normal, (0, 0, -1), atol=.02))
        floor = fit_floor(points, everything, [animal])
        self.assertTrue(np.allclose(floor.normal, (0, -1, 0), atol=.02))
        self.assertAlmostEqual(floor.offset, 1., delta=.02)

    def test_too_little_floor_gives_no_floor(self):
        points = from_above(pig_mask())
        few = np.zeros((SIZE, SIZE), bool)
        few[0, :50] = True
        self.assertIsNone(fit_floor(points, few))

    def test_animal_covered_by_a_nearer_one_is_known_and_neighbours_side_by_side_are_not(self):
        rear, front = np.zeros((SIZE, SIZE), bool), np.zeros((SIZE, SIZE), bool)
        rear[300:400, 250:380] = True
        front[280:420, 380:600] = True
        depth = np.full((SIZE, SIZE), 3., np.float32)
        depth[rear], depth[front] = 2.5, 2.
        self.assertEqual(hidden_behind([rear, front], scene_points(depth, FOCAL)), [True, False])
        depth[rear] = 2.
        self.assertEqual(hidden_behind([rear, front], scene_points(depth, FOCAL)), [False, False])
        apart = np.zeros((SIZE, SIZE), bool)
        apart[100:200, 100:200] = True
        self.assertEqual(hidden_behind([apart, front], scene_points(depth, FOCAL)), [False, False])


class MeasureTests(unittest.TestCase):
    def test_from_above_the_thickness_in_the_picture_is_the_width(self):
        (sample, why), floor = measured(pig_mask(), from_above(pig_mask()))
        self.assertEqual(why, "")
        self.assertAlmostEqual(sample.length_m, 280 * 2.05 / FOCAL, delta=.03)
        self.assertAlmostEqual(sample.thickness_m, 90 * 2.05 / FOCAL, delta=.01)
        self.assertLess(sample.side_share, .02)
        self.assertAlmostEqual(sample.girth_m, ellipse_perimeter(sample.thickness_m, sample.thickness_m * DEPTH_TO_WIDTH), places=6)

    def test_from_the_side_the_thickness_in_the_picture_is_the_depth_of_the_body(self):
        mask = pig_mask(centre=(350, 400))
        (sample, why), floor = measured(mask, from_side(mask))
        self.assertEqual(why, "")
        self.assertTrue(np.allclose(floor.normal, (0, -1, 0), atol=.03), floor.normal)
        self.assertAlmostEqual(sample.length_m, 280 * 3 / FOCAL, delta=.04)
        self.assertGreater(sample.side_share, .97)
        depth = 90 * 3 / FOCAL
        self.assertAlmostEqual(sample.girth_m, ellipse_perimeter(depth / DEPTH_TO_WIDTH, depth), delta=.02)

    def test_the_same_animal_weighs_the_same_from_above_and_from_the_side(self):
        # The same body: 0.30 m wide seen from above, 0.36 m deep seen from the side, at the same distance.
        above = pig_mask()
        (top, _), _ = measured(above, from_above(above, back=2.3333))
        side_mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.rectangle(side_mask, (150, 346), (500, 454), 255, -1)
        cv2.rectangle(side_mask, (100, 370), (150, 430), 255, -1)
        side_mask = side_mask > 0
        (side, why), _ = measured(side_mask, from_side(side_mask, distance=2.3333))
        self.assertEqual(why, "")
        self.assertAlmostEqual(top.thickness_m, .30, delta=.01)
        self.assertAlmostEqual(side.thickness_m, .36, delta=.012)
        self.assertAlmostEqual(side.girth_m, top.girth_m, delta=top.girth_m * .05)

    def test_an_animal_standing_at_an_angle_is_measured_along_its_body_not_along_the_picture(self):
        mask = pig_mask()
        points = slanted(mask)
        shape = pig_shape(mask)
        sample, why = measure_pig(shape, points, mask, FOCAL)
        self.assertEqual(why, "")
        real = float(np.linalg.norm(place(shape.rump, points) - place(shape.neck, points)))
        self.assertAlmostEqual(sample.length_m, real, delta=real * .05)
        self.assertGreater(sample.length_m, shape.length_px * 3 / FOCAL * 1.1)

    def test_unmeasurable_animals_say_why(self):
        mask = pig_mask()
        shape = pig_shape(mask)
        self.assertEqual(measure_pig(pig_shape(bent_mask()), from_above(bent_mask()), bent_mask(), FOCAL), (None, BAD_POSE))
        self.assertEqual(measure_pig(shape, slanted(mask, middle=4., step=.012), mask, FOCAL), (None, END_ON))
        self.assertEqual(measure_pig(shape, np.full((SIZE, SIZE, 3), np.nan, np.float32), mask, FOCAL), (None, NO_DEPTH))
        wide = np.zeros((SIZE, SIZE), np.uint8)
        cv2.ellipse(wide, (350, 350), (110, 90), 0, 0, 360, 255, -1)
        self.assertEqual(measure_pig(pig_shape(wide > 0), from_above(wide > 0), wide > 0, FOCAL, Floor(np.array((0., 0., -1.)), 2.5)),
                         (None, BAD_POSE))

    def test_view_is_told_from_the_body_the_ray_and_the_floor(self):
        along = np.array((1., 0., 0.))
        down, level = np.array((0., 0., 1.)), np.array((0., 0., 1.))
        self.assertAlmostEqual(seen_from_side(along, down, np.array((0., 0., -1.))), 0.)
        self.assertAlmostEqual(seen_from_side(along, level, CAMERA_UP), 1.)
        tilted = np.array((0., -1., -1.)) / np.sqrt(2)
        self.assertAlmostEqual(seen_from_side(along, down, tilted), .5)
        self.assertEqual(seen_from_side(along, down, along), .5)

    def test_tape_formula_matches_the_farm_table(self):
        # Heart girth 1.12 m and length 1.10 m is a pig of about 96 kg.
        self.assertAlmostEqual(tape_mass(1.10, 1.12), 95.6, delta=.1)

    def test_estimate_is_the_middle_of_the_frames_and_one_odd_frame_does_not_move_it(self):
        accumulator = PigAccumulator()
        self.assertIsNone(accumulator.estimate())
        for _ in range(9):
            accumulator.add(PigSample(1., 1.1, .3, 0.))
        steady = accumulator.estimate()
        self.assertAlmostEqual(steady.mass_kg, tape_mass(1., 1.1))
        accumulator.add(PigSample(1.6, 1.5, .4, 0.))
        self.assertAlmostEqual(accumulator.estimate().mass_kg, steady.mass_kg, delta=steady.mass_kg * .01)
        self.assertEqual(accumulator.estimate().samples, 10)

    def test_scale_correction_applies_to_the_frame_it_was_found_in(self):
        accumulator = PigAccumulator()
        accumulator.add(PigSample(1., 1.1, .3, 0.), factor=2.)
        estimate = accumulator.estimate()
        self.assertAlmostEqual(estimate.length_m, 2.)
        self.assertAlmostEqual(estimate.mass_kg, tape_mass(1., 1.1) * 8)

    def test_range_is_wider_while_the_scale_is_only_estimated(self):
        accumulator = PigAccumulator()
        accumulator.add(PigSample(1., 1.1, .3, 0.))
        sure, rough = accumulator.estimate(True), accumulator.estimate(False)
        self.assertEqual(sure.mass_kg, rough.mass_kg)
        self.assertLess(rough.mass_low_kg, sure.mass_low_kg)
        self.assertGreater(rough.mass_high_kg, sure.mass_high_kg)


class DetectionHelperTests(unittest.TestCase):
    def area(self, top, bottom, left, right) -> np.ndarray:
        mask = np.zeros((100, 100), bool)
        mask[top:bottom, left:right] = True
        return mask

    def test_one_animal_found_under_two_names_is_kept_once(self):
        boxes = np.array([[0, 0, 100, 50], [2, 1, 101, 52], [200, 0, 300, 50]], float)
        self.assertEqual(keep_distinct(boxes, [.5, .9, .7]), [1, 2])

    def test_a_part_of_an_animal_is_not_counted_as_another_animal(self):
        whole, head = self.area(20, 80, 10, 90), self.area(30, 50, 15, 35)
        found = distinct_outlines([head, whole], [.9, .6])
        self.assertEqual([index for index, _ in found], [1])
        self.assertEqual(len(distinct_outlines([whole, whole.copy()], [.6, .9])), 1)

    def test_an_outline_over_two_animals_gives_way_to_the_two(self):
        both, left, right = self.area(20, 80, 10, 90), self.area(20, 80, 10, 48), self.area(20, 80, 52, 90)
        found = distinct_outlines([both, left, right], [.9, .7, .7])
        self.assertEqual([index for index, _ in found], [1, 2])

    def test_every_pixel_belongs_to_one_animal_and_empty_outlines_are_ignored(self):
        first, second = self.area(10, 50, 10, 60), self.area(10, 50, 40, 90)
        found = distinct_outlines([first, second, np.zeros((100, 100), bool)], [.9, .8, .99])
        self.assertEqual([index for index, _ in found], [0, 1])
        self.assertFalse((found[0][1] & found[1][1]).any())
        self.assertEqual(int(found[0][1].sum()), 2000)
        self.assertEqual(int(found[1][1].sum()), 1200)
        self.assertEqual(distinct_outlines([], []), [])

    def test_download_size_counts_only_what_is_missing(self):
        with TemporaryDirectory() as temp:
            everything = DETECTOR.weights_size + SEGMENTER.weights_size + DEPTH.weights_size
            self.assertEqual(missing_size_mb(temp), round(everything / 1e6))
            folder = model_path(temp, SEGMENTER)
            folder.mkdir(parents=True)
            with (folder / WEIGHTS_FILE).open("wb") as output:
                output.truncate(SEGMENTER.weights_size)
            self.assertEqual(missing_size_mb(temp), round((DETECTOR.weights_size + DEPTH.weights_size) / 1e6))


class TrackerTests(unittest.TestCase):
    def test_numbers_stay_with_animals_that_move_a_little_and_swap_order(self):
        tracker = PigTracker()
        first = tracker.update(np.array([[0, 0, 100, 50], [300, 0, 400, 50]], float), 0.)
        self.assertEqual([track.track_id for track in first], [1, 2])
        second = tracker.update(np.array([[310, 5, 410, 55], [8, 2, 108, 52]], float), .3)
        self.assertEqual([track.track_id for track in second], [2, 1])

    def test_a_newcomer_gets_a_new_number_and_a_long_gone_number_is_not_reused(self):
        tracker = PigTracker()
        tracker.update(np.array([[0, 0, 100, 50]], float), 0.)
        self.assertEqual(tracker.update(np.array([[0, 0, 100, 50], [500, 300, 600, 350]], float), .2)[1].track_id, 2)
        late = tracker.update(np.array([[0, 0, 100, 50]], float), MEMORY_S + 1)
        self.assertEqual(late[0].track_id, 3)

    def test_photograph_numbers_read_left_to_right_in_rows(self):
        boxes = np.array([[500, 10, 600, 60], [10, 400, 110, 450], [10, 20, 110, 70]], float)
        self.assertEqual(list(reading_order(boxes, 600)), [2, 0, 1])
        self.assertEqual(len(reading_order(np.zeros((0, 4)), 600)), 0)


FAKE_FLOOR, FAKE_BACK = 2.8, 2.4
FAKE_FOCAL = assumed_focal((SIZE, SIZE))


class FakeDetector:
    """White animals on a black floor seen from above, so the session can be tested without neural networks."""
    device_label = "проверка"

    def detect(self, image) -> tuple:
        count, _, stats, _ = cv2.connectedComponentsWithStats((image[:, :, 0] > 127).astype(np.uint8))
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h, area in stats[1:count] if area > 500], float).reshape(-1, 4)
        return boxes, np.ones(len(boxes))

    def outlines(self, image, boxes) -> list:
        white = image[:, :, 0] > 127
        masks = []
        for x0, y0, x1, y1 in boxes.astype(int):
            mask = np.zeros(white.shape, bool)
            mask[y0:y1, x0:x1] = white[y0:y1, x0:x1]
            masks.append(mask)
        return masks

    def depth(self, image) -> np.ndarray:
        return np.where(image[:, :, 0] > 127, FAKE_BACK, FAKE_FLOOR).astype(np.float32)


def textured(mask, shift=0) -> np.ndarray:
    """A pig with spots on its skin that move with it, so that its motion can be followed."""
    rng = np.random.default_rng(7)
    skin = cv2.GaussianBlur(rng.integers(150, 256, (SIZE, SIZE + 200), dtype=np.uint8), (5, 5), 0)
    skin = np.clip(skin.astype(int) * 2 - 200, 140, 255).astype(np.uint8)[:, 200 - shift:200 - shift + SIZE]
    image = np.zeros((SIZE, SIZE, 3), np.uint8)
    image[mask] = skin[mask][:, None]
    return image


def picture(*masks) -> np.ndarray:
    image = np.zeros((SIZE, SIZE, 3), np.uint8)
    for mask in masks:
        image[mask] = 255
    return image


def write_video(path, frames) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20., (SIZE, SIZE))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


# A segment of 250 pixels on the floor of the fake scene, and the length that doubles every size.
SEGMENT_PIXELS = ((100, 600), (350, 600))
SEEN_SEGMENT_M = 250 * FAKE_FLOOR / FAKE_FOCAL


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.session = PigSession(lambda status: FakeDetector())
        self.session.start()
        self.session.set_active(True)
        self.wait(lambda state: state.ready)

    def tearDown(self):
        self.session.stop()
        self.session.join(timeout=5)
        self.temp.cleanup()

    def wait(self, condition, timeout=10.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition(self.session.state):
                return self.session.state
            time.sleep(.01)
        self.fail(f"session state did not arrive: {self.session.state.message}")

    def photo(self, image, name="свиньи.png") -> Path:
        path = Path(self.temp.name) / name
        cv2.imencode(".png", image)[1].tofile(str(path))
        return path

    def test_photo_is_measured_at_once_from_the_estimated_distances(self):
        small = pig_mask(angle=90, centre=(560, 350), scale=.5)
        self.session.post("photo", self.photo(picture(pig_mask(centre=(300, 350), scale=.8), small)))
        state = self.wait(lambda state: len(state.pigs) == 2 and all(pig.estimate for pig in state.pigs))
        self.assertEqual([pig.track_id for pig in state.pigs], [1, 2])
        self.assertEqual(state.scale_source, PICTURE)
        big, little = (pig.estimate for pig in state.pigs)
        self.assertAlmostEqual(big.length_m, 280 * .8 * FAKE_BACK / FAKE_FOCAL, delta=.04)
        self.assertGreater(big.mass_kg, little.mass_kg * 3)
        self.assertIn("измерено: 2", state.message)

    def test_segment_of_known_length_corrects_the_scale_and_can_be_taken_back(self):
        self.session.post("photo", self.photo(picture(pig_mask(scale=.8))))
        rough = self.wait(lambda state: len(state.pigs) == 1 and state.pigs[0].estimate).pigs[0].estimate
        self.session.post("reference", (SEGMENT_PIXELS, SEEN_SEGMENT_M * 2))
        state = self.wait(lambda state: state.scale_source == SEGMENT)
        exact = state.pigs[0].estimate
        self.assertAlmostEqual(exact.length_m, rough.length_m * 2, delta=.01)
        self.assertAlmostEqual(exact.mass_kg, rough.mass_kg * 8, delta=rough.mass_kg * .1)
        self.assertEqual(exact.samples, 1)
        self.assertLess(exact.mass_high_kg / exact.mass_kg, rough.mass_high_kg / rough.mass_kg)
        self.session.post("reference", None)
        state = self.wait(lambda state: state.scale_source == PICTURE)
        self.assertAlmostEqual(state.pigs[0].estimate.length_m, rough.length_m, delta=.005)

    def test_segment_that_cannot_be_measured_is_reported_and_changes_nothing(self):
        self.session.post("photo", self.photo(picture(pig_mask(scale=.8))))
        rough = self.wait(lambda state: len(state.pigs) == 1 and state.pigs[0].estimate).pigs[0].estimate
        self.session.post("reference", (((-80, -80), (100, 100)), 1.))
        state = self.wait(lambda state: state.scale_source == SEGMENT_FAILED)
        self.assertAlmostEqual(state.pigs[0].estimate.length_m, rough.length_m, delta=.005)

    def test_animal_cut_by_the_frame_edge_or_bent_is_numbered_but_not_measured(self):
        self.session.post("photo", self.photo(picture(pig_mask(centre=(110, 120)), bent_mask())))
        state = self.wait(lambda state: len(state.pigs) == 2)
        self.assertEqual({pig.note for pig in state.pigs}, {CUT, BAD_POSE})
        self.assertTrue(all(pig.estimate is None for pig in state.pigs))

    def test_unreadable_photo_is_reported_and_the_session_goes_on(self):
        broken = Path(self.temp.name) / "broken.jpg"
        broken.write_bytes(b"not a picture")
        self.session.post("photo", broken)
        self.wait(lambda state: "Не удалось открыть фото" in state.message)
        self.session.post("photo", self.photo(picture(pig_mask())))
        self.wait(lambda state: len(state.pigs) == 1)

    def test_video_plays_to_the_end_keeps_numbers_and_gathers_frames(self):
        frames = [picture(pig_mask(centre=(250 + 4 * step, 350), scale=.8)) for step in range(30)]
        self.session.post("video", write_video(Path(self.temp.name) / "pigs.avi", frames))
        state = self.wait(lambda state: state.finished)
        self.assertEqual(state.kind, "video")
        self.assertFalse(state.playing)
        self.assertEqual([pig.track_id for pig in state.pigs], [1])
        self.assertGreater(state.pigs[0].estimate.samples, 2)
        self.session.post("play", True)
        self.wait(lambda state: state.playing and not state.finished)
        self.wait(lambda state: state.finished)

    def test_nothing_is_analysed_while_the_tab_is_closed(self):
        self.session.set_active(False)
        self.session.post("video", write_video(Path(self.temp.name) / "pigs.avi", [picture(pig_mask(scale=.8))] * 40))
        self.wait(lambda state: state.kind == "video")
        time.sleep(.5)
        self.assertIsNone(self.session.state.image)
        self.session.set_active(True)
        self.wait(lambda state: state.image is not None)

    def test_pause_stops_the_video_and_reset_gives_numbers_again(self):
        self.session.post("video", write_video(Path(self.temp.name) / "pigs.avi", [picture(pig_mask(scale=.8))] * 200))
        self.wait(lambda state: state.image is not None)
        self.session.post("play", False)
        paused = self.wait(lambda state: not state.playing)
        time.sleep(.4)
        self.assertLess(self.session.state.position_s - paused.position_s, .2)
        self.assertFalse(self.session.state.finished)
        self.session.post("photo", self.photo(picture(pig_mask())))
        self.wait(lambda state: state.kind == "photo" and len(state.pigs) == 1)
        before = self.session.state.serial
        self.session.post("reset")
        state = self.wait(lambda state: state.serial > before and len(state.pigs) == 1)
        self.assertEqual(state.pigs[0].track_id, 1)

    def test_camera_frames_are_taken_one_at_a_time_and_only_while_the_tab_is_open(self):
        self.assertFalse(self.session.wants_frame())
        self.session.post("camera")
        self.wait(lambda state: state.kind == "camera")
        self.assertTrue(self.session.wants_frame())
        self.session.set_active(False)
        self.assertFalse(self.session.wants_frame())
        self.session.set_active(True)
        image = picture(pig_mask(scale=.8))
        self.session.submit(image)
        state = self.wait(lambda state: len(state.pigs) == 1 and state.pigs[0].estimate)
        self.assertEqual(state.scale_source, PICTURE)
        self.wait(lambda state: self.session.wants_frame())
        self.session.post("photo", self.photo(image))
        self.wait(lambda state: state.kind == "photo")
        self.assertFalse(self.session.wants_frame())

    def test_calibrated_stereo_pair_sets_the_scale_without_any_segment(self):
        image = picture(pig_mask(scale=.8))
        self.session.post("camera")
        self.wait(lambda state: state.kind == "camera")
        self.session.submit(image)
        rough = self.wait(lambda state: len(state.pigs) == 1 and state.pigs[0].estimate).pigs[0].estimate
        self.wait(lambda state: self.session.wants_frame())
        self.session.post("reset")
        real = FakeDetector().depth(image) * 2
        self.wait(lambda state: not state.pigs)
        self.session.submit(image, real, np.ones(real.shape, bool), FAKE_FOCAL, (MIDDLE, MIDDLE))
        state = self.wait(lambda state: state.scale_source == STEREO and len(state.pigs) == 1 and state.pigs[0].estimate)
        self.assertAlmostEqual(state.pigs[0].estimate.length_m, rough.length_m * 2, delta=.02)

    def test_saved_result_holds_numbers_only(self):
        self.session.post("photo", self.photo(picture(pig_mask())))
        state = self.wait(lambda state: any(pig.estimate for pig in state.pigs))
        target = Path(self.temp.name) / "output" / "pigs_test" / "result.json"
        self.assertEqual(write_result(target, state.animals, dict(source="photo")), 1)
        saved = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(saved["source"], "photo")
        self.assertEqual(set(saved["pigs"][0]), {"label", "mass_kg", "mass_low_kg", "mass_high_kg", "length_m", "girth_m",
                                                  "samples", "volume_l"})
        self.assertGreater(saved["pigs"][0]["volume_l"], 0)
        self.assertEqual([path.name for path in target.parent.iterdir()], ["result.json"])


class FileTests(unittest.TestCase):
    def test_photo_with_a_cyrillic_name_opens_and_large_ones_are_reduced(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "свинья сверху.png"
            cv2.imencode(".png", np.zeros((300, 2600, 3), np.uint8))[1].tofile(str(path))
            image = read_image(path)
            self.assertEqual(image.shape[:2], (300, 2600))
            self.assertEqual(fitted(image).shape[:2], (148, 1280))
            small = image[:, :800]
            self.assertIs(fitted(small), small)
            self.assertIsNone(read_image(Path(temp) / "нет такого.png"))

    def test_latest_calibration_is_the_newest_of_the_current_one_and_those_kept_with_series(self):
        with TemporaryDirectory() as temp:
            self.assertIsNone(latest_calibration(temp))
            data = Path(temp) / "data"
            current, series = data / "calibration.npz", data / "calibration_20261103_160937" / "calibration.npz"
            unfinished = data / "calibration_20261104_101010"
            for path in (current, series):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            unfinished.mkdir()
            os.utime(current, (1000, 1000))
            os.utime(series, (2000, 2000))
            self.assertEqual(latest_calibration(temp), series)
            os.utime(current, (3000, 3000))
            self.assertEqual(latest_calibration(temp), current)


class CameraFeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.worker = CameraWorker(Path(self.temp.name), CameraConfig(), (9, 6), 25., 8., 64)
        self.left = np.full((4, 6, 3), 7, np.uint8)
        self.worker.raw = (self.left, np.zeros((4, 6, 3), np.uint8))
        self.got = []
        self.sink = SimpleNamespace(wants_frame=lambda: True, submit=lambda *frame: self.got.append(frame))

    def engine(self):
        points = np.dstack((np.zeros((4, 6)), np.zeros((4, 6)), np.full((4, 6), 2.)))
        points[0, 0, 2] = np.nan
        frame = SimpleNamespace(left=self.left + 1, points=points, valid=np.ones((4, 6), bool))
        projection = np.array(((500., 0, 3, 0), (0, 500., 2, 0), (0, 0, 1, 0)))
        self.computed = []
        return SimpleNamespace(calibration=SimpleNamespace(P1=projection), speed_priority=False,
                               compute=lambda *raw: self.computed.append(raw) or frame)

    def test_commands_pause_the_camera_work_and_name_the_tab_to_feed(self):
        self.worker.command("suspend", True)
        self.worker.command("frame_sink", self.sink)
        self.assertTrue(self.worker.suspended)
        self.assertIs(self.worker.frame_sink, self.sink)
        self.worker.command("suspend", False)
        self.assertFalse(self.worker.suspended)

    def test_without_calibration_the_tab_gets_the_plain_left_picture(self):
        self.worker.frame_sink = self.sink
        self.worker.feed_sink()
        self.assertEqual(len(self.got), 1)
        self.assertIs(self.got[0][0], self.left)
        self.assertEqual(len(self.got[0]), 1)

    def test_with_calibration_the_tab_gets_real_depth_and_the_camera_geometry(self):
        self.worker.engine = self.engine()
        self.worker.frame_sink = self.sink
        self.worker.feed_sink()
        image, z, valid, focal, centre = self.got[0]
        self.assertEqual(int(image[0, 0, 0]), 8)
        self.assertEqual(float(z[1, 1]), 2.)
        self.assertFalse(valid[0, 0])
        self.assertTrue(valid[1, 1])
        self.assertEqual((focal, centre), (500., (3., 2.)))

    def test_nothing_is_computed_when_the_tab_is_not_waiting_for_a_frame(self):
        self.worker.engine = self.engine()
        self.worker.feed_sink()
        self.worker.frame_sink = SimpleNamespace(wants_frame=lambda: False, submit=self.sink.submit)
        self.worker.feed_sink()
        self.assertEqual((self.got, self.computed), ([], []))


class TabTextTests(unittest.TestCase):
    def test_numbers_are_read_with_a_comma_or_a_point(self):
        self.assertEqual(parse_number(" 2,5 "), 2.5)
        self.assertEqual(parse_number("120"), 120.)
        for text in ("", "abc", "0", "-3"):
            self.assertIsNone(parse_number(text))

    def test_table_line_gives_the_weight_or_says_why_there_is_none(self):
        accumulator = PigAccumulator()
        accumulator.add(PigSample(1.12, 1.2, .36, 0.))
        estimate = accumulator.estimate()
        row = pig_row(SimpleNamespace(track_id=3, estimate=estimate, note=""))
        self.assertEqual(row[:2], (3, f"{estimate.mass_kg:.0f}"))
        self.assertEqual(row[3:], ("—", "112", "120", 1, ""))
        self.assertIn(BAD_POSE, pig_row(SimpleNamespace(track_id=3, estimate=estimate, note=BAD_POSE))[7])
        self.assertEqual(pig_row(SimpleNamespace(track_id=4, estimate=None, note=CUT)), (4, "—", "—", "—", "—", "—", 0, CUT))
        with_volume = PigAccumulator()
        with_volume.add(PigSample(1.12, 1.2, .36, 0., .095))
        self.assertEqual(pig_row(SimpleNamespace(track_id=3, estimate=with_volume.estimate(), note=""))[3], "95")

    def test_drawing_marks_animals_and_the_segment_without_touching_the_frame(self):
        mask = pig_mask()
        image = picture(mask)
        pig = SimpleNamespace(track_id=1, box=(95., 290., 505., 410.), color=(80, 200, 255), contours=outline_contours(mask),
                              shape=pig_shape(mask), estimate=None, note="")
        state = SimpleNamespace(image=image, pigs=(pig,))
        before = image.copy()
        canvas = draw_pigs(state, ((20, 20), (120, 20)))
        self.assertTrue(np.array_equal(image, before))
        self.assertFalse(np.array_equal(canvas, image))
        self.assertEqual(tuple(canvas[20, 70]), (60, 220, 255))


class TabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = tk.Tk()
        cls.host.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.host.destroy()

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.errors = []
        self.previous = self.host.report_callback_exception
        self.host.report_callback_exception = lambda *error: self.errors.append(error)
        self.window = tk.Toplevel(self.host)
        self.app = StereoApp(self.window, Path(self.temp.name))
        self.window.geometry("1100x700+10+10")
        self.host.update()
        self.tab = self.app.pig_tab

    def tearDown(self):
        self.tab.close()
        self.app.worker = None
        self.window.destroy()
        self.host.update_idletasks()
        self.host.report_callback_exception = self.previous
        self.temp.cleanup()
        self.assertFalse(self.errors, self.errors)

    def pump(self, condition, timeout=10.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.host.update()
            self.tab.poll()
            if condition():
                return
            time.sleep(.01)
        self.fail(f"the tab did not get there: {self.tab.message.get()}")

    def connect(self) -> Queue:
        worker = SimpleNamespace(engine=None, frame=None, scan=None, commands=Queue(), events=Queue(), frames=Queue(),
                                 previews=Queue(), stop_event=Event(), is_alive=lambda: True, max_depth=1.2,
                                 near_minimum=.1, ghost_filter="normal", raw=None)
        self.app.worker, self.app.connected = worker, True
        self.app.sync_ui(force=True)
        return worker.commands

    def sent(self, commands) -> list:
        found = []
        while not commands.empty():
            found.append(commands.get_nowait())
        return found

    def open_with_models(self):
        patches = (patch("stereo.pig_ui.models_ready", return_value=True),
                   patch("stereo.pig_detection.PigDetector", lambda root, status: FakeDetector()))
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.app.show_page("pigs")
        self.pump(lambda: self.tab.session is not None and self.tab.session.state.ready)

    def show_photo(self):
        path = Path(self.temp.name) / "фото.png"
        cv2.imencode(".png", picture(pig_mask(scale=.8)))[1].tofile(str(path))
        self.tab.session.post("photo", path)
        self.pump(lambda: self.tab.state is not None and self.tab.state.kind == "photo" and len(self.tab.state.pigs) == 1
                  and self.tab.state.pigs[0].estimate)

    def row(self) -> tuple:
        return self.tab.table.item(self.tab.table.get_children()[0], "values")

    def test_tab_comes_after_weighing_and_asks_for_models_before_anything_else(self):
        self.assertEqual(list(self.app.workflow_pages), ["cameras", "calibration", "scan", "weighing", "pigs", "advanced"])
        self.app.show_page("pigs")
        self.host.update()
        self.tab.poll()
        self.assertIsNone(self.tab.session)
        self.assertTrue(self.tab.download_button.winfo_ismapped())
        self.assertIn("МБ", self.tab.download_button.cget("text"))
        for press in (self.tab.reset, self.tab.toggle_play, self.tab.rewind, self.tab.open_camera):
            press()
        self.assertIn("скачайте модели", self.tab.message.get())
        self.tab.start_marking()
        self.assertIn("Сначала откройте", self.tab.message.get())

    def test_camera_work_is_paused_only_while_the_tab_is_open(self):
        commands = self.connect()
        self.app.show_page("pigs")
        self.host.update()
        self.tab.poll()
        self.assertIn(("suspend", True), self.sent(commands))
        self.tab.poll()
        self.assertEqual(self.sent(commands), [])
        self.app.show_page("scan")
        self.host.update()
        self.tab.poll()
        self.assertIn(("suspend", False), self.sent(commands))

    def test_cameras_connected_while_the_tab_is_open_are_paused_too(self):
        self.app.show_page("pigs")
        self.host.update()
        self.tab.poll()
        commands = self.connect()
        self.tab.poll()
        self.assertIn(("suspend", True), self.sent(commands))

    def test_photo_is_weighed_at_once_and_a_marked_segment_makes_it_exact(self):
        self.open_with_models()
        self.show_photo()
        self.assertEqual(self.tab.metric_count.get(), "1")
        self.assertNotEqual(self.tab.metric_mean.get(), "—")
        self.assertIn("приблизительный", self.tab.scale_text.get())
        self.assertFalse(self.tab.controls.winfo_ismapped())
        rough = self.tab.state.pigs[0].estimate.mass_kg
        self.assertEqual(self.row()[1], kg_text(rough))
        self.tab.start_marking()
        self.tab.clicked(*SEGMENT_PIXELS[0])
        self.tab.clicked(*SEGMENT_PIXELS[1])
        self.assertIn("Введите его длину", self.tab.message.get())
        self.tab.length.set(f"{SEEN_SEGMENT_M * 200:.2f}".replace(".", ","))
        self.pump(lambda: "уточнён" in self.tab.scale_text.get())
        self.assertAlmostEqual(self.tab.state.pigs[0].estimate.mass_kg, rough * 8, delta=rough * .2)
        self.tab.length.set("")
        self.pump(lambda: "приблизительный" in self.tab.scale_text.get())
        self.tab.save()
        self.assertIn("Сохранено свиней: 1", self.tab.message.get())
        saved = list((Path(self.temp.name) / "output").glob("pigs_*/result.json"))
        self.assertEqual(json.loads(saved[0].read_text(encoding="utf-8"))["scale"], PICTURE)

    def test_video_buttons_appear_only_for_a_video_and_leaving_the_tab_stops_it(self):
        self.open_with_models()
        path = write_video(Path(self.temp.name) / "pigs.avi", [picture(pig_mask(scale=.8))] * 400)
        self.tab.session.post("video", path)
        self.pump(lambda: self.tab.state is not None and self.tab.state.image is not None)
        self.assertTrue(self.tab.controls.winfo_ismapped())
        self.assertEqual(self.tab.play_button.cget("text"), "Пауза")
        self.app.show_page("cameras")
        self.pump(lambda: not self.tab.active)
        time.sleep(.3)
        stopped = self.tab.session.state.position_s
        time.sleep(.5)
        self.assertEqual(self.tab.session.state.position_s, stopped)
        self.app.show_page("pigs")
        self.pump(lambda: self.tab.session.state.position_s > stopped)
        self.assertLess(self.tab.session.state.position_s, stopped + 1.5)
        self.show_photo()
        self.assertFalse(self.tab.controls.winfo_ismapped())

    def test_camera_needs_connected_cameras_and_then_feeds_the_tab(self):
        self.open_with_models()
        self.tab.open_camera()
        self.assertEqual(self.tab.message.get(), NO_CAMERA)
        self.assertFalse(self.tab.camera_on())
        commands = self.connect()
        self.tab.open_camera()
        self.assertIn(("frame_sink", self.tab.session), self.sent(commands))
        self.pump(self.tab.camera_on)
        self.tab.session.submit(picture(pig_mask(scale=.8)))
        self.pump(lambda: self.tab.state is not None and len(self.tab.state.pigs) == 1)
        self.assertEqual(self.tab.metric_count.get(), "1")
        # The cameras were reconnected: another camera thread must learn about the tab.
        commands = self.connect()
        self.tab.poll()
        self.assertIn(("frame_sink", self.tab.session), self.sent(commands))
        self.app.connected = False
        self.tab.session.post("reset")
        self.pump(lambda: "Камеры отключены" in self.tab.message.get())

    def test_new_picture_forgets_the_old_segment(self):
        self.open_with_models()
        self.show_photo()
        self.tab.segment = SEGMENT_PIXELS
        self.tab.length.set("50")
        self.assertIsNotNone(self.tab.reference())
        self.pump(lambda: "уточнён" in self.tab.scale_text.get())
        self.tab.forget_segment()
        self.assertIsNone(self.tab.reference())
        self.pump(lambda: "приблизительный" in self.tab.scale_text.get())

    def test_latest_calibration_loads_without_choosing_a_file(self):
        commands = self.connect()
        self.sent(commands)
        self.app.load_latest_calibration()
        self.assertEqual(self.sent(commands), [])
        self.assertIn("пока нет", self.app.status.get())
        saved = Path(self.temp.name) / "data" / "calibration.npz"
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(b"x")
        self.app.load_latest_calibration()
        self.assertEqual(self.sent(commands), [("load", str(saved))])
        self.assertIn("последнюю калибровку от", self.app.status.get())



def estimate_of(mass) -> PigEstimate:
    return PigEstimate(.5, .5, mass, mass * .9, mass * 1.1, 4)


def read_all(path) -> list:
    video = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(frame)
    video.release()
    return frames


class CalibrationAndVideoTests(SessionTests):
    """Scale from one animal on scales, every frame of a video analysed, and the video written marked."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.output = Path(self.temp.name) / "output"
        self.session = PigSession(lambda status: FakeDetector(), self.output)
        self.session.start()
        self.session.set_active(True)
        self.wait(lambda state: state.ready)

    def test_one_animal_weighed_on_scales_sets_the_scale_of_all(self):
        small = pig_mask(angle=90, centre=(560, 350), scale=.5)
        self.session.post("photo", self.photo(picture(pig_mask(centre=(300, 350), scale=.8), small)))
        state = self.wait(lambda state: len(state.animals) == 2 and all(pig.estimate for pig in state.animals))
        big, little = (pig.estimate for pig in state.animals)
        self.session.post("known", (1, big.mass_kg * 8))
        state = self.wait(lambda state: state.scale_source == WEIGHT)
        self.assertEqual(state.weighed_pig, 1)
        weighed, other = state.animals
        self.assertAlmostEqual(weighed.estimate.mass_kg, big.mass_kg * 8, delta=big.mass_kg * .01)
        self.assertAlmostEqual(weighed.estimate.length_m, big.length_m * 2, delta=.01)
        self.assertAlmostEqual(other.estimate.mass_kg, little.mass_kg * 8, delta=little.mass_kg * .01)
        self.assertEqual(weighed.note, WEIGHED)
        self.assertEqual(state.pigs[0].estimate, weighed.estimate)
        self.assertAlmostEqual(state.camera_height_m, FAKE_FLOOR * 2, delta=.1)
        self.session.post("known", (7, 30.))
        state = self.wait(lambda state: state.scale_source == WEIGHT_FAILED)
        self.assertAlmostEqual(state.animals[0].estimate.mass_kg, big.mass_kg, delta=big.mass_kg * .01)
        self.session.post("known", None)
        self.wait(lambda state: state.scale_source == PICTURE)

    def test_every_frame_of_a_video_is_shown_and_written_again_marked_where_the_pig_is(self):
        side = 1400
        frames = [cv2.resize(textured(pig_mask(centre=(250 + 4 * step, 350), scale=.8), 4 * step), (side, side),
                             interpolation=cv2.INTER_NEAREST) for step in range(30)]
        path = write_video(Path(self.temp.name) / "свиньи.avi", [])
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20., (side, side))
        for frame in frames:
            writer.write(frame)
        writer.release()
        self.session.post("video", path)
        state = self.wait(lambda state: state.finished and state.video_file and state.rendering is None, timeout=30)
        self.assertEqual((state.frame, state.frames), (30, 30))
        # The networks look at the newest frame whenever they are free, not at every one.
        self.assertTrue(2 <= state.animals[0].estimate.samples <= 30, state.animals[0].estimate.samples)
        written = Path(state.video_file)
        self.assertEqual(written.name, "свиньи" + MARKED_SUFFIX + ".mp4")
        self.assertEqual(written.parent.parent, self.output)
        marked = read_all(written)
        self.assertEqual(len(marked), 30)
        self.assertEqual(marked[0].shape, (side, side, 3))
        # The first frame: the body is tinted, the empty corner is left as it was.
        corner = (slice(1300, 1380), slice(1300, 1380))
        self.assertLess(np.abs(marked[0][corner].astype(int) - frames[0][corner]).mean(), 8)
        for step, (shown, original) in enumerate(zip(marked, frames)):
            # Every frame, analysed or carried, is tinted where the body is in that frame and not where it was.
            body = (slice(690, 710), slice(500 + 8 * step - 10, 500 + 8 * step + 10))
            self.assertGreater(np.abs(shown[body].astype(int) - original[body]).mean(), 15, step)
        left_behind = (slice(690, 710), slice(290, 310))
        self.assertGreater(np.abs(marked[0][left_behind].astype(int) - frames[0][left_behind]).mean(), 15)
        self.assertLess(np.abs(marked[29][left_behind].astype(int) - frames[29][left_behind]).mean(), 8)
        saved = json.loads((written.parent / "result.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["source"], saved["frames"], len(saved["pigs"])), ("video", 30, 1))
        self.assertAlmostEqual(saved["pigs"][0]["mass_kg"], state.animals[0].estimate.mass_kg, delta=.1)

    def test_a_segment_marked_after_the_video_rescales_every_frame_already_seen(self):
        frames = [picture(pig_mask(centre=(250 + 4 * step, 350), scale=.8)) for step in range(20)]
        self.session.post("video", write_video(Path(self.temp.name) / "pigs.avi", frames))
        rough = self.wait(lambda state: state.finished and state.rendering is None).animals[0].estimate
        self.session.post("reference", (SEGMENT_PIXELS, SEEN_SEGMENT_M * 2))
        state = self.wait(lambda state: state.scale_source == SEGMENT)
        exact = state.animals[0].estimate
        self.assertEqual(exact.samples, rough.samples)
        self.assertAlmostEqual(exact.mass_kg, rough.mass_kg * 8, delta=rough.mass_kg * .1)
        first = state.video_file
        self.session.post("render")
        self.wait(lambda state: state.rendering is not None or state.video_file != first)
        state = self.wait(lambda state: state.rendering is None and state.video_file)
        saved = json.loads((Path(state.video_file).parent / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["scale"], SEGMENT)
        self.assertAlmostEqual(saved["pigs"][0]["mass_kg"], exact.mass_kg, delta=.1)

    def test_rewinding_starts_the_analysis_and_the_numbers_again(self):
        frames = [picture(pig_mask(centre=(250 + 4 * step, 350), scale=.8)) for step in range(15)]
        self.session.post("video", write_video(Path(self.temp.name) / "pigs.avi", frames))
        self.wait(lambda state: state.finished and state.video_file)
        self.session.post("rewind")
        state = self.wait(lambda state: state.finished and state.video_file)
        self.assertEqual([pig.track_id for pig in state.animals], [1])
        self.assertEqual(state.animals[0].estimate.samples, 15)


class FakeModel:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class PlacementTests(unittest.TestCase):
    """Which model goes to the graphics card, without loading any real one."""

    def detector(self, device, failing=()):
        detector = PigDetector.__new__(PigDetector)
        detector.device, detector.device_name = device, "Видеокарта"
        detector.devices = dict(detector="cpu", outline="cpu", depth="cpu")
        detector.detector, detector.segmenter, detector.depth_model = FakeModel(), FakeModel(), FakeModel()

        def trial(name):
            def run(*_):
                if detector.devices[name] != "cpu" and name in failing:
                    raise RuntimeError("unsupported operator")
            return run
        detector.detect, detector.outlines, detector.depth = trial("detector"), trial("outline"), trial("depth")
        return detector

    def test_on_cuda_every_working_model_goes_to_the_card(self):
        detector = self.detector("cuda")
        detector.place_models()
        self.assertEqual(detector.devices, dict(detector="cuda", outline="cuda", depth="cuda"))
        self.assertEqual(detector.device_label, "Видеокарта")

    def test_a_model_that_fails_on_the_card_returns_to_the_processor(self):
        detector = self.detector("cuda", failing=("outline",))
        detector.place_models()
        self.assertEqual(detector.devices["outline"], "cpu")
        self.assertEqual(detector.segmenter.devices[-1], "cpu")
        self.assertEqual(detector.device_label, "Видеокарта и процессор")

    def test_on_directml_the_detector_is_never_tried_and_a_slow_card_is_not_used(self):
        detector = self.detector("dml")
        with patch("stereo.pig_detection.timed", side_effect=[1., 2., 1., .5]):
            detector.place_models()
        self.assertEqual(detector.devices, dict(detector="cpu", outline="cpu", depth="dml"))
        self.assertEqual(detector.detector.devices, [])
        detector.devices = dict(detector="cpu", outline="cpu", depth="cpu")
        self.assertEqual(detector.device_label, "процессор (Видеокарта не подошла)")


class DrawingTests(unittest.TestCase):
    def test_small_weights_keep_their_tenths(self):
        self.assertEqual((kg_text(1.44), kg_text(6.66), kg_text(19.94), kg_text(20.4), kg_text(105.6)),
                         ("1.4", "6.7", "19.9", "20", "106"))
        self.assertEqual(label_text(3, estimate_of(5.43)), "#3 5.4 kg")
        self.assertEqual(label_text(4, None), "#4")
        row = pig_row(SimpleNamespace(track_id=2, estimate=estimate_of(6.), note="", in_view=True))
        self.assertEqual(row[1:3], ("6.0", "5.4–6.6"))
        self.assertEqual(pig_row(SimpleNamespace(track_id=2, estimate=estimate_of(6.), note=WEIGHED, in_view=False))[7],
                         WEIGHED)
        self.assertEqual(pig_row(SimpleNamespace(track_id=2, estimate=estimate_of(6.), note=CUT, in_view=False))[7], "")

    def test_girth_is_marked_across_the_body_as_wide_as_the_body(self):
        shape = pig_shape(pig_mask())
        ends = girth_line(shape)
        self.assertAlmostEqual(float(np.linalg.norm(ends[1] - ends[0])), shape.girth_px, delta=1)
        self.assertAlmostEqual(float(ends[0][0]), float(ends[1][0]), delta=3)

    def test_marks_of_a_reduced_frame_land_on_the_full_size_frame(self):
        mask = pig_mask()
        mark = SimpleNamespace(track_id=1, box=(95., 290., 505., 410.), color=(0, 0, 255),
                               contours=outline_contours(mask), shape=pig_shape(mask))
        large = draw_marks(np.zeros((SIZE * 2, SIZE * 2, 3), np.uint8), [mark], ["#1 6.0 kg"], 2.)
        self.assertGreater(int(large[700, 500, 2]), 50)
        self.assertEqual(int(large[1300, 1300].sum()), 0)
        self.assertEqual(int(large[350, 250].sum()), 0)


class StatusTextTests(unittest.TestCase):
    def test_where_the_networks_run_and_what_to_do_about_an_idle_card(self):
        self.assertEqual(device_text("NVIDIA GeForce RTX 3060", None), "Считает видеокарта: NVIDIA GeForce RTX 3060.")
        self.assertIn("«Дополнительно»", device_text("процессор", None))
        self.assertIn("не подошла", device_text("процессор", "nvidia"))
        self.assertEqual(device_text("", None), "")

    def test_video_progress_tells_the_speed_and_the_time_left(self):
        state = PigState(kind="video", playing=True, frame=100, frames=400, fps=5., video_fps=25., analysis_fps=1.5)
        self.assertEqual(progress_text(state), " Показ 5 из 25 кадров/с, нейросети 1.5 кадра/с, осталось около 1:00.")
        self.assertIn("45 %", progress_text(PigState(kind="video", finished=True, rendering=.45)))
        self.assertEqual(progress_text(PigState(kind="video", finished=True, fps=5.)), "")

    def test_scale_text_names_the_guessed_camera_height_and_the_weighed_animal(self):
        guessed = scale_text(PigState(scale_source=PICTURE, camera_height_m=.216))
        self.assertIn("22 см над полом", guessed)
        self.assertIn("в разы", guessed)
        self.assertNotIn("{", scale_text(PigState(scale_source=PICTURE)))
        self.assertIn("№2", scale_text(PigState(scale_source=WEIGHT, weighed_pig=2)))


class TabCalibrationTests(TabTests):
    test_tab_comes_after_weighing_and_asks_for_models_before_anything_else = None
    test_camera_work_is_paused_only_while_the_tab_is_open = None
    test_cameras_connected_while_the_tab_is_open_are_paused_too = None
    test_photo_is_weighed_at_once_and_a_marked_segment_makes_it_exact = None
    test_video_buttons_appear_only_for_a_video_and_leaving_the_tab_stops_it = None
    test_camera_needs_connected_cameras_and_then_feeds_the_tab = None
    test_new_picture_forgets_the_old_segment = None
    test_latest_calibration_loads_without_choosing_a_file = None

    def test_typed_weight_of_one_animal_rescales_the_table(self):
        self.open_with_models()
        self.show_photo()
        rough = self.tab.state.animals[0].estimate.mass_kg
        self.tab.known_pig.set("1")
        self.assertIsNone(self.tab.known())
        self.tab.known_mass.set(f"{rough * 8:.3f}".replace(".", ","))
        self.assertEqual(self.tab.known(), (1, round(rough * 8, 3)))
        self.pump(lambda: "по весу свиньи №1" in self.tab.scale_text.get())
        self.assertEqual(self.row()[1], kg_text(rough * 8))
        self.assertEqual(self.row()[7], WEIGHED)
        self.tab.known_pig.set("1,5")
        self.assertIsNone(self.tab.known())
        self.pump(lambda: "приблизительный" in self.tab.scale_text.get())

    def test_finished_video_offers_to_write_it_again_and_to_open_it(self):
        self.open_with_models()
        frames = [picture(pig_mask(centre=(250 + 4 * step, 350), scale=.8)) for step in range(12)]
        self.tab.session.post("video", write_video(Path(self.temp.name) / "pigs.avi", frames))
        self.pump(lambda: self.tab.state is not None and self.tab.state.video_file and self.tab.state.rendering is None)
        self.assertTrue(self.tab.video_actions.winfo_ismapped())
        self.assertEqual(str(self.tab.render_button.cget("state")), "normal")
        self.assertEqual(str(self.tab.open_video_button.cget("state")), "normal")
        self.assertEqual(str(self.tab.play_button.cget("state")), "disabled")
        self.assertIn("Видео с разметкой:", self.tab.message.get())
        self.assertIn("кадров: 12", self.tab.message.get())
        self.assertIn(str(Path("output") / "pigs_"), self.tab.message.get())
        self.assertTrue(Path(self.tab.state.video_file).is_relative_to(Path(self.temp.name) / "output"))
        self.assertIn("проверка", self.tab.device.get())
        self.show_photo()
        self.assertFalse(self.tab.video_actions.winfo_ismapped())



class VolumeTests(unittest.TestCase):
    def test_a_box_shaped_body_seen_from_above_has_the_volume_of_its_elliptic_slices(self):
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.rectangle(mask, (200, 305), (499, 394), 255, -1)
        mask = mask > 0
        (sample, why), _ = measured(mask, from_above(mask))
        self.assertEqual(why, "")
        width, length = 90 * 2.05 / FOCAL, 300 * 2.05 / FOCAL
        expected = np.pi / 4 * width * width * DEPTH_TO_WIDTH * length
        self.assertAlmostEqual(sample.volume_m3, expected, delta=expected * .06)
        self.assertAlmostEqual(sample.mass_kg, sample.volume_m3 * BODY_DENSITY_KG_M3 * LEGS_FACTOR)

    def test_a_narrow_head_weighs_less_than_a_body_as_wide_as_the_trunk(self):
        pig = pig_mask()
        (shaped, _), _ = measured(pig, from_above(pig))
        block = np.zeros((SIZE, SIZE), np.uint8)
        cv2.rectangle(block, (95, 305), (505, 394), 255, -1)
        (boxy, _), _ = measured(block > 0, from_above(block > 0))
        self.assertLess(shaped.volume_m3, boxy.volume_m3 * .95)

    def test_volume_and_tape_agree_on_an_ordinary_pig(self):
        pig = pig_mask()
        (sample, _), _ = measured(pig, from_above(pig))
        tape = tape_mass(sample.length_m, sample.girth_m)
        self.assertAlmostEqual(sample.mass_kg, tape, delta=tape * .35)

    def test_estimate_keeps_the_volume_and_scales_it_as_a_cube(self):
        accumulator = PigAccumulator()
        for volume in (.09, .1, .11):
            accumulator.add(PigSample(1., 1.1, .3, 0., volume))
        estimate = accumulator.estimate()
        self.assertAlmostEqual(estimate.volume_l, 100.)
        self.assertAlmostEqual(estimate.mass_kg, .1 * BODY_DENSITY_KG_M3 * LEGS_FACTOR)
        self.assertAlmostEqual(estimate.scaled(2.).volume_l, 800.)


class TrackingTests(unittest.TestCase):
    def test_an_outline_follows_a_body_moving_across_the_frame(self):
        frames = [textured(pig_mask(centre=(250 + 6 * step, 350)), 6 * step) for step in range(10)]
        mask = pig_mask(centre=(250, 350))
        shape = pig_shape(mask)
        box = (-5., 290., 405., 410.)
        steps = carried([(outline_contours(mask), shape, box)],
                        [tracking_gray(frame)[0] for frame in frames], tracking_gray(frames[0])[1])
        self.assertEqual(len(steps), 10)
        last = steps[-1][0]
        self.assertGreater(len(last.points), 10)
        for before, after in ((shape.neck, last.now_shape.neck), (shape.rump, last.now_shape.rump)):
            self.assertAlmostEqual(after[0], before[0] + 54, delta=4)
            self.assertAlmostEqual(after[1], before[1], delta=4)
        self.assertAlmostEqual(last.now_shape.girth_px, shape.girth_px, delta=shape.girth_px * .05)

    def test_motions_compose_and_a_grown_body_has_wider_slices(self):
        shift = np.array(((1., 0., 10.), (0., 1., 0.)))
        grow = np.array(((2., 0., 0.), (0., 2., 0.)))
        self.assertTrue(np.allclose(compose(grow, shift), ((2., 0., 20.), (0., 2., 0.))))
        shape = pig_shape(pig_mask())
        bigger = moved_shape(shape, grow)
        self.assertAlmostEqual(bigger.girth_px, shape.girth_px * 2)
        self.assertAlmostEqual(bigger.slices[5], shape.slices[5] * 2)
        self.assertIsNone(moved_shape(None, IDENTITY))

    def test_a_flat_outline_without_anything_to_follow_stays_where_it_was(self):
        frame = picture(pig_mask())
        gray, scale = tracking_gray(frame)
        item = follower(outline_contours(pig_mask()), None, (95., 290., 505., 410.), gray, scale)
        moved = follow(gray, gray, [item], scale)[0]
        self.assertTrue(np.allclose(moved.matrix, IDENTITY))


class SlowDetector(FakeDetector):
    """Neural networks that take their time, as they do on a processor."""

    def __init__(self, seconds):
        self.seconds = seconds

    def depth(self, image):
        time.sleep(self.seconds)
        return super().depth(image)


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()

    def tearDown(self):
        self.session.stop()
        self.session.join(timeout=5)
        self.temp.cleanup()

    def start(self, seconds):
        self.session = PigSession(lambda status: SlowDetector(seconds), Path(self.temp.name) / "output")
        self.session.start()
        self.session.set_active(True)
        end = time.monotonic() + 10
        while not self.session.state.ready and time.monotonic() < end:
            time.sleep(.01)

    def play(self, frames, fps):
        path = Path(self.temp.name) / "pigs.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (SIZE, SIZE))
        for step in range(frames):
            writer.write(textured(pig_mask(centre=(250 + 3 * step, 350), scale=.8), 3 * step))
        writer.release()
        started = time.monotonic()
        self.session.post("video", path)
        end = started + 30
        shown = set()
        while time.monotonic() < end:
            state = self.session.state
            shown.add(state.frame)
            if state.finished and state.rendering is None and state.video_file:
                return state, time.monotonic() - started, shown
            time.sleep(.005)
        self.fail(f"the video did not end: {self.session.state.message}")

    def test_slow_networks_do_not_hold_the_video_back_and_every_frame_gets_its_outline(self):
        self.start(.25)
        state, elapsed, shown = self.play(40, 20.)
        self.assertLess(elapsed, 40 * .25)
        self.assertLess(state.animals[0].estimate.samples, 20)
        self.assertEqual(sorted(self.session.marks), list(range(40)))
        self.assertTrue(all(self.session.marks[index] for index in range(40)))
        self.assertGreater(len(shown), 25)

    def test_slow_networks_slow_the_video_evenly_instead_of_in_jumps(self):
        session = PigSession(lambda status: FakeDetector())
        session.networks = SimpleNamespace(count=2)
        session.fps = 25.
        self.session = SimpleNamespace(stop=lambda: None, join=lambda timeout: None)
        self.assertAlmostEqual(session.interval(), 1 / 25)
        session.merge_gap = 4.
        self.assertAlmostEqual(session.interval(), 1 / (25 * .25 * .9))
        session.merge_gap = .01
        self.assertAlmostEqual(session.interval(), 1 / 25)

    def test_the_video_waits_rather_than_carry_outlines_too_far(self):
        self.start(.15)
        with patch("stereo.pig_weighing.MAX_LAG_S", .02):
            state, elapsed, _ = self.play(30, 100.)
        # At its own rate the video lasts 0.3 s; two frames per pass of the networks take longer.
        self.assertGreater(elapsed, 1.5)
        self.assertEqual(sorted(self.session.marks), list(range(30)))


def raw_row(track_id, mass, source=PICTURE) -> PigRow:
    return PigRow(track_id, estimate_of(mass * 2), "", True, estimate_of(mass), source)


class PigDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.database = PigDatabase(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_records_keep_the_estimate_before_correction_and_come_back_newest_first(self):
        self.assertEqual(self.database.entries(), [])
        first = self.database.add(make_entry(raw_row(2, 1.), "8", "№2", "video", "test"))
        second = self.database.add(make_entry(raw_row(3, 1.2), "9,5", "№3", "photo", "test"))
        entries = self.database.entries()
        self.assertEqual([entry.id for entry in entries], [second, first])
        self.assertEqual((entries[1].real_weight_kg, entries[1].estimated_weight_kg, entries[1].scale), (8., 1., PICTURE))
        self.assertAlmostEqual(entries[1].length_factor, 2.)
        self.database.delete(second)
        self.assertEqual([entry.id for entry in self.database.entries()], [first])
        self.assertTrue((Path(self.temp.name) / "data" / "pigs" / "pigs.sqlite").is_file())

    def test_typing_slips_and_unmeasured_pigs_are_refused(self):
        for text in ("", "abc", "0", "1000"):
            with self.assertRaises(ValueError):
                make_entry(raw_row(2, 1.), text, "", "video", "test")
        with self.assertRaises(ValueError):
            make_entry(PigRow(2, None, "", True), "8", "", "video", "test")
        with self.assertRaises(ValueError):
            make_entry(None, "8", "", "video", "test")

    def test_correction_is_the_middle_of_the_records_of_the_same_kind(self):
        entries = [make_entry(raw_row(index, 1.), str(mass), "", "video", "test")
                   for index, mass in enumerate((8., 8. * 1.1 ** 3, 8. / 1.1 ** 3))]
        # A slip: the pig on the scales was not the one numbered, so the ratio is beyond belief.
        entries.append(make_entry(raw_row(5, .001), "300", "", "video", "test"))
        entries.append(make_entry(raw_row(9, 1., STEREO), "1", "", "camera", "test"))
        found = correction(entries, PICTURE)
        self.assertAlmostEqual(found.factor, 2., places=6)
        self.assertEqual(found.count, 3)
        self.assertAlmostEqual(found.spread, (2.2 - 2 / 1.1) / 4, places=6)
        self.assertAlmostEqual(correction(entries, STEREO).factor, 1.)
        self.assertIsNone(correction([], PICTURE))
        text = summary_text(entries[:1], PICTURE)
        self.assertIn("×2,00", text)
        self.assertIn("хотя бы 3", text)
        self.assertIn("Добавить в базу", summary_text([], PICTURE))

    def test_spreadsheet_gets_one_row_per_pig(self):
        self.database.add(make_entry(raw_row(2, 1.), "8", "=№2", "video", "test"))
        target = Path(self.temp.name) / "pigs.csv"
        self.assertEqual(export_csv(self.database.entries(), target), 1)
        lines = target.read_text(encoding="utf-8-sig").splitlines()
        self.assertTrue(lines[0].startswith("id;created;label;real_weight_kg"))
        self.assertIn(";'=№2;8.0;picture;", lines[1])


class LearnedScaleTests(CalibrationAndVideoTests):
    """Pigs weighed earlier correct a later picture of the same place."""
    test_one_animal_weighed_on_scales_sets_the_scale_of_all = None
    test_every_frame_of_a_video_is_shown_and_written_again_marked_where_the_pig_is = None
    test_a_segment_marked_after_the_video_rescales_every_frame_already_seen = None
    test_rewinding_starts_the_analysis_and_the_numbers_again = None

    def test_correction_from_the_database_applies_only_while_the_scale_is_guessed(self):
        self.session.post("photo", self.photo(picture(pig_mask(scale=.8))))
        rough = self.wait(lambda state: state.animals and state.animals[0].estimate).animals[0]
        self.assertAlmostEqual(rough.raw.mass_kg, rough.estimate.mass_kg)
        self.session.post("learned", (2., 4, .1))
        state = self.wait(lambda state: state.scale_source == LEARNED)
        self.assertEqual(state.learned_pigs, 4)
        self.assertAlmostEqual(state.animals[0].estimate.mass_kg, rough.estimate.mass_kg * 8, delta=.01)
        self.assertAlmostEqual(state.animals[0].raw.mass_kg, rough.raw.mass_kg)
        self.assertIn("взвешенных свиней 4", scale_text(state))
        self.session.post("reference", (SEGMENT_PIXELS, SEEN_SEGMENT_M * 3))
        state = self.wait(lambda state: state.scale_source == SEGMENT)
        self.assertAlmostEqual(state.animals[0].estimate.mass_kg, rough.estimate.mass_kg * 27, delta=rough.estimate.mass_kg * .2)
        self.session.post("reference", None)
        self.session.post("learned", None)
        self.wait(lambda state: state.scale_source == PICTURE)


class TabDatabaseTests(TabCalibrationTests):
    test_typed_weight_of_one_animal_rescales_the_table = None
    test_finished_video_offers_to_write_it_again_and_to_open_it = None

    def test_weighed_pig_goes_to_the_database_and_corrects_the_next_picture(self):
        self.open_with_models()
        self.assertIn("пока нет свиней", self.tab.database_text.get())
        self.show_photo()
        rough = self.tab.state.animals[0].estimate.mass_kg
        self.tab.add_to_database()
        self.assertIn("Введите номер", self.tab.message.get())
        self.tab.known_pig.set("1")
        self.tab.known_mass.set(f"{rough * 2:.2f}")
        self.tab.add_to_database()
        self.assertIn("добавлена в базу", self.tab.message.get())
        self.assertIn("В базе свиней: 1", self.tab.database_text.get())
        self.assertEqual(len(PigDatabase(self.temp.name).entries()), 1)
        # Another picture: its numbers are new, the weight typed for this one no longer applies.
        self.tab.forget_segment()
        self.show_photo()
        self.pump(lambda: "поправлен по базе" in self.tab.scale_text.get())
        self.assertAlmostEqual(self.tab.state.animals[0].estimate.mass_kg, rough * 2, delta=rough * .05)
        self.tab.export_database()
        self.assertTrue(list((Path(self.temp.name) / "output").glob("pigs_database_*.csv")))
        self.tab.remove_last()
        self.assertEqual(PigDatabase(self.temp.name).entries(), [])
        self.pump(lambda: "приблизительный" in self.tab.scale_text.get())


if __name__ == "__main__":
    unittest.main()
