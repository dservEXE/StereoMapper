"""Person measuring, tracking and the weighing tab, on a synthetic person of known size."""
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stereo.app import StereoApp
from stereo.calibration import Calibration
from stereo.camera import CameraConfig
from stereo.depth import ALIGNMENT_INTERVAL, DepthEngine
from stereo.body_measure import (BODY_DENSITY_KG_M3, BodyAccumulator, fill_holes, HANDS_FEET_FACTOR, KEYPOINT_COUNT,
                                 L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, MIN_SAMPLES, PersonObservation,
                                 R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, SLICES, body_slices, measure_person, outline_height,
                                 outline_depth, outline_disparity, part_width, rescaled, slice_profile, slices_volume, span_width)
from stereo.people_detection import (DETECTOR, DOWNLOAD_SIZE_MB, SEGMENTER, SOURCES, crop_matrix, decode_heatmaps, download_models,
                                     model_path, models_ready, outline_hints, peak, pose_crop)
from stereo.experiment_files import experiment_setup
from stereo.people_tracking import PersonTracker, box_iou
from stereo.weighing import WeighingSession
from stereo.worker import CameraWorker
from stereo.weighing_ui import draw_people, experiment_report, experiment_title, person_report, saved_experiments

WIDTH, HEIGHT = 960, 720
K = np.array([[600., 0, 480], [0, 600., 360], [0, 0, 1]])
TORSO_WIDTH, TORSO_DEPTH, TORSO_LENGTH = .40, .25, .52
ARM_LENGTH, ARM_WIDTH = .55, .09
LEG_LENGTH, LEG_WIDTH = .90, .15
HEAD_WIDTH, HEAD_LENGTH = .17, .25
# The drawn legs end in round caps of half their width, which belong to the outline.
TRUE_HEIGHT = HEAD_LENGTH + TORSO_LENGTH + LEG_LENGTH + LEG_WIDTH / 2


def true_mass() -> float:
    """Mass of the synthetic body under the same model, with limbs round and head as ellipsoid."""
    round_part = lambda width, length: math.pi / 4 * width ** 2 * length
    volume = (math.pi / 4 * TORSO_WIDTH * TORSO_DEPTH * TORSO_LENGTH
              + 2 * round_part(ARM_WIDTH, ARM_LENGTH) + 2 * round_part(LEG_WIDTH, LEG_LENGTH)
              + round_part(HEAD_WIDTH, HEAD_LENGTH) * 2 / 3)
    return volume * BODY_DENSITY_KG_M3 * HANDS_FEET_FACTOR


def person_scene(depth=2.5, yaw=0., center=0., wall=4., noise=0., seed=0, color=(40, 40, 200), base=None):
    """Depth frame of a T-pose body seen at a turn of `yaw` degrees, with its joints."""
    scale = K[0, 0] / depth
    turn = math.cos(math.radians(yaw))
    x = lambda metres: K[0, 2] + (center + metres) * scale
    y = lambda metres: K[1, 2] + metres * scale
    if base is None:
        z, image = np.full((HEIGHT, WIDTH), wall, np.float32), np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
    else:
        z, image = base
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    neck, hip, knee, ankle = -.4, -.4 + TORSO_LENGTH, -.4 + TORSO_LENGTH + LEG_LENGTH / 2, -.4 + TORSO_LENGTH + LEG_LENGTH
    half = (TORSO_WIDTH * turn) / 2
    apparent = math.hypot(TORSO_WIDTH * turn, TORSO_DEPTH * math.sin(math.radians(yaw)))
    pt = lambda a, b: (int(round(x(a))), int(round(y(b))))
    thick = lambda metres: max(1, int(round(metres * scale)))
    cv2.rectangle(mask, pt(-apparent / 2, neck), pt(apparent / 2, hip), 255, -1)
    cv2.line(mask, pt(0, neck), pt(0, neck - HEAD_LENGTH + HEAD_WIDTH / 2), 255, thick(HEAD_WIDTH))
    cv2.circle(mask, pt(0, neck - HEAD_LENGTH + HEAD_WIDTH / 2), thick(HEAD_WIDTH / 2), 255, -1)
    arm = half + ARM_LENGTH * turn
    leg = .1 * turn + (LEG_WIDTH / 2 if yaw < 60 else 0)
    for side in (-1, 1):
        cv2.line(mask, pt(side * half, neck + .03), pt(side * arm, neck + .03), 255, thick(ARM_WIDTH))
        cv2.line(mask, pt(side * leg, hip), pt(side * leg, ankle), 255, thick(LEG_WIDTH))
    body = mask > 0
    z = z.copy()
    z[body] = depth
    image = image.copy()
    image[body] = color
    if noise:
        generator = np.random.default_rng(seed)
        z = z + generator.normal(0, noise, z.shape).astype(np.float32)
    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
    points = np.dstack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
    keypoints = np.zeros((KEYPOINT_COUNT, 2), np.float32)
    joints = {L_SHOULDER: (-half, neck + .03), R_SHOULDER: (half, neck + .03),
              L_ELBOW: (-half - ARM_LENGTH * turn / 2, neck + .03), R_ELBOW: (half + ARM_LENGTH * turn / 2, neck + .03),
              L_WRIST: (-arm, neck + .03), R_WRIST: (arm, neck + .03),
              L_HIP: (-leg, hip), R_HIP: (leg, hip), L_KNEE: (-leg, knee), R_KNEE: (leg, knee),
              L_ANKLE: (-leg, ankle), R_ANKLE: (leg, ankle)}
    # Measure the torso from the neck, as a pose model's shoulders are level with it.
    for index, (a, b) in joints.items():
        keypoints[index] = x(a), y(b)
    scores = np.zeros(KEYPOINT_COUNT, np.float32)
    scores[list(joints)] = .9
    box = (x(-arm) - 4, y(neck - HEAD_LENGTH) - 4, x(arm) + 4, y(ankle) + 4)
    return points, np.ones(z.shape, bool), PersonObservation(box, .9, keypoints, scores), z, image


def body_outline(depth=2.5, **kwargs):
    """The exact outline of the synthetic body, as the segmentation model would supply it."""
    kwargs.pop("noise", None), kwargs.pop("seed", None)
    z = person_scene(depth=depth, **kwargs)[3]
    return np.abs(z - depth) < .001


def sample_at(yaw=0., **kwargs):
    points, valid, observation, *_ = person_scene(yaw=yaw, **kwargs)
    return measure_person(points, valid, K, observation, body_outline(yaw=yaw, **kwargs))


def turning_body(frames=60, **kwargs):
    body = BodyAccumulator()
    for index in range(frames):
        yaw = 90 * abs(math.sin(index / frames * math.pi * 2))
        sample = sample_at(yaw=yaw, **kwargs)
        if sample is not None:
            body.add(sample)
    return body


class MeasureTests(unittest.TestCase):
    def test_height_and_widths_come_from_known_body_dimensions(self):
        sample = sample_at()
        self.assertAlmostEqual(sample.depth, 2.5, places=3)
        self.assertAlmostEqual(sample.height, TRUE_HEIGHT, delta=.03)
        self.assertAlmostEqual(sample.widths["torso"], TORSO_WIDTH, delta=.02)
        self.assertAlmostEqual(sample.widths["thigh_l"], LEG_WIDTH, delta=.02)
        self.assertAlmostEqual(sample.widths["forearm_r"], ARM_WIDTH, delta=.02)
        self.assertAlmostEqual(sample.lengths["torso"], TORSO_LENGTH, delta=.04)

    def test_size_does_not_depend_on_distance(self):
        near, far = sample_at(depth=2.), sample_at(depth=3.2, wall=5.)
        self.assertAlmostEqual(near.height, far.height, delta=.03)
        self.assertAlmostEqual(near.widths["torso"], far.widths["torso"], delta=.02)

    def test_wall_right_behind_a_person_is_not_counted_as_body(self):
        sample = sample_at(wall=2.9)
        self.assertAlmostEqual(sample.widths["torso"], TORSO_WIDTH, delta=.02)
        self.assertAlmostEqual(sample.height, TRUE_HEIGHT, delta=.03)

    def test_small_holes_in_depth_do_not_cut_a_part_short(self):
        points, valid, observation, *_ = person_scene()
        valid = valid.copy()
        y, x = (observation.keypoints[L_SHOULDER] + observation.keypoints[R_HIP])[::-1] / 2
        valid[int(y) - 40:int(y) + 40, int(x) + 30] = False
        sample = measure_person(points, valid, K, observation, body_outline())
        self.assertAlmostEqual(sample.widths["torso"], TORSO_WIDTH, delta=.02)

    def test_measurement_survives_depth_noise(self):
        sample = sample_at(noise=.008, seed=2)
        self.assertAlmostEqual(sample.widths["torso"], TORSO_WIDTH, delta=.03)
        self.assertAlmostEqual(sample.height, TRUE_HEIGHT, delta=.06)

    def test_unseen_torso_gives_no_sample_rather_than_a_guess(self):
        points, valid, observation, *_ = person_scene()
        scores = observation.keypoint_scores.copy()
        scores[L_HIP] = .1
        self.assertIsNone(measure_person(points, valid, K, observation._replace(keypoint_scores=scores), body_outline()))

    def test_missing_outline_gives_no_sample(self):
        points, valid, observation, *_ = person_scene()
        self.assertIsNone(measure_person(points, valid, K, observation, None))

    def test_body_distance_ignores_background_filling_plain_clothing(self):
        # As on a live recording: stereo fills part of a plain shirt with the wall behind.
        z = np.full((100, 100), 2.7, np.float32)
        z += np.random.default_rng(1).normal(0, .02, z.shape).astype(np.float32)
        z[:55] = np.random.default_rng(2).uniform(3.2, 4.6, (55, 100)).astype(np.float32)
        mask = np.ones(z.shape, bool)
        self.assertAlmostEqual(outline_depth(z, np.ones(z.shape, bool), mask), 2.7, delta=.05)
        self.assertGreater(float(np.median(z)), 2.75)

    def test_torso_seen_from_the_side_counts_bust_and_buttocks_off_the_bone_line(self):
        # The line from shoulder to hip runs near the back from the side: 5 px one way, 35 the other.
        mask = np.zeros((200, 200), bool)
        mask[:, 95:136] = True
        self.assertAlmostEqual(span_width(mask, (100., 20.), (100., 180.), 1.), 41., delta=2)
        self.assertLess(part_width(mask, (100., 20.), (100., 180.), 1.), 15.)

    def test_too_little_depth_on_the_body_gives_no_distance(self):
        z = np.full((50, 50), 2., np.float32)
        self.assertIsNone(outline_depth(z, np.zeros(z.shape, bool), np.ones(z.shape, bool)))

    def test_legs_held_together_are_not_measured_as_one_wide_leg(self):
        mask = np.zeros((200, 200), bool)
        mask[:, 80:120] = True  # two 20 px legs side by side, bones 20 px apart
        width = part_width(mask, (90., 20.), (90., 180.), 1.)
        self.assertLess(width, 30.)

    def test_no_depth_on_the_torso_gives_no_sample(self):
        points, valid, observation, *_ = person_scene()
        self.assertIsNone(measure_person(points, np.zeros_like(valid), K, observation, body_outline()))

    def test_missing_legs_leave_height_unknown(self):
        points, valid, observation, *_ = person_scene()
        scores = observation.keypoint_scores.copy()
        scores[[L_ANKLE, R_ANKLE, L_KNEE, R_KNEE]] = 0
        sample = measure_person(points, valid, K, observation._replace(keypoint_scores=scores), body_outline())
        self.assertIsNone(sample.height)
        self.assertNotIn("shin_l", sample.widths)


class OutlineDistanceTests(unittest.TestCase):
    @staticmethod
    def body(shift=0):
        mask = np.zeros((480, 640), bool)
        mask[60:420, 280 - shift:360 - shift] = True
        mask[100:130, 200 - shift:440 - shift] = True
        return mask

    def test_shift_between_outlines_is_found_regardless_of_shape(self):
        self.assertEqual(outline_disparity(self.body(), self.body(12)), 12.)

    def test_half_pixel_shift_is_resolved(self):
        left, right = self.body(), self.body(12)
        right[:, :] = False
        right[60:420, 268:347] = True  # one edge moved by a pixel: 12 and 13
        self.assertAlmostEqual(outline_disparity(left, right), 12.5, places=3)

    def test_disagreeing_or_empty_outlines_are_rejected(self):
        empty = np.zeros((480, 640), bool)
        self.assertIsNone(outline_disparity(self.body(), empty))
        other = np.zeros((480, 640), bool)
        other[60:420, 100:120] = True
        wide = np.zeros((480, 640), bool)
        wide[60:240, 280:360] = True
        wide[240:420, 400:600] = True
        self.assertIsNone(outline_disparity(wide, other))
        self.assertIsNone(outline_disparity(self.body(), self.body()))

    def test_body_parts_at_disagreeing_distances_reject_the_frame(self):
        left = np.zeros((480, 640), bool)
        right = np.zeros((480, 640), bool)
        left[60:420, 280:360] = True
        right[60:300, 268:348] = True
        right[300:420, 250:330] = True  # legs traced 18 px further over than the upper body's 12
        self.assertIsNone(outline_disparity(left, right))

    def test_small_differences_between_body_parts_are_tolerated(self):
        left = np.zeros((480, 640), bool)
        right = np.zeros((480, 640), bool)
        left[60:420, 280:360] = True
        right[60:300, 268:348] = True
        right[300:420, 267:347] = True
        self.assertEqual(outline_disparity(left, right), 12.)

    def test_given_distance_replaces_the_depth_map(self):
        points, valid, observation, *_ = person_scene()
        sample = measure_person(points, valid, K, observation, body_outline(), depth=3.)
        self.assertEqual(sample.depth, 3.)
        self.assertAlmostEqual(sample.height, TRUE_HEIGHT * 3 / 2.5, delta=.05)

    def test_session_takes_the_distance_from_both_outlines(self):
        frame = FakeFrame()
        frame.right = np.roll(frame.left, -18, axis=1)
        calls = []

        class TwoViews(FakeDetector):
            def outline(self, image, observation):
                calls.append(observation)
                return frame.outline if image is frame.left else np.roll(frame.outline, -18, axis=1)
        session = WeighingSession(lambda status: None)
        distance = session.outline_distance(TwoViews(), frame, K, .06, frame.outline, frame.observation)
        self.assertAlmostEqual(distance, 600 * .06 / 18, places=3)

    def test_prompt_for_the_second_view_moves_with_the_expected_shift(self):
        frame = FakeFrame()
        frame.right = frame.left
        seen = []

        class Recorder(FakeDetector):
            def outline(self, image, observation):
                seen.append(observation)
                return None
        session = WeighingSession(lambda status: None)
        self.assertIsNone(session.outline_distance(Recorder(), frame, K, .06, frame.outline, frame.observation))
        expected = 600 * .06 / 2.5
        self.assertAlmostEqual(frame.observation.box[0] - seen[0].box[0], expected, delta=.5)

    def test_without_a_baseline_the_depth_map_is_used(self):
        frame = FakeFrame()
        frame.right = frame.left
        session = WeighingSession(lambda status: None)
        self.assertIsNone(session.outline_distance(FakeDetector(), frame, K, None, frame.outline, frame.observation))


class HeightTests(unittest.TestCase):
    def test_height_runs_from_crown_to_soles_along_the_body(self):
        mask = np.zeros((480, 640), bool)
        mask[50:430, 300:340] = True
        self.assertAlmostEqual(outline_height(mask, (0., 1.)), 379, delta=2)

    def test_tilted_body_is_measured_along_its_own_axis(self):
        mask = np.zeros((480, 640), np.uint8)
        cv2.line(mask, (250, 60), (390, 420), 1, 20)
        direction = np.array((140., 360.)) / np.hypot(140, 360)
        self.assertAlmostEqual(outline_height(mask > 0, direction), np.hypot(140, 360), delta=25)

    def test_person_cut_off_by_the_frame_has_no_height(self):
        mask = np.zeros((480, 640), bool)
        mask[100:480, 300:340] = True
        self.assertIsNone(outline_height(mask, (0., 1.)))
        mask = np.zeros((480, 640), bool)
        mask[0:300, 300:340] = True
        self.assertIsNone(outline_height(mask, (0., 1.)))

    def test_outstretched_arms_do_not_change_the_height(self):
        mask = np.zeros((480, 640), bool)
        mask[50:430, 300:340] = True
        plain = outline_height(mask, (0., 1.))
        mask[120:140, 100:540] = True
        self.assertAlmostEqual(outline_height(mask, (0., 1.)), plain, delta=3)


class LimbDepthTests(unittest.TestCase):
    def test_arm_overlapping_the_body_from_the_side_cannot_inflate_the_volume(self):
        plain = BodyAccumulator()
        overlapped = BodyAccumulator()
        for index in range(60):
            yaw = 90 * abs(math.sin(index / 60 * math.pi * 2))
            sample = sample_at(yaw=yaw)
            plain.add(sample)
            widths = dict(sample.widths)
            if yaw >= 60:
                for key in ("upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r"):
                    widths[key] = .30
            overlapped.add(sample._replace(widths=widths))
        # The cap allows 30% on about 7 litres of arms; uncapped this would add about 16 litres.
        self.assertLess(overlapped.estimate().volume_l - plain.estimate().volume_l, 3.)


class SliceTests(unittest.TestCase):
    def test_slices_follow_a_changing_width_along_the_body(self):
        mask = np.zeros((320, 300), bool)
        mask[0:160, 120:180] = True
        mask[160:320, 100:200] = True
        widths, split = body_slices(mask, (150., 0.), (150., 320.), 140, 1.)
        self.assertEqual(len(widths), SLICES)
        self.assertAlmostEqual(widths[3], 60, delta=2)
        self.assertAlmostEqual(widths[-3], 100, delta=2)
        self.assertFalse(any(split))

    def test_two_separate_legs_are_recognised(self):
        mask = np.zeros((320, 300), bool)
        mask[:, 110:130] = True
        mask[:, 170:190] = True
        widths, split = body_slices(mask, (150., 0.), (150., 320.), 140, 1.)
        self.assertTrue(all(split))
        self.assertAlmostEqual(widths[10], 40, delta=2)

    def test_slice_reaching_both_ends_is_an_arm_and_not_counted_as_body(self):
        mask = np.zeros((320, 300), bool)
        mask[:, 130:170] = True
        mask[20:30, :] = True
        widths, _ = body_slices(mask, (150., 0.), (150., 320.), 140, 1.)
        self.assertTrue(math.isnan(widths[2]))
        self.assertAlmostEqual(widths[20], 40, delta=2)

    def test_arms_joining_the_shoulder_slices_are_replaced_by_the_chest_width(self):
        row = [.40] * SLICES
        row[0] = row[1] = .75
        sample = sample_at()._replace(slice_widths=tuple(row), slice_split=(False,) * SLICES)
        profile = slice_profile([sample] * 6)
        self.assertAlmostEqual(float(profile[0]), .40, places=3)
        self.assertAlmostEqual(float(profile[10]), .40, places=3)

    def test_wide_hips_lower_down_are_kept(self):
        row = [.32] * SLICES
        row[14:20] = [.46] * 6
        sample = sample_at()._replace(slice_widths=tuple(row), slice_split=(False,) * SLICES)
        self.assertAlmostEqual(float(slice_profile([sample] * 6)[16]), .46, places=3)

    def test_level_the_outline_missed_is_filled_from_its_neighbours_not_counted_as_nothing(self):
        # As in a saved experiment: the top quarter of the front profile read as zero.
        mask = np.zeros((320, 300), bool)
        mask[80:320, 120:180] = True
        widths, _ = body_slices(mask, (150., 0.), (150., 320.), 140, 1.)
        self.assertTrue(math.isnan(widths[0]))
        sample = sample_at()._replace(slice_widths=widths, slice_split=(False,) * SLICES)
        profile = slice_profile([sample] * 6)
        self.assertAlmostEqual(float(profile[0]), 60, delta=2)

    def test_a_bust_or_buttocks_adds_volume_only_where_it_is(self):
        flat = np.full(SLICES, .30)
        side = np.full(SLICES, .20)
        bigger = side.copy()
        bigger[5:9] = .30
        split = np.zeros(SLICES, bool)
        self.assertGreater(slices_volume(flat, bigger, split, 1.4), slices_volume(flat, side, split, 1.4))
        added = slices_volume(flat, bigger, split, 1.4) - slices_volume(flat, side, split, 1.4)
        self.assertAlmostEqual(added, math.pi / 4 * .30 * .10 * 4 * 1.4 / SLICES, places=5)

    def test_separate_legs_count_as_two_ellipses_not_one_wide_one(self):
        front, side = np.full(SLICES, .32), np.full(SLICES, .16)
        together = slices_volume(front, side, np.zeros(SLICES, bool), 1.)
        apart = slices_volume(front, side, np.ones(SLICES, bool), 1.)
        self.assertAlmostEqual(apart, together, places=6)
        self.assertAlmostEqual(apart, SLICES * 2 * math.pi / 4 * .16 * .16 / SLICES, places=6)


class EstimateTests(unittest.TestCase):
    def test_turning_person_gives_front_and_side_and_a_close_mass(self):
        estimate = turning_body().estimate()
        self.assertTrue(estimate.observed_side)
        self.assertAlmostEqual(estimate.widths_m["torso"], TORSO_WIDTH, delta=.03)
        self.assertAlmostEqual(estimate.side_widths_m["torso"], TORSO_DEPTH, delta=.05)
        self.assertAlmostEqual(estimate.height_m, TRUE_HEIGHT, delta=.04)
        self.assertLess(abs(estimate.mass_kg - true_mass()) / true_mass(), .15, (estimate.mass_kg, true_mass()))
        self.assertLess(estimate.mass_low_kg, estimate.mass_kg)
        self.assertGreater(estimate.mass_high_kg, estimate.mass_kg)

    def test_person_who_does_not_turn_gets_a_preliminary_mass_with_a_wider_range(self):
        body = BodyAccumulator()
        for _ in range(30):
            body.add(sample_at())
        estimate = body.estimate()
        self.assertFalse(estimate.observed_side)
        self.assertIn("предварительная", estimate.message)
        turned = turning_body().estimate()
        narrow = (turned.mass_high_kg - turned.mass_low_kg) / turned.mass_kg
        self.assertGreater((estimate.mass_high_kg - estimate.mass_low_kg) / estimate.mass_kg, narrow)

    def test_range_never_claims_better_than_fifteen_percent(self):
        body = BodyAccumulator()
        for _ in range(30):
            body.add(sample_at())
        estimate = body.estimate()
        self.assertGreaterEqual((estimate.mass_high_kg - estimate.mass_kg) / estimate.mass_kg, .15)

    def test_no_mass_until_enough_frames_are_collected(self):
        body = BodyAccumulator()
        for _ in range(MIN_SAMPLES - 1):
            body.add(sample_at())
        estimate = body.estimate()
        self.assertIsNone(estimate.mass_kg)
        self.assertIn("сбор", estimate.message)

    def test_person_without_legs_in_view_gets_no_mass(self):
        points, valid, observation, *_ = person_scene()
        scores = observation.keypoint_scores.copy()
        scores[[L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]] = 0
        body = BodyAccumulator()
        for _ in range(MIN_SAMPLES + 5):
            body.add(measure_person(points, valid, K, observation._replace(keypoint_scores=scores), body_outline()))
        estimate = body.estimate()
        self.assertIsNone(estimate.mass_kg)
        self.assertIn("целиком", estimate.message)

    def test_wrong_distances_do_not_change_the_measured_sizes(self):
        body = BodyAccumulator()
        generator = np.random.default_rng(4)
        for _ in range(30):
            reading = sample_at()
            body.add(rescaled(reading, reading.depth * float(generator.uniform(.85, 1.25))))
        for sample in body.steady():
            self.assertAlmostEqual(sample.height, body.steady()[0].height, places=6)
            self.assertAlmostEqual(sample.widths["torso"] / sample.height, TORSO_WIDTH / TRUE_HEIGHT, delta=.015)

    def test_a_person_walking_keeps_one_size(self):
        body = BodyAccumulator()
        for depth in np.linspace(2., 3.2, 25):
            body.add(sample_at(depth=float(depth), wall=6.))
        estimate = body.estimate()
        self.assertAlmostEqual(estimate.height_m, TRUE_HEIGHT, delta=.04)
        self.assertAlmostEqual(estimate.widths_m["torso"], TORSO_WIDTH, delta=.03)

    def test_frame_with_a_cut_outline_does_not_set_the_scale(self):
        body = BodyAccumulator()
        for _ in range(20):
            body.add(sample_at())
        reading = sample_at()
        body.add(reading._replace(height=reading.height * .3))
        self.assertEqual(len(body.steady()), 20)
        self.assertAlmostEqual(body.estimate().height_m, TRUE_HEIGHT, delta=.04)

    def test_old_samples_are_dropped_so_memory_stays_bounded(self):
        body = BodyAccumulator()
        sample = sample_at()
        for _ in range(500):
            body.add(sample)
        self.assertLessEqual(len(body.samples), 400)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PersonTracker()

    def scene(self, *people):
        """Each person is (centre in metres, clothing colour)."""
        base, observations = None, []
        for center, color in people:
            *_, observation, z, image = person_scene(center=center, color=color, base=base)
            base = (z, image)
            observations.append(observation)
        return base[1], observations

    def update(self, now, *people):
        image, observations = self.scene(*people)
        results = self.tracker.update(image, observations, [2.5] * len(observations), now)
        return [track.track_id for track, *_ in results]

    def test_people_keep_their_labels_while_walking(self):
        red, blue = (40, 40, 200), (200, 60, 40)
        first = self.update(0., (-1., red), (1., blue))
        self.assertEqual(len(set(first)), 2)
        for step in range(1, 10):
            ids = self.update(step * .2, (-1. + step * .03, red), (1. - step * .03, blue))
            self.assertEqual(ids, first)

    def test_input_order_does_not_change_the_labels(self):
        red, blue = (40, 40, 200), (200, 60, 40)
        a, b = self.update(0., (-1., red), (1., blue))
        self.assertEqual(self.update(.2, (1., blue), (-1., red)), [b, a])

    def test_person_who_leaves_and_returns_gets_the_same_label(self):
        red, blue = (40, 40, 200), (200, 60, 40)
        a, b = self.update(0., (-1., red), (1., blue))
        self.assertEqual(self.update(.2, (1., blue)), [b])
        self.assertEqual(self.update(5., (1., blue), (.0, red)), [b, a])

    def test_new_person_in_different_clothes_gets_a_new_label(self):
        a = self.update(0., (-1., (40, 40, 200)))[0]
        ids = self.update(.2, (-1., (40, 40, 200)), (1., (40, 200, 40)))
        self.assertEqual(ids[0], a)
        self.assertNotEqual(ids[1], a)

    def test_label_is_forgotten_after_a_long_absence(self):
        red = (40, 40, 200)
        a = self.update(0., (-1., red))[0]
        self.assertNotEqual(self.update(100., (-1., red))[0], a)

    def test_box_overlap(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.)


class Fetcher:
    """Serves fake model files; can corrupt the weights."""
    def __init__(self, corrupt=False):
        self.corrupt, self.urls = corrupt, []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.urls.append(url)
        return io.BytesIO(b"bad" if self.corrupt or url.endswith("safetensors") else b"{}")


class ModelFileTests(unittest.TestCase):
    def test_all_three_models_are_required_and_the_size_is_announced_correctly(self):
        self.assertEqual(len(SOURCES), 3)
        self.assertIn(SEGMENTER, SOURCES)
        self.assertEqual(DOWNLOAD_SIZE_MB, round(sum(source.weights_size for source in SOURCES) / 1e6))

    def test_nothing_is_ready_before_download(self):
        with TemporaryDirectory() as temp:
            self.assertFalse(models_ready(temp))

    def test_downloads_are_pinned_to_exact_revisions(self):
        for source in SOURCES:
            self.assertRegex(source.revision, r"^[0-9a-f]{40}$")
            self.assertIn(source.revision, source.url("model.safetensors"))

    def test_corrupt_weights_are_rejected_and_not_kept(self):
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "контрольная"):
                download_models(temp, opener=Fetcher())
            directory = model_path(temp, DETECTOR)
            self.assertFalse((directory / "model.safetensors").exists())
            self.assertFalse((directory / "model.safetensors.part").exists())
            self.assertFalse(models_ready(temp))

    def test_verified_download_makes_models_ready_and_is_not_repeated(self):
        sources = tuple(source.__class__(source.folder, source.repository, source.revision, 16,
                                         hashlib.sha256(source.repository.encode()[:16].ljust(16)).hexdigest())
                        for source in SOURCES)

        def serve(request, timeout=None):
            url = request.full_url
            calls.append(url)
            if url.endswith("safetensors"):
                source = next(item for item in sources if item.repository in url)
                return io.BytesIO(source.repository.encode()[:16].ljust(16))
            return io.BytesIO(b"{}")
        calls = []
        with TemporaryDirectory() as temp, patch("stereo.people_detection.SOURCES", sources):
            download_models(temp, opener=serve)
            self.assertTrue(models_ready(temp))
            count = len(calls)
            download_models(temp, opener=serve)
            self.assertEqual(len(calls), count)


class PoseGeometryTests(unittest.TestCase):
    def test_crop_keeps_the_box_inside_at_the_model_aspect(self):
        center_x, center_y, width, height = pose_crop((100, 50, 60, 200), 192, 256)
        self.assertEqual((center_x, center_y), (130, 150))
        self.assertAlmostEqual(width / height, 192 / 256)
        self.assertGreaterEqual(height, 200)
        self.assertGreaterEqual(width, 60)

    def test_wide_box_grows_in_height_instead(self):
        width, height = pose_crop((0, 0, 300, 100), 192, 256)[2:]
        self.assertAlmostEqual(width / height, 192 / 256)
        self.assertGreaterEqual(width, 300)

    def test_joint_found_in_a_heatmap_maps_back_to_its_image_position(self):
        crop = pose_crop((100, 50, 60, 200), 192, 256)
        matrix = crop_matrix(crop, 192, 256)
        image_x, image_y = 130.4, 90.7
        input_x, input_y = matrix @ (image_x, image_y, 1)
        # The output is four times smaller than the input, as for this model.
        hx, hy = input_x * 47 / 191, input_y * 63 / 255
        u, v = np.meshgrid(np.arange(48.), np.arange(64.))
        heatmap = np.exp(-((u - hx) ** 2 + (v - hy) ** 2) / (2 * 2. ** 2)).astype(np.float32)
        keypoints, scores = decode_heatmaps(heatmap[None], crop)
        self.assertAlmostEqual(float(keypoints[0, 0]), image_x, delta=.6)
        self.assertAlmostEqual(float(keypoints[0, 1]), image_y, delta=.6)
        self.assertAlmostEqual(float(scores[0]), 1., delta=.1)

    def test_peak_at_the_edge_has_no_offset(self):
        self.assertEqual(peak(np.array((3., 2., 1.)), 0), 0.)


class OutlineHintTests(unittest.TestCase):
    def test_hints_cover_the_clothes_between_the_joints(self):
        _, _, observation, *_ = person_scene()
        hints = np.array(outline_hints(observation))
        kp = observation.keypoints
        neck, pelvis = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2, (kp[L_HIP] + kp[R_HIP]) / 2
        chest = neck + (pelvis - neck) * .4
        self.assertLess(float(np.min(np.linalg.norm(hints - chest, axis=1))), 1.)
        thigh = (kp[L_HIP] + kp[L_KNEE]) / 2
        self.assertLess(float(np.min(np.linalg.norm(hints - thigh, axis=1))), 1.)

    def test_unseen_joints_add_no_invented_points(self):
        _, _, observation, *_ = person_scene()
        none = observation._replace(keypoint_scores=np.zeros(KEYPOINT_COUNT, np.float32))
        self.assertEqual(outline_hints(none), [])

    def test_a_top_missed_inside_the_outline_is_filled(self):
        mask = np.zeros((100, 100), bool)
        mask[10:90, 30:70] = True
        mask[30:60, 40:60] = False
        filled = fill_holes(mask)
        self.assertTrue(filled[45, 50])
        self.assertFalse(filled[45, 10])
        self.assertEqual(int(filled.sum()), 80 * 40)


class FakeFrame:
    def __init__(self, **kwargs):
        self.points, self.valid, self.observation, _, self.left = person_scene(**kwargs)
        self.outline = body_outline(**kwargs)


class FakeDetector:
    def __init__(self, status=None):
        self.current = None

    def detect(self, image):
        return [self.current]

    def outline(self, image, observation):
        return self.mask


class SessionTests(unittest.TestCase):
    def run_session(self, count=25, factory=None):
        detector = FakeDetector()
        session = WeighingSession(factory or (lambda status: detector))
        session.start()
        self.addCleanup(self.stop, session)
        for index in range(count):
            frame = FakeFrame(yaw=90 * abs(math.sin(index / count * math.pi * 2)))
            detector.current, detector.mask = frame.observation, frame.outline
            session.submit(frame, K)
            deadline = time.time() + 5
            while (session.frames.qsize() or session.outlines.qsize()) and time.time() < deadline:
                time.sleep(.002)
            time.sleep(.02)
        return session

    @staticmethod
    def stop(session):
        session.stop_event.set()
        session.join(timeout=5)

    def test_collects_measurements_under_one_label(self):
        state = self.run_session().snapshot()
        self.assertEqual([person.track_id for person in state.people], [1])
        self.assertGreater(state.people[0].estimate.samples, 12)
        self.assertIsNotNone(state.image)

    def test_reset_starts_measuring_again(self):
        session = self.run_session()
        session.reset()
        frame = FakeFrame()
        session.submit(frame, K)
        deadline = time.time() + 5
        while session.reset_requested and time.time() < deadline:
            time.sleep(.01)
        time.sleep(.1)
        self.assertLess(len(session.bodies.get(1, BodyAccumulator()).samples), 3)

    def test_saved_result_holds_numbers_only_and_says_it_is_unvalidated(self):
        session = self.run_session()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "deep" / "result.json"
            self.assertEqual(session.save(path), 1)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(data["validated_against_scales"], False)
        self.assertGreater(data["people"][0]["samples"], 12)
        self.assertNotIn("image", json.dumps(data))
        self.assertEqual(len(data["people"][0]["slices_front_m"]), SLICES)

    def test_saved_result_records_the_camera_and_calibration_quality(self):
        session = self.run_session()
        calibration = type("C", (), dict(rms=.42, epipolar_error=.3, baseline=.06, size=(640, 480)))()
        setup = experiment_setup(type("Config", (), dict(mode="usb"))(), calibration)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            session.save(path, setup)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["setup"]["camera"], "usb")
        self.assertEqual(data["setup"]["calibration"], dict(rms_px=.42, vertical_error_px=.3, baseline_m=.06,
                                                            image_size_px=[640, 480]))

    def test_chance_detection_without_a_measurement_is_not_saved(self):
        session = self.run_session()
        with session.lock:
            session.state = replace(session.state, known={**session.state.known, 99: BodyAccumulator().estimate()})
        with TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            session.save(path)
            labels = [person["label"] for person in json.loads(path.read_text(encoding="utf-8"))["people"]]
        self.assertNotIn(99, labels)

    def test_model_loading_failure_is_reported_not_silent(self):
        messages = []

        def broken(status):
            raise OSError("no file")
        session = WeighingSession(broken, messages.append)
        session.start()
        session.join(timeout=5)
        self.assertIn("no file", session.snapshot().message)
        self.assertTrue(messages)

    def test_slow_outlines_do_not_hold_up_the_labels(self):
        detector = FakeDetector()
        frame = FakeFrame()
        detector.current, detector.mask = frame.observation, frame.outline
        release = Event()

        def slow_outline(image, observation):
            release.wait(5)
            return detector.mask
        detector.outline = slow_outline
        session = WeighingSession(lambda status: detector)
        session.start()
        self.addCleanup(self.stop, session)
        self.addCleanup(release.set)
        session.submit(frame, K)
        deadline = time.time() + 5
        while not session.snapshot().people and time.time() < deadline:
            time.sleep(.01)
        self.assertEqual([person.track_id for person in session.snapshot().people], [1])
        self.assertEqual(session.bodies[1].samples, [])

    def test_the_person_measured_longest_ago_goes_next(self):
        session = WeighingSession(lambda status: None)
        session.measured_at = {1: 5., 2: 1., 3: 9.}
        frame = FakeFrame()
        session.request_outline(frame, K, None, [(1, "a"), (2, "b"), (3, "c")])
        self.assertEqual(session.outlines.get()[3], 2)

    def test_newest_frame_replaces_a_waiting_one(self):
        session = WeighingSession(lambda status: FakeDetector())
        first, second = FakeFrame(), FakeFrame()
        session.submit(first, K)
        session.submit(second, K)
        self.assertEqual(session.frames.qsize(), 1)
        self.assertIs(session.frames.get()[0], second)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.estimate = turning_body().estimate()

    def test_report_is_labelled_as_approximate_with_a_range(self):
        text = person_report(2, self.estimate)
        self.assertIn("Человек №2", text)
        self.assertIn("около", text)
        self.assertIn(f"{self.estimate.mass_low_kg:.0f}", text)

    def test_outline_of_the_measured_body_is_visible_on_the_picture(self):
        session = WeighingSession(lambda status: None)
        frame = FakeFrame()
        session.outlines_seen[1] = frame.outline
        state = session.analyse(_single(frame), frame, K, 0.)
        self.assertIs(state.people[0].outline, frame.outline)
        drawn = draw_people(frame.left, state)
        ys, xs = np.nonzero(frame.outline)
        self.assertFalse(np.array_equal(drawn[ys[len(ys) // 2], xs[len(xs) // 2]], frame.left[ys[len(ys) // 2], xs[len(xs) // 2]]))

    def test_drawing_labels_people_without_touching_the_source_image(self):
        frame = FakeFrame()
        original = frame.left.copy()
        state = WeighingSession(lambda status: None).analyse(_single(frame), frame, K, 0.)
        drawn = draw_people(frame.left, state)
        np.testing.assert_array_equal(frame.left, original)
        self.assertFalse(np.array_equal(drawn, original))


def _single(frame):
    detector = FakeDetector()
    detector.current = frame.observation
    return detector


class SpeedPriorityTests(unittest.TestCase):
    def engine(self):
        K1 = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
        size = (640, 480)
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, np.zeros(5), K1, np.zeros(5), size, np.eye(3),
                                                          np.array((-.06, 0., 0.)), alpha=0)
        calibration = Calibration(size, K1, np.zeros(5), K1, np.zeros(5), np.eye(3), np.array((-.06, 0., 0.)),
                                  R1, R2, P1, P2, Q, np.array(roi1), np.array(roi2), .1, .1, CameraConfig().signature())
        return DepthEngine(calibration, 8., 64, "fast", None, False, True, False, estimate_surfaces=False,
                           acceleration="cpu")

    def pair(self):
        left = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 255, (480, 640, 3), np.uint8), (3, 3), 0)
        return left, np.roll(left, -10, axis=1)

    def test_it_skips_the_near_search_and_most_alignment_checks_but_only_when_asked(self):
        engine = self.engine()
        self.addCleanup(engine.close)
        left, right = self.pair()
        with patch("stereo.depth.alignment_error", return_value=.1) as alignment,                 patch.object(engine, "match_pair", wraps=engine.match_pair) as matching:
            engine.compute(left, right)
            normal = matching.call_count
            self.assertEqual(alignment.call_count, 1)
            matching.reset_mock()
            alignment.reset_mock()
            engine.speed_priority = True
            for _ in range(ALIGNMENT_INTERVAL):
                engine.compute(left, right)
            self.assertEqual(matching.call_count, ALIGNMENT_INTERVAL)
            self.assertEqual(alignment.call_count, 1)
            self.assertGreater(normal, 1)
            matching.reset_mock()
            engine.speed_priority = False
            engine.compute(left, right)
            self.assertEqual(matching.call_count, normal)

    def test_alignment_warning_still_comes_through_at_speed(self):
        engine = self.engine()
        self.addCleanup(engine.close)
        left, right = self.pair()
        engine.speed_priority = True
        engine.frames_since_alignment = ALIGNMENT_INTERVAL
        with patch("stereo.depth.alignment_error", return_value=9.):
            self.assertIn("расходится", engine.compute(left, right).quality_warning)

    def test_map_and_scan_keep_the_full_search_by_default(self):
        engine = self.engine()
        self.addCleanup(engine.close)
        self.assertFalse(engine.speed_priority)


class WorkerCommandTests(unittest.TestCase):
    def worker(self, temp):
        worker = CameraWorker(temp, CameraConfig(), (9, 6), 25., 8., 64)
        self.addCleanup(worker.close_maps)
        self.addCleanup(worker.stop_weighing)
        return worker

    def test_weighing_needs_a_calibrated_camera(self):
        with TemporaryDirectory() as temp:
            worker = self.worker(temp)
            with self.assertRaisesRegex(ValueError, "калибровка"):
                worker.command("weighing", True)
            self.assertIsNone(worker.weighing)

    def test_starting_turns_depth_on_and_stopping_ends_the_thread(self):
        with TemporaryDirectory() as temp, patch("stereo.people_detection.PersonDetector", lambda root, status: FakeDetector()):
            worker = self.worker(temp)
            worker.engine = object()
            worker.command("weighing", True)
            session = worker.weighing
            self.assertTrue(session.is_alive())
            self.assertTrue(worker.depth_enabled)
            worker.command("weighing", False)
            self.assertIsNone(worker.weighing)
            self.assertFalse(session.is_alive())

    def test_second_start_keeps_the_same_session_and_its_measurements(self):
        with TemporaryDirectory() as temp, patch("stereo.people_detection.PersonDetector", lambda root, status: FakeDetector()):
            worker = self.worker(temp)
            worker.engine = object()
            worker.command("weighing", True)
            session = worker.weighing
            worker.command("weighing", True)
            self.assertIs(worker.weighing, session)

    def test_weighing_cannot_start_during_an_object_scan_but_can_stop(self):
        with TemporaryDirectory() as temp:
            worker = self.worker(temp)
            worker.engine, worker.scan = object(), object()
            with self.assertRaisesRegex(ValueError, "снимок"):
                worker.command("weighing", True)
            worker.command("weighing", False)
            worker.scan = None

    def test_saving_without_a_session_is_refused_clearly(self):
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "вкладку"):
                self.worker(temp).command("weighing_save", Path(temp) / "r.json")


class ExperimentHistoryTests(unittest.TestCase):
    def save(self, temp, stamp, mass=71.4):
        folder = Path(temp) / "output" / f"weighing_{stamp}"
        folder.mkdir(parents=True)
        person = dict(label=1, samples=30, height_m=1.8, widths_m=dict(torso=.3), side_widths_m=dict(torso=.2),
                      lengths_m={}, volume_l=70., mass_kg=mass, mass_low_kg=60., mass_high_kg=80.,
                      side_observed=True, views=dict(front=9, side=8))
        (folder / "result.json").write_text(json.dumps(dict(people=[person])), encoding="utf-8")

    def test_newest_comes_first_and_broken_results_are_skipped(self):
        with TemporaryDirectory() as temp:
            self.save(temp, "20261106_100000")
            self.save(temp, "20261106_120000")
            broken = Path(temp) / "output" / "weighing_20261106_130000"
            broken.mkdir()
            (broken / "result.json").write_text("{not json", encoding="utf-8")
            names = [name for name, _ in saved_experiments(temp)]
        self.assertEqual(names, ["weighing_20261106_120000", "weighing_20261106_100000"])

    def test_no_folder_means_no_experiments(self):
        with TemporaryDirectory() as temp:
            self.assertEqual(saved_experiments(temp), [])

    def test_title_shows_time_and_mass(self):
        title = experiment_title("weighing_20261106_215432", dict(people=[dict(mass_kg=40.9)]))
        self.assertIn("06.11 21:54:32", title)
        self.assertIn("41 кг", title)

    def test_report_of_a_saved_result_matches_the_live_wording(self):
        with TemporaryDirectory() as temp:
            self.save(temp, "20261106_100000")
            text = experiment_report(saved_experiments(temp)[0][1])
        self.assertIn("Человек №1", text)
        self.assertIn("Рост: 180 см", text)
        self.assertIn("около 71 кг", text)

    def test_result_saved_before_slices_existed_still_opens(self):
        self.assertIn("Человек №3", experiment_report(dict(people=[dict(label=3)])))


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
        self.window = tk.Toplevel(self.host)
        self.app = StereoApp(self.window, Path(self.temp.name))
        self.tab = self.app.weighing_tab
        self.sent = []
        self.app.send = lambda name, value=None: self.sent.append((name, value))

    def tearDown(self):
        self.app.worker = None
        self.window.destroy()
        self.host.update_idletasks()
        self.temp.cleanup()

    def connect(self):
        self.app.connected = True
        self.app.worker = type("Worker", (), dict(engine=object(), frame=None, scan=None, is_tof=False,
                                                  is_alive=lambda self: True))()

    def open_tab(self):
        self.app.show_page("weighing")
        self.host.update()
        self.tab.sync()

    def test_tab_comes_right_after_scanning(self):
        titles = [self.app.workflow.tab(tab, "text") for tab in self.app.workflow.tabs()]
        self.assertEqual(titles.index("4  Взвешивание"), titles.index("3  Сканирование") + 1)

    def test_without_models_it_offers_the_download_and_starts_nothing(self):
        self.connect()
        self.open_tab()
        self.assertEqual(self.sent, [])
        self.assertIn(str(DOWNLOAD_SIZE_MB), self.tab.download_button.cget("text"))

    def test_opening_starts_and_leaving_stops_when_models_are_present(self):
        self.tab.available = True
        self.connect()
        self.open_tab()
        self.assertEqual(self.sent, [("weighing", True)])
        self.app.show_page("scan")
        self.host.update()
        self.tab.sync()
        self.assertEqual(self.sent, [("weighing", True), ("weighing", False)])

    def test_leaving_the_tab_really_reaches_the_weighing_tab(self):
        # A later bind without add="+" once replaced this handler, and weighing kept running.
        script = self.app.workflow.bind("<<NotebookTabChanged>>")
        self.assertEqual(script.count("if {"), 2, script)
        self.tab.available = True
        self.connect()
        self.open_tab()
        self.app.show_page("scan")
        self.window.update()
        self.assertEqual(self.sent[-1], ("weighing", False))

    def test_disconnect_stops_without_sending_to_a_missing_worker(self):
        self.tab.available = True
        self.connect()
        self.open_tab()
        self.app.connected = False
        self.sent.clear()
        self.tab.poll(None)
        self.assertEqual(self.sent, [])
        self.assertFalse(self.tab.active)

    def test_nothing_runs_until_a_camera_is_connected(self):
        self.tab.available = True
        self.open_tab()
        self.assertEqual(self.sent, [])
        self.assertIn("Подключите", self.tab.message.get())

    def test_saved_experiments_are_listed_and_open_when_chosen(self):
        ExperimentHistoryTests().save(self.temp.name, "20261106_100000")
        self.tab.refresh_history()
        rows = self.tab.history_list.get_children()
        self.assertEqual(len(rows), 1)
        self.tab.history_list.selection_set(rows[0])
        self.tab.show_experiment()
        self.assertIn("Человек №1", self.tab.history_text.get())

    def fill_form(self, height="158", weight="53,5", label="A-01"):
        self.tab.entry_height.set(height)
        self.tab.entry_weight.set(weight)
        self.tab.entry_label.set(label)

    def live_state(self):
        frame = FakeFrame()
        session = WeighingSession(lambda status: None)
        session.bodies[1] = BodyAccumulator()
        for _ in range(20):
            session.bodies[1].add(sample_at())
        return session.analyse(_single(frame), frame, K, 2.)

    def test_live_measurement_with_real_data_goes_into_the_database(self):
        self.tab.shown = self.live_state()
        self.tab.refresh_sources()
        self.assertTrue(self.tab.entry_source.get().startswith("Сейчас"))
        self.fill_form()
        self.tab.add_to_database()
        self.assertIn("Записано", self.tab.entry_feedback.get())
        entries = self.tab.database.entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].real_weight_kg, 53.5)
        self.assertEqual(entries[0].label, "A-01")
        self.assertGreater(entries[0].measurement["samples"], 0)
        self.assertEqual(len(self.tab.database_table.get_children()), 1)
        self.assertEqual(self.tab.entry_height.get(), "")

    def test_wrong_input_is_explained_and_nothing_is_saved(self):
        self.tab.shown = self.live_state()
        self.tab.refresh_sources()
        self.fill_form(height="abc")
        self.tab.add_to_database()
        self.assertIn("Рост", self.tab.entry_feedback.get())
        self.assertEqual(self.tab.database.entries(), [])
        self.assertEqual(self.tab.entry_height.get(), "abc")

    def test_nothing_measured_yet_asks_for_a_measurement_first(self):
        self.fill_form()
        self.tab.add_to_database()
        self.assertIn("замер", self.tab.entry_feedback.get())
        self.assertEqual(self.tab.database.entries(), [])

    def test_saved_experiment_can_be_paired_with_real_data_later(self):
        ExperimentHistoryTests().save(self.temp.name, "20261106_100000")
        self.tab.refresh_history()
        self.tab.history_list.selection_set(self.tab.history_list.get_children()[0])
        self.tab.use_experiment()
        self.assertIn("06.11 10:00:00", self.tab.entry_source.get())
        self.fill_form()
        self.tab.add_to_database()
        self.assertEqual(self.tab.database.entries()[0].estimated_weight_kg, 71.4)

    def test_real_data_is_written_into_the_saved_experiment_folder(self):
        ExperimentHistoryTests().save(self.temp.name, "20261106_100000")
        self.tab.refresh_history()
        self.tab.history_list.selection_set(self.tab.history_list.get_children()[0])
        self.tab.use_experiment()
        self.fill_form(height="180", weight="70", label="B-02")
        self.tab.add_to_database()
        path = Path(self.temp.name) / "output" / "weighing_20261106_100000" / "result.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        real = data["people"][0]["real"]
        self.assertEqual((real["label"], real["height_cm"], real["weight_kg"]), ("B-02", 180., 70.))
        self.assertAlmostEqual(real["weight_error_kg"], 1.4)
        self.assertAlmostEqual(real["height_error_cm"], 0.)
        self.assertIs(data["validated_against_scales"], True)
        self.assertEqual(data["people"][0]["mass_kg"], 71.4)
        self.assertIn("Папка замера", self.tab.entry_feedback.get())

    def test_live_measurement_with_real_data_gets_its_own_experiment_folder(self):
        self.tab.shown = self.live_state()
        self.tab.refresh_sources()
        self.fill_form()
        self.tab.add_to_database()
        folders = list((Path(self.temp.name) / "output").glob("weighing_*"))
        self.assertEqual(len(folders), 1)
        data = json.loads((folders[0] / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(data["people"][0]["real"]["weight_kg"], 53.5)
        self.assertIn("saved", data["setup"])
        self.assertEqual(len(self.tab.history_items), 1)

    def test_a_folder_that_cannot_be_written_is_reported_but_the_database_keeps_the_entry(self):
        ExperimentHistoryTests().save(self.temp.name, "20261106_100000")
        self.tab.refresh_history()
        self.tab.refresh_sources()
        self.fill_form()
        with patch("stereo.weighing_ui.attach_real_data", side_effect=OSError("занято")):
            self.tab.add_to_database()
        self.assertIn("не записано: занято", self.tab.entry_feedback.get())
        self.assertEqual(len(self.tab.database.entries()), 1)

    def test_deleting_asks_first_and_removes_only_after_a_yes(self):
        self.tab.shown = self.live_state()
        self.tab.refresh_sources()
        self.fill_form()
        self.tab.add_to_database()
        row = self.tab.database_table.get_children()[0]
        self.tab.database_table.selection_set(row)
        with patch("stereo.weighing_ui.messagebox.askyesno", return_value=False):
            self.tab.delete_entry()
        self.assertEqual(len(self.tab.database.entries()), 1)
        with patch("stereo.weighing_ui.messagebox.askyesno", return_value=True):
            self.tab.delete_entry()
        self.assertEqual(self.tab.database.entries(), [])
        self.assertEqual(self.tab.database_table.get_children(), ())

    def test_export_writes_a_file_and_says_where(self):
        self.tab.shown = self.live_state()
        self.tab.refresh_sources()
        self.fill_form()
        self.tab.add_to_database()
        self.tab.export_database()
        self.assertIn("Сохранено записей: 1", self.tab.database_summary.get())
        self.assertEqual(len(list((Path(self.temp.name) / "output").glob("people_database_*.csv"))), 1)

    def test_large_figures_show_the_person_measured_most(self):
        state = self.live_state()
        self.tab.show_metrics(state)
        self.assertIn("кг", self.tab.metric_weight.get())
        self.assertIn("см", self.tab.metric_height.get())

    def test_poll_shows_the_latest_state_once(self):
        self.tab.available = True
        self.connect()
        self.open_tab()
        frame = FakeFrame()
        state = WeighingSession(lambda status: None).analyse(_single(frame), frame, K, 2.)
        with patch.object(self.tab.panel, "set") as show:
            self.tab.poll(dict(weighing=state))
            self.tab.poll(dict(weighing=state))
        self.assertEqual(show.call_count, 1)
        self.assertIn("Человек №1", self.tab.report.get())


if __name__ == "__main__":
    unittest.main()
