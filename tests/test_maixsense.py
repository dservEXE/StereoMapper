"""MaixSense A075V protocol, geometry and depth engine, without the physical camera."""
from contextlib import contextmanager
import json
from pathlib import Path
from queue import Empty
import struct
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
from stereo.camera import CameraConfig
from stereo.depth import DepthFrame, tracking_gray
from stereo.ghost_filter import MultiViewConfirmation
from stereo.maixsense import (FrameFormatError, MaixSensePump, TOF_LABEL, TofGeometry, build_frame_config,
                              decode_frame, normalize_address, parse_info)
from stereo.map_integrator import MapIntegrator
from stereo.mapping import VisualOdometry
from stereo.object_scan import ObjectScan
from stereo.precision import PrecisionSnapshot
from stereo.tof_engine import TofEngine, TofOdometry, tof_depth_tolerance, tof_precision_snapshot
from stereo.worker import CameraWorker

# Captured from a real A075V, with the serial number replaced.
LIVE_INFO = bytes.fromhex(
    "4f504e38303038414400000000544553"
    "5453455249414c303030310000000000"
    "00333032095101020062566743c56267"
    "43e33f2643289cf442a8daee3ca433d0"
    "3d5437b7be8096c93a00c1c7b9000000"
    "00")
DEPTH_SIZE = 320 * 240 * 2
STATUS_SIZE = 320 * 240


def make_body(depth=None, ir=None, status=None, rgb=None, frame_id=7, config=None):
    depth = np.full((240, 320), 800, np.uint16) if depth is None else depth
    ir = np.full((240, 320), 500, np.uint16) if ir is None else ir
    status = np.zeros((240, 320), np.uint8) if status is None else status
    jpeg = b""
    if rgb is not None:
        jpeg = cv2.imencode(".jpg", rgb)[1].tobytes()
    deep_size = DEPTH_SIZE * 2 + STATUS_SIZE
    return (struct.pack("<QQ", frame_id, 1234) + (config or build_frame_config())
            + struct.pack("<ii", deep_size, len(jpeg))
            + depth.astype("<u2").tobytes() + ir.astype("<u2").tobytes() + status.tobytes() + jpeg)


def textured_ir(seed=3):
    generator = np.random.default_rng(seed)
    noise = generator.integers(0, 4096, (30, 40), dtype=np.uint16).astype(np.float32)
    return cv2.resize(noise, (320, 240), interpolation=cv2.INTER_CUBIC).clip(0, 4095).astype(np.uint16)


def flat_geometry():
    """Geometry with no lens distortion, for exact pinhole checks."""
    return TofGeometry(parse_info(LIVE_INFO)._replace(distortion=(0., 0., 0., 0., 0.)))


def raw_from(depth_mm=800, status=None, ir=None, rgb=None, geometry=None):
    depth = np.full((240, 320), depth_mm, np.uint16) if np.isscalar(depth_mm) else depth_mm
    frame = decode_frame(make_body(depth=depth, status=status, ir=ir, rgb=rgb))
    return (geometry or flat_geometry()).rectify(frame)


class ProtocolTests(unittest.TestCase):
    def test_decode_splits_all_streams_of_a_16_bit_frame(self):
        depth = np.arange(76800, dtype=np.uint16).reshape(240, 320) % 2000
        ir = (depth + 5).astype(np.uint16)
        status = (depth % 4).astype(np.uint8)
        rgb = np.full((480, 640, 3), 90, np.uint8)
        frame = decode_frame(make_body(depth, ir, status, rgb, frame_id=42))
        self.assertEqual(frame.frame_id, 42)
        np.testing.assert_array_equal(frame.depth, depth)
        np.testing.assert_array_equal(frame.ir, ir)
        np.testing.assert_array_equal(frame.status, status)
        self.assertEqual(frame.rgb.shape, (480, 640, 3))

    def test_frame_without_rgb_is_still_a_valid_depth_frame(self):
        self.assertIsNone(decode_frame(make_body()).rgb)

    def test_frame_still_in_previous_8_bit_mode_is_rejected_not_misread(self):
        stale = build_frame_config(deep_mode=1, ir_mode=1)
        with self.assertRaises(FrameFormatError):
            decode_frame(make_body(config=stale))

    def test_truncated_and_inconsistent_frames_are_rejected(self):
        body = make_body()
        for broken in (b"", body[:20], body[:1000], body[:-1]):
            with self.assertRaises(FrameFormatError):
                decode_frame(broken)

    def test_corrupt_jpeg_keeps_depth_and_reports_no_rgb(self):
        good = make_body()
        header = good[:28] + struct.pack("<ii", DEPTH_SIZE * 2 + STATUS_SIZE, 4)
        frame = decode_frame(header + good[36:] + b"junk")
        self.assertIsNone(frame.rgb)

    def test_info_reads_factory_lens_parameters_of_a_real_camera(self):
        info = parse_info(LIVE_INFO)
        self.assertEqual(info.sensor, "OPN8008AD")
        self.assertEqual(info.serial, "TESTSERIAL0001")
        self.assertAlmostEqual(info.fx, 231.33743, places=3)
        self.assertAlmostEqual(info.fy, 231.38582, places=3)
        self.assertAlmostEqual(info.cx, 166.24956, places=3)
        self.assertAlmostEqual(info.cy, 122.30499, places=3)
        # OpenCV order k1, k2, p1, p2, k3 from the device order k1, k2, k3, p1, p2.
        expected = (0.02916, 0.10166, 0.00154, -0.00038, -0.35784)
        for actual, value in zip(info.distortion, expected):
            self.assertAlmostEqual(actual, value, places=4)

    def test_info_rejects_short_or_nonsense_answers(self):
        for broken in (b"", LIVE_INFO[:50], LIVE_INFO[:41] + b"\0" * 40):
            with self.assertRaises(ValueError):
                parse_info(broken)

    def test_address_accepts_host_and_rejects_anything_but_a_plain_http_host(self):
        self.assertEqual(normalize_address("192.168.233.1"), "http://192.168.233.1")
        self.assertEqual(normalize_address(" http://192.168.233.1/ "), "http://192.168.233.1")
        self.assertEqual(normalize_address("http://camera.local:8080"), "http://camera.local:8080")
        for bad in ("", "ftp://192.168.233.1", "https://192.168.233.1", "http://a:b@192.168.233.1",
                    "http://192.168.233.1/x", "http://192.168.233.1?x=1", "http://"):
            with self.assertRaises(ValueError):
                normalize_address(bad)

    def test_stereo_calibration_signature_is_unaffected_by_the_new_address_field(self):
        """Saved stereo calibrations compare this exact string; it must not change."""
        expected = '{"left": 0, "mode": "dual", "right": 1, "swap": false}'
        self.assertEqual(CameraConfig().signature(), expected)
        self.assertEqual(CameraConfig(address="http://10.0.0.1").signature(), expected)


class GeometryTests(unittest.TestCase):
    def test_undistorted_raw_keeps_tof_size_and_converts_millimetres_to_metres(self):
        ir_bgr, _, depth, status, color = raw_from(1000)
        self.assertEqual(ir_bgr.shape, (240, 320, 3))
        self.assertEqual(depth.dtype, np.float32)
        self.assertAlmostEqual(float(depth[120, 160]), 1.0, places=5)
        self.assertEqual(status.shape, (240, 320))
        self.assertEqual(color.shape, (240, 320, 3))

    def test_lens_correction_moves_corner_pixels_but_not_the_optical_centre(self):
        real = TofGeometry(parse_info(LIVE_INFO))
        self.assertAlmostEqual(float(real.map_x[122, 166]), 166., delta=.6)
        self.assertAlmostEqual(float(real.map_y[122, 166]), 122., delta=.6)
        corner_shift = np.hypot(real.map_x[5, 5] - 5, real.map_y[5, 5] - 5)
        self.assertGreater(float(corner_shift), 1.)

    def test_depth_resampling_never_blends_across_a_depth_edge(self):
        depth = np.full((240, 320), 500, np.uint16)
        depth[:, 160:] = 1500
        _, _, values, _, _ = raw_from(depth, geometry=TofGeometry(parse_info(LIVE_INFO)))
        present = np.unique(np.round(values[values > 0], 3))
        self.assertTrue(set(present.tolist()) <= {.5, 1.5}, present)

    def test_pixels_outside_the_source_are_marked_invalid(self):
        geometry = flat_geometry()
        geometry.map_x = geometry.map_x + 400.
        _, _, depth, status, _ = geometry.rectify(decode_frame(make_body()))
        self.assertTrue((status == 255).all())
        self.assertTrue((depth == 0).all())

    def test_color_uses_rgb_in_the_shared_view_and_infrared_outside_it(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[:, :] = (0, 0, 255)
        ir = np.full((240, 320), 400, np.uint16)
        *_, color = raw_from(ir=ir, rgb=rgb)
        center, corner = color[120, 160], color[2, 2]
        self.assertGreater(int(center[2]), int(center[0]) + 100)
        self.assertEqual(int(corner[0]), int(corner[2]))

    def test_color_sampling_shifts_with_distance_because_the_lenses_are_apart(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[240:] = (0, 255, 0)
        row_of_edge = lambda color: int(np.argmax(color[:, 160, 1] > 128))
        *_, near = raw_from(250, rgb=rgb)
        *_, far = raw_from(1500, rgb=rgb)
        self.assertGreater(abs(row_of_edge(near) - row_of_edge(far)), 15)

    def test_camera_without_rgb_still_gives_a_complete_infrared_view(self):
        ir_bgr, rgb, _, _, color = raw_from()
        np.testing.assert_array_equal(color, ir_bgr)
        np.testing.assert_array_equal(rgb, ir_bgr)

    def test_infrared_view_uses_fixed_scale_so_tracking_is_stable_between_frames(self):
        ir = np.full((240, 320), 1000, np.uint16)
        bright = ir.copy()
        bright[:20, :20] = 4000
        first, *_ = raw_from(ir=ir)
        second, *_ = raw_from(ir=bright)
        self.assertEqual(int(first[200, 200, 0]), int(second[200, 200, 0]))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.geometry = flat_geometry()
        self.engine = TofEngine(self.geometry.calibration, max_depth=1.5, temporal=False)

    def compute(self, *args, **kwargs):
        return self.engine.compute(*raw_from(*args, geometry=self.geometry, **kwargs))

    def test_flat_surface_reprojects_with_pinhole_geometry(self):
        frame = self.compute(1000)
        K = self.geometry.calibration.P1
        self.assertAlmostEqual(float(frame.points[100, 200, 2]), 1., places=4)
        self.assertAlmostEqual(float(frame.points[100, 200, 0]), (200 - K[0, 2]) / K[0, 0], places=4)
        self.assertAlmostEqual(float(frame.points[100, 200, 1]), (100 - K[1, 2]) / K[1, 1], places=4)
        self.assertEqual(frame.method, "tof")
        self.assertEqual(frame.units, "metres")
        self.assertTrue(frame.valid[100, 200])

    def test_only_normal_status_pixels_are_measurements(self):
        status = np.zeros((240, 320), np.uint8)
        status[50:60, 50:60] = 1  # underexposed
        status[70:80, 70:80] = 2  # overexposed
        status[90:100, 90:100] = 3  # bad
        frame = self.compute(800, status=status)
        for y, x in ((55, 55), (75, 75), (95, 95)):
            self.assertFalse(frame.valid[y, x])
            self.assertTrue(np.isnan(frame.points[y, x]).all())
        self.assertTrue(frame.valid[150, 150])

    def test_depth_outside_the_selected_range_is_rejected(self):
        for value in (50, 1900):
            self.assertFalse(self.compute(value).valid.any(), value)

    def test_dim_return_claiming_to_be_close_is_rejected_as_wrapped_phase(self):
        # Values from a real saved frame: a cabinet at about 4 m read as 0.62 m with IR 46.
        frame = self.compute(620, ir=np.full((240, 320), 46, np.uint16))
        self.assertFalse(frame.valid.any())

    def test_bright_near_surface_and_ordinary_wall_are_kept(self):
        for depth_mm, ir in ((330, 1131), (400, 694), (360, 1016), (970, 180), (1400, 90)):
            frame = self.compute(depth_mm, ir=np.full((240, 320), ir, np.uint16))
            self.assertGreater(float(frame.valid.mean()), .9, (depth_mm, ir))

    def test_only_the_wrapped_region_is_removed_from_a_mixed_scene(self):
        depth = np.full((240, 320), 400, np.uint16)
        ir = np.full((240, 320), 800, np.uint16)
        depth[:, 200:], ir[:, 200:] = 620, 46
        frame = self.compute(depth, ir=ir)
        self.assertTrue(frame.valid[:, :190].all())
        self.assertFalse(frame.valid[:, 210:].any())

    def test_independent_measurement_uses_the_same_signal_check(self):
        raw = raw_from(620, ir=np.full((240, 320), 46, np.uint16), geometry=self.geometry)
        self.assertFalse(self.engine.single(*raw).valid.any())

    def test_signal_filter_can_be_switched_off_and_back_on_live(self):
        raw = raw_from(620, ir=np.full((240, 320), 46, np.uint16), geometry=self.geometry)
        self.engine.signal_filter = False
        self.assertGreater(float(self.engine.compute(*raw).valid.mean()), .9)
        self.engine.signal_filter = True
        self.assertFalse(self.engine.compute(*raw).valid.any())

    def test_holes_are_never_filled(self):
        status = np.zeros((240, 320), np.uint8)
        status[100:140, 100:140] = 3
        frame = self.compute(800, status=status)
        self.assertEqual(int(frame.valid[100:140, 100:140].sum()), 0)

    def test_flying_pixel_between_surfaces_is_removed(self):
        depth = np.full((240, 320), 600, np.uint16)
        depth[:, 160:] = 1200
        depth[120, 160] = 900  # lone mixed-range return on the silhouette
        frame = self.compute(depth)
        self.assertFalse(frame.valid[120, 160])
        self.assertTrue(frame.valid[120, 100])
        self.assertTrue(frame.valid[120, 220])

    def test_surfaces_stay_sharp_at_a_depth_edge(self):
        depth = np.full((240, 320), 600, np.uint16)
        depth[:, 160:] = 1200
        z = self.compute(depth).points[:, :, 2]
        values = z[np.isfinite(z)]
        self.assertTrue(np.all((np.abs(values - .6) < .001) | (np.abs(values - 1.2) < .001)))

    def test_temporal_filter_reduces_noise_but_follows_real_motion(self):
        engine = TofEngine(self.geometry.calibration, max_depth=1.5, temporal=True)
        generator = np.random.default_rng(1)
        spreads = []
        for _ in range(6):
            noisy = (800 + generator.normal(0, 4, (240, 320))).astype(np.uint16)
            frame = engine.compute(*raw_from(noisy, geometry=self.geometry))
            spreads.append(float(np.nanstd(frame.points[:, :, 2])))
        self.assertLess(spreads[-1], spreads[0] * .8)
        moved = engine.compute(*raw_from(1100, geometry=self.geometry))
        self.assertAlmostEqual(float(moved.points[120, 160, 2]), 1.1, places=3)

    def test_previous_reset_clears_temporal_memory_after_stream_restart(self):
        engine = TofEngine(self.geometry.calibration, max_depth=1.5, temporal=True)
        engine.compute(*raw_from(800, geometry=self.geometry))
        engine.previous = None
        frame = engine.compute(*raw_from(805, geometry=self.geometry))
        self.assertAlmostEqual(float(frame.points[120, 160, 2]), .805, places=4)

    def test_tracking_uses_infrared_while_the_cloud_uses_color(self):
        rgb = np.full((480, 640, 3), (0, 0, 255), np.uint8)
        frame = self.compute(800, rgb=rgb, ir=textured_ir())
        self.assertEqual(frame.tracking.ndim, 2)
        self.assertGreater(int(frame.left[120, 160, 2]), int(frame.left[120, 160, 0]) + 100)
        self.assertGreater(float(frame.tracking.std()), 5.)

    def test_tracking_gray_falls_back_to_left_for_stereo_frames(self):
        left = np.full((4, 4, 3), 90, np.uint8)
        frame = DepthFrame(left, left, np.zeros((4, 4)), np.zeros((4, 4, 3)), np.ones((4, 4), bool), left)
        self.assertEqual(tracking_gray(frame).shape, (4, 4))

    def test_calibration_is_factory_geometry_and_claims_no_stereo_baseline(self):
        calibration = self.geometry.calibration
        self.assertIsNone(calibration.baseline)
        np.testing.assert_array_equal(calibration.R1, np.eye(3))
        self.assertEqual(calibration.size, (320, 240))
        with TemporaryDirectory() as temp:
            path = Path(temp) / "deep" / "calibration.npz"
            calibration.save(path)
            with np.load(path) as stored:
                self.assertEqual(str(stored["kind"]), "tof")
                np.testing.assert_allclose(stored["K"], calibration.K)

    def test_tolerance_grows_with_distance(self):
        near, far = tof_depth_tolerance(np.array([.3, 1.5]))
        self.assertLess(near, far)
        self.assertGreaterEqual(near, .005)


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.geometry = flat_geometry()
        self.ir = textured_ir()

    def raws(self, values):
        return [raw_from(value, ir=self.ir, geometry=self.geometry) for value in values]

    def test_median_of_five_rejects_one_outlying_frame(self):
        result = tof_precision_snapshot(self.geometry.calibration, self.raws([800, 801, 799, 800, 900]))
        self.assertIsInstance(result, PrecisionSnapshot)
        self.assertAlmostEqual(float(result.frame.points[120, 160, 2]), .8, places=3)
        self.assertIs(result.calibration, result.source_calibration)

    def test_snapshot_follows_the_signal_filter_setting(self):
        dim = (np.full((240, 320), 40, np.uint16) + self.ir // 500).astype(np.uint16)
        raws = [raw_from(620, ir=dim, geometry=self.geometry) for _ in range(5)]
        # This dim view has too little detail for the separate stillness check.
        with patch("stereo.tof_engine.verify_stationary"):
            kept = tof_precision_snapshot(self.geometry.calibration, raws, signal_filter=False)
            self.assertGreater(float(kept.frame.valid.mean()), .9)
            with self.assertRaisesRegex(ValueError, "устойчивой глубины"):
                tof_precision_snapshot(self.geometry.calibration, raws, signal_filter=True)

    def test_needs_five_frames(self):
        with self.assertRaises(ValueError):
            tof_precision_snapshot(self.geometry.calibration, self.raws([800, 801]))

    def test_camera_motion_during_the_snapshot_is_refused(self):
        raws = self.raws([800, 800, 800, 800])
        moved = raw_from(800, ir=np.roll(self.ir, 25, axis=1), geometry=self.geometry)
        with self.assertRaises(ValueError):
            tof_precision_snapshot(self.geometry.calibration, raws + [moved])

    def test_unstable_pixels_are_not_reported_as_measured(self):
        noisy = [np.full((240, 320), 800, np.uint16) for _ in range(5)]
        for index, depth in enumerate(noisy):
            depth[100:110, 100:110] = 500 + index * 100
        result = tof_precision_snapshot(self.geometry.calibration, self.raws(noisy))
        self.assertFalse(result.frame.valid[105, 105])
        self.assertTrue(result.frame.valid[200, 200])


class FakeClient:
    def __init__(self, fail_times=0):
        self.fail_times, self.configured, self.requests = fail_times, 0, 0
        self.frame_id = 0

    def info(self):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("no route")
        return parse_info(LIVE_INFO)

    def configure(self):
        self.configured += 1

    def frame(self):
        self.requests += 1
        time.sleep(.002)
        self.frame_id += 1
        return decode_frame(make_body(frame_id=self.frame_id))


class PumpTests(unittest.TestCase):
    def run_pump(self, client):
        pump = MaixSensePump(CameraConfig(mode="tof"), lambda text: None, client=client, retry_delay=.01)
        pump.start()
        self.addCleanup(self.stop, pump)
        return pump

    @staticmethod
    def stop(pump):
        pump.stop_event.set()
        pump.join(timeout=3)

    def test_delivers_raw_tuple_with_the_capture_pump_interface(self):
        pump = self.run_pump(FakeClient())
        raw, captured, sequence = pump.take(Event())
        self.assertEqual(len(raw), 5)
        self.assertEqual(raw[0].shape, (240, 320, 3))
        self.assertGreater(captured, 0)
        self.assertGreaterEqual(sequence, 1)
        self.assertEqual(pump.info["size"], (320, 240))
        self.assertEqual(pump.info["serial"], "TESTSERIAL0001")
        self.assertIsNotNone(pump.geometry)
        self.assertGreaterEqual(pump.stream_generation, 1)

    def test_camera_is_configured_before_frames_are_read(self):
        client = FakeClient()
        pump = self.run_pump(client)
        pump.take(Event())
        self.assertEqual(client.configured, 1)

    def test_unreachable_camera_fails_with_a_clear_message(self):
        pump = self.run_pump(FakeClient(fail_times=1000))
        with self.assertRaisesRegex(RuntimeError, "MaixSense"):
            pump.take(Event())

    def test_recovers_after_a_short_outage_and_marks_a_new_stream(self):
        class Flaky(FakeClient):
            def frame(self):
                if self.requests == 3:
                    self.requests += 1
                    raise OSError("reset")
                return super().frame()
        pump = self.run_pump(Flaky())
        generations = set()
        for _ in range(12):
            pump.take(Event())
            generations.add(pump.stream_generation)
        self.assertGreaterEqual(len(generations), 2)
        self.assertEqual(pump.restart_count, 1)

    def test_stale_frames_after_mode_switch_are_skipped_without_a_restart(self):
        class Stale(FakeClient):
            def frame(self):
                self.requests += 1
                if self.requests <= 2:
                    raise FrameFormatError("old mode")
                return decode_frame(make_body(frame_id=self.requests))
        pump = self.run_pump(Stale())
        pump.take(Event())
        self.assertEqual(pump.restart_count, 0)

    def test_stops_promptly(self):
        pump = self.run_pump(FakeClient())
        pump.take(Event())
        pump.stop_event.set()
        pump.join(timeout=2)
        self.assertFalse(pump.is_alive())


class MappingCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.geometry = flat_geometry()
        self.engine = TofEngine(self.geometry.calibration, max_depth=1.5, temporal=False)
        self.ir = textured_ir()

    def frame(self):
        return self.engine.compute(*raw_from(800, ir=self.ir, geometry=self.geometry))

    def test_odometry_tracks_on_infrared_even_when_the_color_view_is_blank(self):
        odometry = VisualOdometry(self.geometry.calibration.P1[:, :3])
        first = self.frame()
        first.left[:] = 0
        pose, _ = odometry.update(first)
        self.assertIsNotNone(pose)
        second = self.frame()
        second.left[:] = 0
        pose, status = odometry.update(second)
        self.assertIsNotNone(pose, status)
        self.assertLess(float(np.linalg.norm(pose[:3, 3])), .01)

    def test_confirmation_accepts_repeated_tof_surfaces_without_a_baseline(self):
        K = self.geometry.calibration.P1[:, :3]
        confirmation = MultiViewConfirmation(K, None, 2, depth_tolerance=tof_depth_tolerance)
        frame = self.frame()
        accepted, _ = confirmation.filter(frame, np.eye(4))
        self.assertFalse(accepted.valid.any())
        accepted, _ = confirmation.filter(frame, np.eye(4))
        self.assertGreater(float(accepted.valid.mean()), .9)

    def test_confirmation_rejects_a_surface_that_moved_beyond_sensor_noise(self):
        K = self.geometry.calibration.P1[:, :3]
        confirmation = MultiViewConfirmation(K, None, 2, depth_tolerance=tof_depth_tolerance)
        confirmation.filter(self.frame(), np.eye(4))
        moved = self.engine.compute(*raw_from(1000, ir=self.ir, geometry=self.geometry))
        accepted, _ = confirmation.filter(moved, np.eye(4))
        self.assertFalse(accepted.valid.any())

    def test_stereo_confirmation_still_requires_a_baseline(self):
        K = self.geometry.calibration.P1[:, :3]
        with self.assertRaises((TypeError, ValueError)):
            MultiViewConfirmation(K, None, 2)

    def test_object_scan_accumulates_and_saves_measured_points(self):
        with TemporaryDirectory() as temp:
            engine = TofEngine(self.geometry.calibration, max_depth=1.2, temporal=False)
            scan = ObjectScan(Path(temp) / "scan", self.geometry.calibration, engine=engine,
                              minimum_seconds=5, automatic=False, voxel_mm=2.)
            try:
                raw = raw_from(800, ir=self.ir, geometry=self.geometry)
                for _ in range(6):
                    _, report = scan.process(raw)
                self.assertGreater(report["points"], 1000, report["tracking"])
                result = scan.finish("test")
                self.assertTrue((Path(result["directory"]) / "scan.ply").exists())
                self.assertEqual(result["voxel_mm"], 2.)
            finally:
                scan.close()


class MovingCameraTests(unittest.TestCase):
    """Render a fixed scene from known camera poses and recover the motion."""

    def setUp(self):
        self.geometry = flat_geometry()
        self.K = self.geometry.calibration.K
        self.engine = TofEngine(self.geometry.calibration, max_depth=2., temporal=False)
        depth = np.full((240, 320), 1.0, np.float32)
        depth[60:150, 40:150] = .6
        depth[100:200, 200:290] = .75
        generator = np.random.default_rng(5)
        depth += cv2.resize(generator.normal(0, .03, (12, 16)).astype(np.float32), (320, 240),
                            interpolation=cv2.INTER_CUBIC)
        u, v = np.meshgrid(np.arange(320.), np.arange(240.))
        z = depth.astype(np.float64)
        self.scene = np.dstack(((u - self.K[0, 2]) * z / self.K[0, 0],
                                (v - self.K[1, 2]) * z / self.K[1, 1], z)).reshape(-1, 3)
        self.gray = (textured_ir(9) // 16).astype(np.uint8).reshape(-1)

    def view(self, pose, gray=None):
        """Measured view of the scene from a camera-to-world pose, with occlusion and holes."""
        world_to_camera = np.linalg.inv(pose)
        points = self.scene @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = points[:, 2]
        x = np.rint(self.K[0, 0] * points[:, 0] / z + self.K[0, 2]).astype(int)
        y = np.rint(self.K[1, 1] * points[:, 1] / z + self.K[1, 2]).astype(int)
        keep = np.flatnonzero((x >= 0) & (x < 320) & (y >= 0) & (y < 240) & (z > .05))
        keep = keep[np.argsort(-z[keep])]
        depth = np.zeros((240, 320), np.float32)
        image = np.zeros((240, 320), np.uint8)
        depth[y[keep], x[keep]] = z[keep]
        image[y[keep], x[keep]] = self.gray[keep] if gray is None else gray
        image = cv2.medianBlur(image, 3)
        status = np.where(depth > 0, 0, 1).astype(np.uint8)
        ir_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return self.engine.compute(ir_bgr, ir_bgr, depth, status, ir_bgr)

    @staticmethod
    def pose(x=0., y=0., z=0., degrees=0.):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0, np.deg2rad(degrees), 0.]))[0]
        pose[:3, 3] = (x, y, z)
        return pose

    def assert_close(self, actual, expected, millimetres=6., degrees=.6):
        error = np.linalg.inv(expected) @ actual
        self.assertLess(float(np.linalg.norm(error[:3, 3])) * 1000, millimetres)
        self.assertLess(float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(error[:3, :3])[0]))), degrees)

    def test_follows_a_moving_camera(self):
        odometry = TofOdometry(self.K)
        self.assertIsNotNone(odometry.update(self.view(np.eye(4)))[0])
        for step in range(1, 9):
            expected = self.pose(x=.012 * step, z=.006 * step, degrees=.5 * step)
            pose, status = odometry.update(self.view(expected))
            self.assertIsNotNone(pose, status)
        self.assert_close(pose, expected, millimetres=12., degrees=1.)

    def test_still_camera_keeps_exactly_the_same_pose(self):
        odometry = TofOdometry(self.K)
        first, _ = odometry.update(self.view(np.eye(4)))
        for _ in range(5):
            pose, status = odometry.update(self.view(np.eye(4)))
            np.testing.assert_array_equal(pose, first)
        self.assertIn("неподвижна", status)

    def test_slow_motion_still_adds_up_past_the_still_threshold(self):
        odometry = TofOdometry(self.K)
        odometry.update(self.view(np.eye(4)))
        for step in range(1, 12):
            expected = self.pose(x=.001 * step)
            pose, _ = odometry.update(self.view(expected))
        self.assert_close(pose, expected)
        self.assertGreater(float(pose[0, 3]), .004)

    def test_featureless_view_is_tracked_by_geometry_alone(self):
        odometry = TofOdometry(self.K)
        odometry.update(self.view(np.eye(4), gray=120))
        for step in range(1, 5):
            expected = self.pose(x=.008 * step)
            pose, status = odometry.update(self.view(expected, gray=120))
            self.assertIsNotNone(pose, status)
        self.assertIn("геометрии", status)
        self.assert_close(pose, expected, millimetres=8.)

    def test_a_jump_it_cannot_explain_is_refused_rather_than_guessed(self):
        odometry = TofOdometry(self.K)
        odometry.update(self.view(np.eye(4), gray=120))
        pose, status = odometry.update(self.view(self.pose(x=.45, degrees=35.), gray=120))
        self.assertIsNone(pose, status)
        self.assertIn("потеряно", status)
        recovered, _ = odometry.update(self.view(np.eye(4), gray=120))
        np.testing.assert_array_almost_equal(recovered, np.eye(4))

    def test_still_camera_does_not_write_the_same_view_to_the_map_again(self):
        odometry = TofOdometry(self.K)
        cloud = SimpleNamespace(added=[], add=lambda frame, pose: cloud.added.append(pose))
        integrator = MapIntegrator(odometry, cloud, SimpleNamespace(add=lambda *args: None))
        integrator.start()
        try:
            frame = self.view(np.eye(4))
            for _ in range(4):
                integrator.queue.put(frame)
                deadline = time.time() + 5
                while not integrator.queue.empty() and time.time() < deadline:
                    time.sleep(.01)
                time.sleep(.1)
            self.assertEqual(integrator.attempts, 4)
            self.assertEqual(len(cloud.added), 1)
            integrator.queue.put(self.view(self.pose(x=.02)))
            deadline = time.time() + 5
            while len(cloud.added) < 2 and time.time() < deadline:
                time.sleep(.01)
            self.assertEqual(len(cloud.added), 2)
        finally:
            integrator.stop_event.set()
            integrator.join(timeout=3)

    def test_empty_view_does_not_crash(self):
        odometry = TofOdometry(self.K)
        odometry.update(self.view(np.eye(4)))
        blank = np.zeros((240, 320, 3), np.uint8)
        empty = self.engine.compute(blank, blank, np.zeros((240, 320), np.float32),
                                    np.ones((240, 320), np.uint8), blank)
        pose, status = odometry.update(empty)
        self.assertIsNone(pose, status)


class TexturedClient(FakeClient):
    """A still, detailed scene: enough infrared texture for tracking and a flat wall."""
    def __init__(self):
        super().__init__()
        self.ir = textured_ir()

    def frame(self):
        self.requests += 1
        time.sleep(.01)
        self.frame_id += 1
        return decode_frame(make_body(ir=self.ir, frame_id=self.frame_id))


class WorkerTests(unittest.TestCase):
    """The real worker thread, map and export over a fake camera link."""

    def start_worker(self, root, **kwargs):
        config = CameraConfig(mode="tof")
        client = TexturedClient()
        source = lambda cfg, status: MaixSensePump(cfg, status, client=client, retry_delay=.01)
        patcher = patch("stereo.worker.MaixSensePump", source)
        patcher.start()
        self.addCleanup(patcher.stop)
        worker = CameraWorker(root, config, (9, 6), 25., 8., 64, **kwargs)
        worker.start()
        self.worker = worker
        return worker

    @contextmanager
    def session(self):
        """Stop the worker before its temporary directory goes: Windows locks open files."""
        self.worker = None
        with TemporaryDirectory() as temp:
            try:
                yield temp
            finally:
                if self.worker:
                    self.stop(self.worker)

    @staticmethod
    def stop(worker):
        worker.stop_event.set()
        worker.join(timeout=10)

    @staticmethod
    def wait_for(worker, kinds, seconds=20.):
        seen, deadline = {}, time.time() + seconds
        while time.time() < deadline and not set(kinds) <= set(seen):
            try:
                kind, value = worker.events.get(timeout=.1)
            except Empty:
                continue
            seen[kind] = value
            if kind == "error":
                raise AssertionError(value)
        missing = set(kinds) - set(seen)
        if missing:
            raise AssertionError(f"No events {missing}; got {list(seen)}")
        return seen

    def test_connects_without_any_calibration_and_measures_straight_away(self):
        with self.session() as temp:
            worker = self.start_worker(temp)
            events = self.wait_for(worker, ("connected", "tof_ready", "depth_enabled"))
            self.assertEqual(events["connected"], (320, 240))
            self.assertIs(events["depth_enabled"], True)
            self.assertEqual(events["tof_ready"]["serial"], "TESTSERIAL0001")
            self.assertLessEqual(worker.max_depth, 2.)
            packet = worker.frames.get(timeout=10)
            while packet["depth"] is None:
                packet = worker.frames.get(timeout=10)
            self.assertGreater(float(packet["depth"].valid.mean()), .8)
            self.assertAlmostEqual(float(np.nanmedian(packet["depth"].points[:, :, 2])), .8, places=2)
            self.assertFalse((Path(temp) / "data" / "calibration.npz").exists())

    def test_saved_stereo_calibration_is_neither_loaded_nor_overwritten(self):
        with self.session() as temp:
            stored = Path(temp) / "data" / "calibration.npz"
            stored.parent.mkdir(parents=True)
            stored.write_bytes(b"stereo calibration")
            worker = self.start_worker(temp)
            self.wait_for(worker, ("tof_ready",))
            self.assertEqual(stored.read_bytes(), b"stereo calibration")
            self.assertIsNone(worker.calibration.baseline)

    def test_map_accumulates_points_and_exports_them(self):
        with self.session() as temp:
            worker = self.start_worker(temp)
            self.wait_for(worker, ("tof_ready",))
            worker.commands.put(("mapping", True))
            deadline = time.time() + 20
            count = 0
            while time.time() < deadline and count < 1000:
                try:
                    count = worker.frames.get(timeout=1)["map_count"]
                except Empty:
                    pass
            self.assertGreater(count, 1000)
            worker.commands.put(("export", None))
            exported = self.wait_for(worker, ("exported",))["exported"]
            for name in ("visible_cloud.ply", "accumulated_cloud.ply", "metadata.json", "raw_left.png"):
                self.assertTrue((Path(exported) / name).exists(), name)
            metadata = json.loads((Path(exported) / "metadata.json").read_text(encoding="utf-8"))
            self.assertIsNone(metadata["baseline_m"])

    def test_signal_filter_starts_from_the_setting_and_switches_while_running(self):
        with self.session() as temp:
            worker = self.start_worker(temp, signal_filter=False)
            self.wait_for(worker, ("tof_ready",))
            self.assertFalse(worker.engine.signal_filter)
            worker.commands.put(("signal_filter", True))
            deadline = time.time() + 5
            while not worker.engine.signal_filter and time.time() < deadline:
                time.sleep(.02)
            self.assertTrue(worker.engine.signal_filter)
            self.assertTrue(worker.signal_filter)

    def test_stereo_calibration_commands_are_refused_clearly(self):
        with self.session() as temp:
            worker = self.start_worker(temp)
            self.wait_for(worker, ("tof_ready",))
            for name in ("capture", "calibrate", "new_series", "load", "restore_series"):
                with self.assertRaisesRegex(ValueError, "MaixSense"):
                    worker.command(name)
            worker.command("board_settings", (9, 6, 25.))

    def test_object_scan_runs_through_the_worker(self):
        with self.session() as temp:
            worker = self.start_worker(temp)
            self.wait_for(worker, ("tof_ready",))
            worker.commands.put(("scan_start", dict(voxel_mm=2., maximum_seconds=0, automatic=False,
                                                    max_depth=1.2, near_minimum=.15, width=None)))
            self.wait_for(worker, ("scan_started",))
            deadline, points = time.time() + 20, 0
            while time.time() < deadline and points < 1000:
                try:
                    kind, value = worker.events.get(timeout=.2)
                except Empty:
                    continue
                if kind == "error":
                    self.fail(value)
                if kind == "scan_progress":
                    points = value["points"]
            self.assertGreater(points, 1000)
            worker.commands.put(("scan_stop", None))
            result = self.wait_for(worker, ("scan_finished",))["scan_finished"]
            self.assertTrue((Path(result["directory"]) / "scan.ply").exists())

    def test_precision_snapshot_runs_through_the_worker(self):
        with self.session() as temp:
            worker = self.start_worker(temp)
            self.wait_for(worker, ("tof_ready",))
            worker.commands.put(("precision", None))
            result = self.wait_for(worker, ("precision",))["precision"]
            self.assertAlmostEqual(float(np.nanmedian(result.frame.points[:, :, 2])), .8, places=2)
            self.assertGreater(float(result.frame.valid.mean()), .8)


class AppTests(unittest.TestCase):
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

    def tearDown(self):
        self.app.worker = None
        self.window.destroy()
        self.host.update_idletasks()
        self.temp.cleanup()

    def test_mode_list_offers_the_camera_and_parses_its_address(self):
        self.app.vars["mode"].set(TOF_LABEL)
        self.app.vars["tof_address"].set("192.168.233.1")
        config, *_ = self.app.parse()
        self.assertEqual(config.mode, "tof")
        self.assertEqual(config.address, "http://192.168.233.1")

    def test_bad_address_is_reported_only_when_the_camera_is_selected(self):
        self.app.vars["tof_address"].set("ftp://x")
        self.app.vars["mode"].set("Две камеры")
        self.assertEqual(self.app.parse()[0].mode, "dual")
        self.app.vars["mode"].set(TOF_LABEL)
        with self.assertRaises(ValueError):
            self.app.parse()

    def test_saved_setting_restores_the_mode_and_address(self):
        (Path(self.temp.name) / "settings.json").write_text(json.dumps(
            dict(settings_version=3, mode="tof", tof_address="10.0.0.5")), encoding="utf-8")
        self.app.load_settings()
        self.assertEqual(self.app.vars["mode"].get(), TOF_LABEL)
        self.assertEqual(self.app.vars["tof_address"].get(), "10.0.0.5")

    def test_signal_filter_switch_defaults_on_and_reaches_the_worker_options(self):
        self.assertTrue(self.app.depth_options()["signal_filter"])
        self.app.vars["tof_signal_filter"].set(False)
        self.assertFalse(self.app.depth_options()["signal_filter"])

    def test_signal_filter_switch_is_saved_and_restored(self):
        (Path(self.temp.name) / "settings.json").write_text(json.dumps(
            dict(settings_version=3, tof_signal_filter=False)), encoding="utf-8")
        self.app.load_settings()
        self.assertFalse(self.app.vars["tof_signal_filter"].get())

    def test_changing_the_switch_while_connected_sends_it_at_once(self):
        sent = []
        self.app.send = lambda name, value=None: sent.append((name, value))
        self.app.vars["tof_signal_filter"].set(False)
        self.app.update_signal_filter()
        self.assertEqual(sent, [])
        self.app.connected = True
        self.app.update_signal_filter()
        self.assertEqual(sent, [("signal_filter", False)])

    def test_calibration_actions_are_disabled_for_the_tof_camera(self):
        self.app.worker = SimpleNamespace(is_tof=True, engine=object(), frame=None, scan=None,
                                          is_alive=lambda: True)
        self.app.connected = True
        self.app.calibration_pairs = 20
        self.app.sync_ui(force=True)
        self.assertEqual(str(self.app.capture_button.cget("state")), "disabled")
        self.assertEqual(str(self.app.calibrate_button.cget("state")), "disabled")
        self.assertFalse(self.app.capture_calibration())
        self.assertEqual(self.app.calibration_badge.get(), "Калибровка не нужна")


if __name__ == "__main__":
    unittest.main()
