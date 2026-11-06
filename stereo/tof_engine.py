"""Turn measured ToF frames into the same metric DepthFrame the stereo engine produces."""
import time

import cv2
import numpy as np

from .depth import DepthFrame, depth_colors, scene_color_limits
from .ghost_filter import supported_disparity
from .mapping import VisualOdometry
from .precision import PrecisionSnapshot, verify_stationary

TOF_MAX_RANGE_M = 2.
TOF_MIN_RANGE_M = .10
STATUS_NORMAL = 0
# Support check in log depth, so the allowed jump is a fraction of distance.
FLYING_LOG_TOLERANCE = .03
FLYING_MIN_NEIGHBOURS = 3
# Measured on a live camera: temporal difference median about 7 mm at 0.6 m.
MEDIAN_AGREEMENT_M = .03
TEMPORAL_ALPHA = .5
# Local contrast: on a live dark scene ORB found 464 features with it and 22 without.
TRACKING_CLIP_LIMIT = 3.
PRECISION_FRAMES = 5
PRECISION_MIN_SAMPLES = 4
# Below the camera's measured frame-to-frame jitter of about 1.6 mm and 0.1 degrees.
STILL_STEP_M, STILL_STEP_DEG = .003, .25
REFINE_LIMIT_M, REFINE_LIMIT_DEG = .03, 3.
ICP_ONLY_STEP_M, ICP_ONLY_STEP_DEG = .06, 6.
ICP_MIN_AGREEMENT = .5
AGREEMENT_STEP = 4


def tof_depth_tolerance(z):
    """Allowed depth disagreement between observations of one surface, in metres."""
    return np.maximum(.006, .015 * np.asarray(z))


def rotation_degrees(rotation) -> float:
    return float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(rotation)[0])))


class TofOdometry(VisualOdometry):
    """Infrared features for reach, geometric ICP for precision, on an ideal pinhole camera.

    Measured on live frames: ICP alone is within 0.4 mm for small steps but fails on
    large ones; features still work there. ICP alone also carries textureless views.
    """

    def __init__(self, K, initial_pose=None):
        # A small patch suits 320 x 240: the default 31-pixel margin discards much of the view.
        super().__init__(K, initial_pose, orb=cv2.ORB_create(nfeatures=1000, fastThreshold=12,
                                                             edgeThreshold=15, patchSize=15))
        self.icp = cv2.rgbd.ICPOdometry_create(self.K.astype(np.float64), TOF_MIN_RANGE_M, TOF_MAX_RANGE_M)
        self.anchor = None
        self.anchor_pose = None
        # Repeating a still view only thickens surfaces with sensor noise and smears people.
        self.still = False

    @staticmethod
    def geometry(frame) -> tuple:
        depth = np.nan_to_num(frame.points[:, :, 2]).astype(np.float32)
        return frame.tracking, depth, np.uint8(depth > 0) * 255

    def refine(self, frame, relative: np.ndarray | None) -> np.ndarray | None:
        """ICP from the anchor view; returns the camera-to-world pose, or None if it fails."""
        source, target = self.geometry(self.anchor), self.geometry(frame)
        guess = np.eye(4) if relative is None else np.linalg.inv(relative)
        try:
            success, moved = self.icp.compute(*source, *target, None, guess)
        except cv2.error:
            return None
        if not success or not np.isfinite(moved).all() or self.agreement(frame, moved) < ICP_MIN_AGREEMENT:
            return None
        return self.anchor_pose @ np.linalg.inv(moved)

    def agreement(self, frame, moved: np.ndarray) -> float:
        """Share of anchor surface samples found again in the new view: ICP can 'succeed' wrongly."""
        points = self.anchor.points[::AGREEMENT_STEP, ::AGREEMENT_STEP]
        points = points[self.anchor.valid[::AGREEMENT_STEP, ::AGREEMENT_STEP]].astype(np.float64)
        if not len(points):
            return 0.
        camera = points @ moved[:3, :3].T + moved[:3, 3]
        z = np.maximum(camera[:, 2], .01)
        u = np.rint(self.K[0, 0] * camera[:, 0] / z + self.K[0, 2]).astype(int)
        v = np.rint(self.K[1, 1] * camera[:, 1] / z + self.K[1, 2]).astype(int)
        height, width = frame.valid.shape
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        measured = frame.points[v[inside], u[inside], 2]
        found = np.abs(np.nan_to_num(measured, nan=1e6) - z[inside]) <= 2 * tof_depth_tolerance(z[inside])
        return float(np.count_nonzero(found) / len(points))

    def choose(self, frame, feature_pose: np.ndarray | None) -> np.ndarray | None:
        if feature_pose is None:
            refined = self.refine(frame, None)
            if refined is None:
                return None
            step = np.linalg.inv(self.anchor_pose) @ refined
            trusted = (np.linalg.norm(step[:3, 3]) <= ICP_ONLY_STEP_M
                       and rotation_degrees(step[:3, :3]) <= ICP_ONLY_STEP_DEG)
            return refined if trusted else None
        refined = self.refine(frame, np.linalg.inv(self.anchor_pose) @ feature_pose)
        if refined is None:
            return feature_pose
        # Geometry may slide along a flat surface; features veto a refinement that drifts away.
        change = np.linalg.inv(feature_pose) @ refined
        agrees = (np.linalg.norm(change[:3, 3]) <= REFINE_LIMIT_M
                  and rotation_degrees(change[:3, :3]) <= REFINE_LIMIT_DEG)
        return refined if agrees else feature_pose

    def update(self, frame):
        if self.anchor is None:
            self.set_reference(frame)
            self.anchor, self.anchor_pose = frame, self.world_pose.copy()
            return self.world_pose.copy(), "Начало карты; двигайте камеру медленно"
        feature_pose, inliers, recovered = (self.locate(frame) if self.reference is not None
                                            else (None, None, False))
        pose = self.choose(frame, feature_pose)
        if pose is None:
            return None, "Слежение потеряно: вернитесь к последнему ракурсу; точки не добавляются"
        step = np.linalg.inv(self.anchor_pose) @ pose
        # Stay on a fixed anchor while still, so sensor noise cannot accumulate into drift
        # and slow movement still adds up to a detectable shift.
        if (np.linalg.norm(step[:3, 3]) < STILL_STEP_M
                and rotation_degrees(step[:3, :3]) < STILL_STEP_DEG):
            self.motion_mm = self.motion_deg = 0.
            self.still = True
            return self.anchor_pose.copy(), "Камера неподвижна; новых точек для карты нет"
        self.still = False
        self.commit(pose, frame)
        self.anchor, self.anchor_pose = frame, pose.copy()
        support = f"{inliers} опор" if feature_pose is not None else "только по геометрии"
        return pose.copy(), ("Положение восстановлено; " if recovered else "") + (
            f"{support} · смещение {self.motion_mm:.1f} мм · поворот {self.motion_deg:.1f}°; карта накапливается")


class TofEngine:
    """Drop-in for DepthEngine: same attributes for the worker, map, scan and interface."""
    acceleration = "ToF · камера измеряет глубину сама"
    acceleration_warning = None
    aligned_preview = True
    source_calibration = None
    depth_tolerance = staticmethod(tof_depth_tolerance)
    odometry_class = TofOdometry

    def __init__(self, calibration, max_depth=TOF_MAX_RANGE_M, temporal=True, near_minimum=TOF_MIN_RANGE_M):
        self.calibration = calibration
        self.maximum = min(float(max_depth), TOF_MAX_RANGE_M)
        self.minimum = max(float(near_minimum), TOF_MIN_RANGE_M)
        self.search_minimum = self.geometric_minimum = self.minimum
        self.temporal = temporal
        self.estimate_surfaces = False
        self.previous = None
        self.last_timings = {}
        width, height = calibration.size
        self.roi_mask = np.ones((height, width), bool)
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        K = calibration.K
        self.ray_x, self.ray_y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        self.maps = [(u, v), (u, v)]
        self.contrast = cv2.createCLAHE(TRACKING_CLIP_LIMIT, (8, 8))

    def accepted(self, depth: np.ndarray, status: np.ndarray) -> np.ndarray:
        valid = (status == STATUS_NORMAL) & (depth >= self.minimum) & (depth <= self.maximum)
        # Mixed returns on silhouettes have no neighbours on a common surface.
        log_depth = np.log(np.maximum(depth, 1e-3))
        return supported_disparity(log_depth, valid, FLYING_MIN_NEIGHBOURS, FLYING_LOG_TOLERANCE)

    def denoise(self, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Median only inside fully measured neighbourhoods: no filling, no edge blending."""
        interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        median = cv2.medianBlur(depth, 3)
        usable = interior & (np.abs(median - depth) <= MEDIAN_AGREEMENT_M)
        return np.where(usable, median, depth)

    def blend_previous(self, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if self.previous is not None:
            old_depth, old_valid = self.previous
            same = valid & old_valid & (np.abs(depth - old_depth) <= tof_depth_tolerance(depth))
            depth = np.where(same, TEMPORAL_ALPHA * depth + (1 - TEMPORAL_ALPHA) * old_depth, depth)
        self.previous = (depth, valid)
        return depth

    def build(self, ir_bgr, depth, valid, color) -> DepthFrame:
        z = np.where(valid, depth, np.nan).astype(np.float32)
        points = np.dstack((self.ray_x * z, self.ray_y * z, z))
        limits = scene_color_limits(z, valid, self.maximum)
        return DepthFrame(color, ir_bgr, np.full(z.shape, np.nan, np.float32), points, valid,
                          depth_colors(z, valid, *limits), method="tof", color_limits=limits,
                          tracking=self.contrast.apply(ir_bgr[:, :, 0]))

    def compute(self, ir_bgr, rgb, depth, status, color) -> DepthFrame:
        started = time.perf_counter()
        valid = self.accepted(depth, status)
        filtered = self.denoise(depth, valid)
        if self.temporal:
            filtered = self.blend_previous(filtered, valid)
        frame = self.build(ir_bgr, filtered, valid, color)
        self.last_timings = {"filter_ms": (time.perf_counter() - started) * 1000}
        return frame

    def single(self, ir_bgr, rgb, depth, status, color) -> DepthFrame:
        """One independent measurement, untouched by temporal memory."""
        valid = self.accepted(depth, status)
        return self.build(ir_bgr, self.denoise(depth, valid), valid, color)

    def distance_report(self) -> list:
        fx = float(self.calibration.K[0, 0])
        return [dict(distance_m=distance, overlap_fraction=1., search_available=self.minimum <= distance <= self.maximum,
                     lateral_mm_per_pixel=distance / fx * 1000) for distance in (.15, .3, .5, 1., 1.5)]

    def close(self) -> None:
        self.previous = None


def tof_precision_snapshot(calibration, raws, max_depth=TOF_MAX_RANGE_M) -> PrecisionSnapshot:
    """Median of several stationary frames, keeping only pixels that repeat."""
    if len(raws) < PRECISION_FRAMES:
        raise ValueError(f"Для точного снимка нужно {PRECISION_FRAMES} отдельных кадров.")
    for raw in raws[1:]:
        verify_stationary(raws[0][0], raw[0])
    engine = TofEngine(calibration, max_depth, temporal=False)
    frames = [engine.single(*raw) for raw in raws]
    stack = np.stack([frame.points[:, :, 2] for frame in frames])
    values = np.ma.masked_invalid(stack)
    z = np.ma.median(values, axis=0).filled(np.nan).astype(np.float32)
    spread = np.ma.median(np.ma.abs(values - z), axis=0).filled(np.inf)
    samples = np.isfinite(stack).sum(axis=0)
    valid = (samples >= PRECISION_MIN_SAMPLES) & np.isfinite(z) & (spread <= tof_depth_tolerance(np.nan_to_num(z)))
    if valid.sum() < 100:
        raise ValueError("Слишком мало устойчивой глубины. Проверьте расстояние и неподвижность камеры.")
    return PrecisionSnapshot(engine.build(raws[-1][0], np.nan_to_num(z), valid, raws[-1][4]), frames,
                             raws[-1], calibration, calibration)
