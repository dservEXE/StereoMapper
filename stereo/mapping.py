from pathlib import Path

import cv2
import numpy as np
from .map_store import VoxelMap


def estimate_motion(objects, pixels, K):
    if len(objects) < 20:
        return None, 0
    objects, pixels = np.asarray(objects, np.float32), np.asarray(pixels, np.float32)
    if not np.isfinite(objects).all() or not np.isfinite(pixels).all():
        return None, 0
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objects, pixels, K, None, iterationsCount=150, reprojectionError=2.5,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None, 0
    if not ok or inliers is None or len(inliers) < 20 or len(inliers) / len(objects) < 0.45:
        return None, 0 if inliers is None else len(inliers)
    idx = inliers.ravel()
    try:
        rvec, tvec = cv2.solvePnPRefineLM(objects[idx], pixels[idx], K, None, rvec, tvec)
    except cv2.error:
        return None, len(inliers)
    R, _ = cv2.Rodrigues(rvec)
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if not np.isfinite(tvec).all() or np.linalg.norm(tvec) > 0.5 or angle > np.deg2rad(30):
        return None, len(inliers)
    if np.mean((objects[idx] @ R.T + tvec.ravel())[:, 2] > 0) < 0.95:
        return None, len(inliers)
    projected, _ = cv2.projectPoints(np.asarray(objects, np.float32)[idx], rvec, tvec, K, None)
    error = np.linalg.norm(projected.reshape(-1, 2) - np.asarray(pixels)[idx], axis=1)
    if np.median(error) > 2.0:
        return None, len(inliers)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = R, tvec.ravel()
    return transform, len(inliers)


class VisualOdometry:
    """Stereo metric odometry. No loop closure, global optimization or IMU."""
    def __init__(self, K, initial_pose=None):
        self.K = K.copy()
        self.orb = cv2.ORB_create(nfeatures=1800, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reference = None
        self.world_pose = np.eye(4) if initial_pose is None else np.array(initial_pose, dtype=float, copy=True)
        from .cloud_orientation import rotation_matrix
        if (self.world_pose.shape != (4,4) or not np.isfinite(self.world_pose).all()
                or not np.allclose(self.world_pose[3], [0,0,0,1])):
            raise ValueError("Некорректное начальное положение камеры.")
        rotation_matrix(self.world_pose[:3,:3])
        self.keyframes = []
        self.relocalizations = 0
        self.motion_mm = self.motion_deg = 0.

    def set_reference(self, frame):
        gray = cv2.cvtColor(frame.left, cv2.COLOR_BGR2GRAY)
        mask = np.uint8(frame.valid) * 255
        kp, des = self.orb.detectAndCompute(gray, mask)
        if des is None or len(kp) < 80:
            return False
        subpixel = np.asarray([p.pt for p in kp], np.float32)
        coords = np.rint(subpixel).astype(int)
        points = frame.points[coords[:, 1], coords[:, 0]]
        # Keep the measured Z but use the feature's actual ray, avoiding the
        # lateral quantization error of rounding ORB coordinates to a pixel.
        points[:, 0] = (subpixel[:, 0] - self.K[0, 2]) * points[:, 2] / self.K[0, 0]
        points[:, 1] = (subpixel[:, 1] - self.K[1, 2]) * points[:, 2] / self.K[1, 1]
        good = np.isfinite(points).all(axis=1)
        if np.count_nonzero(good) < 80:
            return False
        self.reference = (points[good].copy(), des[good].copy(), self.world_pose.copy())
        remember = not self.keyframes
        if self.keyframes:
            previous = self.keyframes[-1][2]
            angle = np.linalg.norm(cv2.Rodrigues(previous[:3, :3].T @ self.world_pose[:3, :3])[0])
            remember = np.linalg.norm(previous[:3, 3] - self.world_pose[:3, 3]) > .05 or angle > np.deg2rad(15)
        if remember:
            self.keyframes.append((points[good][:800].copy(), des[good][:800].copy(), self.world_pose.copy()))
            if len(self.keyframes) > 40:
                # Keep the first view as a possible anchor after a full turn.
                self.keyframes.pop(1)
        return True

    def match_reference(self, reference, kp, descriptors):
        objects, reference_descriptors, reference_pose = reference
        pairs = self.matcher.knnMatch(reference_descriptors, descriptors, k=2)
        backwards = {m.queryIdx: m.trainIdx for m in self.matcher.match(descriptors, reference_descriptors)}
        selected, used = [], set()
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance:
                m = pair[0]
                if m.trainIdx not in used and backwards.get(m.trainIdx) == m.queryIdx:
                    used.add(m.trainIdx)
                    selected.append(m)
        source = [objects[m.queryIdx] for m in selected]
        target = [kp[m.trainIdx].pt for m in selected]
        motion, inliers = estimate_motion(source, target, self.K)
        return (reference_pose @ np.linalg.inv(motion) if motion is not None else None), inliers

    def update(self, frame):
        if self.reference is None:
            if self.set_reference(frame):
                return self.world_pose.copy(), "Начало карты; двигайте камеру медленно"
            return None, "Нужно больше резких деталей и достоверной глубины"
        gray = cv2.cvtColor(frame.left, cv2.COLOR_BGR2GRAY)
        kp, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(kp) < 20:
            return None, "Слежение потеряно: мало деталей; вернитесь к последнему ракурсу"
        pose, inliers = self.match_reference(self.reference, kp, descriptors)
        recovered = False
        if pose is None:
            candidates = self.keyframes[-10:] + self.keyframes[:1]
            for reference in reversed(candidates):
                candidate, count = self.match_reference(reference, kp, descriptors)
                if candidate is not None and count >= 40:
                    pose, inliers, recovered = candidate, count, True
                    self.relocalizations += 1
                    break
        if pose is None:
            return None, f"Слежение потеряно ({inliers} опор); точки не добавляются"
        self.motion_mm = float(np.linalg.norm(pose[:3, 3] - self.world_pose[:3, 3]) * 1000)
        self.motion_deg = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(self.world_pose[:3, :3].T @ pose[:3, :3])[0])))
        self.world_pose = pose
        self.set_reference(frame)
        return self.world_pose.copy(), ("Положение восстановлено; " if recovered else "") + (
            f"{inliers} опор · смещение {self.motion_mm:.1f} мм · поворот {self.motion_deg:.1f}°; карта накапливается")


def top_view(points, trajectory=None, extent=8.0, size=500, pose=None, surround=False, plane="xz", estimated=None):
    """Projection of measured surface points on XZ, NOT a free-space grid."""
    image = np.full((size, size, 3), (30, 25, 21), np.uint8)
    axes = (0, 1) if plane == "xy" else (0, 2)
    finite = points[np.isfinite(points).all(axis=1)]
    if surround:
        parts = [finite[:, axes]] if len(finite) else []
        if trajectory is not None and len(trajectory):
            parts.append(np.asarray(trajectory)[:, axes])
        if pose is not None:
            parts.append(pose[:3, 3][list(axes)][None])
        if parts:
            extent = max(extent, float(np.max(np.abs(np.vstack(parts)))) * 1.1)
    scale = (size - 40) / (2 * extent if surround else extent)

    def project(p):
        return np.rint(np.column_stack((size / 2 + p[:, axes[0]] * scale,
                                        (size / 2 if surround else size - 20) - p[:, axes[1]] * scale))).astype(int)

    for metres in range(-int(extent) if surround else 0, int(extent) + 1, max(1, int(extent / 8))):
        y = int((size / 2 if surround else size - 20) - metres * scale)
        cv2.line(image, (20, y), (size - 20, y), (58, 48, 39), 1)
        cv2.putText(image, f"{metres} m", (22, y - 3), cv2.FONT_HERSHEY_SIMPLEX, .35, (180, 170, 150), 1)
    if surround:
        cv2.line(image, (size // 2, 20), (size // 2, size - 20), (75, 65, 55), 1)
    if estimated is not None and len(estimated):
        estimates = estimated[np.isfinite(estimated).all(axis=1)]
        pixels = project(estimates)
        good = (pixels[:, 0] >= 0) & (pixels[:, 0] < size) & (pixels[:, 1] >= 0) & (pixels[:, 1] < size)
        image[pixels[good, 1], pixels[good, 0]] = (105, 105, 105)
    if len(finite):
        pixels = project(finite)
        good = (pixels[:, 0] >= 0) & (pixels[:, 0] < size) & (pixels[:, 1] >= 0) & (pixels[:, 1] < size)
        image[pixels[good, 1], pixels[good, 0]] = (220, 197, 82)
    if trajectory is not None and len(trajectory):
        pixels = project(np.asarray(trajectory))
        # Clipping in cv2 keeps paths outside the visible region safe.
        cv2.polylines(image, [pixels.reshape(-1, 1, 2)], False, (72, 177, 255), 2)
    marker = tuple(project(pose[:3, 3][None])[0]) if pose is not None else (size // 2, size // 2 if surround else size - 20)
    cv2.drawMarker(image, marker, (240, 240, 240), cv2.MARKER_TRIANGLE_UP, 14, 2)
    if pose is not None:
        endpoint = tuple(project((pose[:3, 3] + pose[:3, 2] * min(.5, extent / 5))[None])[0])
        cv2.arrowedLine(image, marker, endpoint, (90, 210, 255), 2, tipLength=.25)
    return image


def export_ply(path, points, colors, units="metres"):
    good = np.isfinite(points).all(axis=1)
    points, colors = points[good], colors[good]
    if not len(points):
        raise ValueError("Нет достоверных 3D-точек для сохранения.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\ncomment units {units}; X right Y down Z forward\n")
        stream.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(stream, np.column_stack((points, colors)), fmt="%.5f %.5f %.5f %d %d %d")
