"""Rigid alignment of saved point clouds, without changing their metric scale.

Landmarks give a global initial pose; ICP only refines an already overlapping
pair.  A low ICP residual alone cannot establish that two scans show the same
object, so the viewer must let the user inspect an alignment before merging it.
All transforms map source coordinates into target coordinates.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ICPResult:
    transform: np.ndarray
    before_rmse: float
    after_rmse: float
    correspondences: int
    converged: bool
    usable: bool
    iterations: int
    reason: str


def _points(values, name, minimum=0):
    if np.iscomplexobj(values):
        raise ValueError(f"{name}: нужны действительные координаты Nx3.")
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: нужны координаты Nx3.") from error
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"{name}: нужно не менее {minimum} точек Nx3.")
    if not np.isfinite(points).all():
        raise ValueError(f"{name}: координаты содержат NaN или бесконечность.")
    return points


def _transform(values):
    if np.iscomplexobj(values):
        raise ValueError("Матрица преобразования должна быть действительной.")
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Нужна жёсткая матрица преобразования 4x4.") from error
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Нужна конечная матрица преобразования 4x4.")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError("Допустимы только поворот и перенос, без масштаба и отражения.")
    return matrix


def _well_spread(points):
    """Three noncollinear points suffice; a planar scan is valid geometry."""
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return len(singular) >= 2 and singular[0] > 0 and singular[1] > singular[0] * 1e-4


def rigid_transform(source, target):
    """Least-squares Kabsch pose from corresponding, noncollinear landmarks.

    ``source[i]`` must identify the same physical feature as ``target[i]``.
    At least three well-separated landmarks are required.  The result is a
    proper 4x4 rigid transform: reflection and scale are never fitted.
    Invalid, coincident or nearly collinear landmark sets raise ValueError.
    """
    source = _points(source, "Исходные ориентиры", 3)
    target = _points(target, "Опорные ориентиры", 3)
    if source.shape != target.shape:
        raise ValueError("Число исходных и опорных ориентиров должно совпадать.")
    if not _well_spread(source) or not _well_spread(target):
        raise ValueError("Выберите разнесённые ориентиры, не лежащие на одной прямой.")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def transform_points(points, transform):
    """Apply a proper rigid 4x4 transform to an Nx3 array, returning float64."""
    points = _points(points, "Облако")
    matrix = _transform(transform)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


class _Nearest:
    def __init__(self, points):
        self.points = points
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            import cv2
            # Centre before float32 conversion so large world translations do
            # not destroy millimetre detail in the FLANN index.
            self.center = points.mean(axis=0)
            self.data = np.ascontiguousarray(points - self.center, dtype=np.float32)
            self.index = cv2.flann_Index(self.data, dict(algorithm=1, trees=4))
            self.tree = None
        else:
            self.tree = cKDTree(points)

    def query(self, points):
        if self.tree is not None:
            _, indices = self.tree.query(points, k=1)
        else:
            query = np.ascontiguousarray(points - self.center, dtype=np.float32)
            indices, _ = self.index.knnSearch(query, 1, params=dict(checks=128))
            indices = indices[:, 0]
        # Compute actual distances in double precision for either backend.
        distances = np.linalg.norm(points - self.points[indices], axis=1)
        return distances, indices


def _sample(points, limit, seed):
    if len(points) <= limit:
        return points
    indices = np.random.default_rng(seed).choice(len(points), limit, replace=False)
    return points[np.sort(indices)]


def _scale(points):
    return float(np.linalg.norm(np.percentile(points, 80, axis=0)
                                - np.percentile(points, 20, axis=0)))


def _pairs(distances, indices, gate):
    valid = np.flatnonzero(distances <= gate)
    if not len(valid):
        return valid
    d = distances[valid]
    median = float(np.median(d))
    mad = float(np.median(np.abs(d - median)))
    cutoff = min(float(np.percentile(d, 80)), median + max(2.5 * mad, gate * 1e-6))
    trimmed = valid[d <= cutoff]
    if len(trimmed) < min(6, len(valid)):
        trimmed = valid[np.argsort(d, kind="stable")[:6]]
    valid = trimmed
    # Many source points must not drag a fit onto a single target patch.
    order = valid[np.argsort(distances[valid], kind="stable")]
    _, first = np.unique(indices[order], return_index=True)
    return order[first]


def refine_icp(source, target, initial=None, max_distance=None,
               max_iterations=40, sample_limit=12000):
    """Robust, trimmed point-to-point ICP for a nearby, overlapping pair.

    ``max_distance`` is in the same units as the clouds (normally metres).
    Its default is 15% of the smaller cloud's robust central extent.  Outlying
    matches are trimmed and each target sample is used only once per step.
    Work is bounded by ``sample_limit`` for each cloud and ``max_iterations``.

    The RMSE values use the *same* initial source support and distance cap, so
    discarding hard matches cannot masquerade as improvement.  A fit needs at
    least six distinct, noncollinear matches and support from at least 12% of
    the smaller sample.  A failed/no-progress result has ``usable=False`` and
    returns the initial pose unchanged; callers must check this flag.  Exact
    initial alignment is usable and converged without moving the cloud.
    ICP is local and does not search for a pose between disjoint scans.
    """
    source = _points(source, "Исходное облако", 6)
    target = _points(target, "Опорное облако", 6)
    initial = np.eye(4) if initial is None else _transform(initial).copy()
    for value, name, minimum in ((max_iterations, "max_iterations", 1),
                                  (sample_limit, "sample_limit", 6)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} должен быть целым числом не менее {minimum}.")
    if max_distance is not None:
        if isinstance(max_distance, (bool, np.bool_)) or not np.isscalar(max_distance):
            raise ValueError("max_distance должен быть положительным числом.")
        try:
            gate = float(max_distance)
        except (TypeError, ValueError) as error:
            raise ValueError("max_distance должен быть положительным числом.") from error
        if not np.isfinite(gate) or gate <= 0:
            raise ValueError("max_distance должен быть положительным числом.")
    source = _sample(source, sample_limit, 17)
    target = _sample(target, sample_limit, 29)
    if max_distance is None:
        gate = .15 * min(_scale(source), _scale(target))

    def failure(reason, before=float("inf"), count=0, iterations=0):
        return ICPResult(initial.copy(), before, before, count, False, False, iterations, reason)

    if gate <= 0 or not _well_spread(source) or not _well_spread(target):
        return failure("degenerate_geometry")
    nearest = _Nearest(target)
    current = transform_points(source, initial)
    distances, indices = nearest.query(current)
    support = _pairs(distances, indices, gate)
    minimum = max(6, int(np.ceil(min(len(source), len(target)) * .12)))
    if len(support) < minimum:
        return failure("insufficient_overlap", count=len(support))
    if not _well_spread(current[support]) or not _well_spread(target[indices[support]]):
        return failure("degenerate_correspondences", count=len(support))
    before = float(np.sqrt(np.mean(distances[support] ** 2)))
    if before <= gate * 1e-6:
        return ICPResult(initial.copy(), before, before, len(support), True, True, 0, "already_aligned")

    best_transform, best_rmse = initial.copy(), before
    best_count, iterations, reason = len(support), 0, "iteration_limit"
    for iteration in range(max_iterations):
        pairs = _pairs(distances, indices, gate)
        if len(pairs) < minimum:
            reason = "insufficient_correspondences"
            break
        try:
            delta = rigid_transform(current[pairs], target[indices[pairs]])
        except (ValueError, np.linalg.LinAlgError):
            reason = "degenerate_correspondences"
            break
        candidate_transform = delta @ best_transform
        candidate = transform_points(source, candidate_transform)
        candidate_distances, candidate_indices = nearest.query(candidate)
        candidate_pairs = _pairs(candidate_distances, candidate_indices, gate)
        # Fixed support plus capped penalties prevents a disappearing overlap
        # region from producing an artificially lower error.
        rmse = float(np.sqrt(np.mean(np.minimum(candidate_distances[support], gate) ** 2)))
        iterations = iteration + 1
        if len(candidate_pairs) < minimum or rmse > best_rmse:
            reason = "stalled"
            break
        improvement = best_rmse - rmse
        movement = float(np.max(np.linalg.norm(candidate[support] - current[support], axis=1)))
        best_transform, best_rmse, best_count = candidate_transform, rmse, len(candidate_pairs)
        current, distances, indices = candidate, candidate_distances, candidate_indices
        if (best_rmse <= gate * 1e-6 or improvement <= max(best_rmse * 1e-5, gate * 1e-9)
                or movement <= gate * 1e-5):
            reason = "converged"
            break

    if before - best_rmse <= max(before * .001, gate * 1e-6):
        return failure("no_improvement", before, len(support), iterations)
    return ICPResult(best_transform, before, best_rmse, best_count, reason == "converged", True,
                     iterations, reason)
