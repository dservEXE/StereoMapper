"""Straight-line distance between two points picked in a cloud."""
import numpy as np

# How many metres one cloud unit holds. Other units have no known scale.
METRES_PER_UNIT = {"metres": 1.0, "millimetres": 0.001}
MILLIMETRES_PER_METRE = 1000.0


def point_distance(first, second) -> float:
    """Euclidean distance between two 3D points, in the cloud's own units."""
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Для замера нужны две точки с координатами X, Y, Z.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("У выбранной точки нет конечных координат.")
    return float(np.linalg.norm(b - a))


def format_distance(distance, units="metres") -> str:
    """Millimetres below one metre, metres from one metre; unscaled clouds keep their units."""
    distance = float(distance)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("Расстояние должно быть конечным и неотрицательным.")
    scale = METRES_PER_UNIT.get(units)
    if scale is None:
        return f"{distance:.4g} усл. ед. (масштаб облака не указан)"
    millimetres = round(distance * scale * MILLIMETRES_PER_METRE, 1)
    if millimetres < MILLIMETRES_PER_METRE:
        return f"{millimetres:.1f} мм"
    return f"{distance * scale:.3f} м"
