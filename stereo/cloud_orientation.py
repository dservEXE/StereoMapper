"""Explicit scene axes and portable camera bookmarks (OpenCV X right/Y down)."""
import json
from pathlib import Path
import numpy as np
from .trackball import axis_rotation


def rotation_matrix(value):
    if np.iscomplexobj(value):raise ValueError("Матрица вращения должна содержать действительные числа.")
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-6)):
        raise ValueError("Ориентация должна быть матрицей вращения без отражения.")
    return matrix.copy()


def canonical_view(name, frame):
    turns = {"front": np.eye(3), "back": axis_rotation(1, np.pi),
             "top": axis_rotation(0, np.pi/2), "bottom": axis_rotation(0, -np.pi/2),
             "right": axis_rotation(1, np.pi/2), "left": axis_rotation(1, -np.pi/2),
             "side": axis_rotation(1, np.pi/2)}
    return turns[name] @ rotation_matrix(frame)


def frame_from_plane(points, previous):
    """Keep front heading while making the selected plane horizontal.

    The normal sign is chosen closest to the existing up direction. A user
    can flip up explicitly; three unlabelled points cannot resolve that sign.
    """
    p = np.asarray(points, dtype=float)
    old = rotation_matrix(previous)
    if p.shape != (3, 3) or not np.isfinite(p).all():
        raise ValueError("Укажите три точки одной горизонтальной поверхности.")
    a, b = p[1]-p[0], p[2]-p[0]
    up = np.cross(a, b)
    scale = np.linalg.norm(a)*np.linalg.norm(b)
    if scale < 1e-20 or np.linalg.norm(up) <= scale*1e-3:
        raise ValueError("Точки почти на одной прямой. Выберите разнесённые точки плоскости.")
    up /= np.linalg.norm(up)
    if up @ (-old[1]) < 0: up = -up
    forward = old[2] - (old[2] @ up)*up
    if np.linalg.norm(forward) < 1e-6:
        forward = np.cross(old[0], -up)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return rotation_matrix(np.array([right, -up, np.cross(right, -up)]))


def initial_camera_pose(calibration):
    """Rectified-left camera to initial raw-left camera, assumed held level.

    R1 maps raw left rays INTO rectified rays; its inverse cancels calibration
    roll in the world frame. This convention does not measure gravity.
    """
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix(calibration.R1).T
    return pose


def camera_state(value):
    if not isinstance(value, dict): raise ValueError("Повреждено сохранённое положение вида.")
    result = {"rotation": rotation_matrix(value["rotation"]).tolist()}
    for key, shape in (("center", (3,)), ("pan", (2,))):
        array = np.asarray(value[key], dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Некорректные координаты сохранённого вида.")
        result[key] = array.tolist()
    for key in ("zoom", "extent"):
        number = float(value[key])
        if not np.isfinite(number) or number <= 0:
            raise ValueError("Некорректный масштаб сохранённого вида.")
        result[key] = number
    if value.get("projection") not in ("perspective", "orthographic"):
        raise ValueError("Неизвестный тип проекции.")
    result["projection"] = value["projection"]
    return result


def view_state(value):
    """Validate and copy a JSON-only view bundle; no executable serialization."""
    try:
        if value is None: return None
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("Неизвестная версия настроек вида.")
        bookmarks = value["bookmarks"]
        if not isinstance(bookmarks, dict) or len(bookmarks) > 100:
            raise ValueError("Допускается не более 100 сохранённых видов.")
        if any(not isinstance(n, str) or not n.strip() or len(n) > 80 for n in bookmarks):
            raise ValueError("Некорректное название вида.")
        return {"version": 1, "frame": rotation_matrix(value["frame"]).tolist(),
                "current": camera_state(value["current"]),
                "bookmarks": {name: camera_state(camera) for name, camera in bookmarks.items()}}
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Повреждены настройки ориентации и видов.") from error


def views_path(source):
    return Path(str(source) + ".views.json")


def save_views(path, state):
    from .cloud_document import _atomic_write
    encoded = json.dumps({"format": "stereo-cloud-views", "views": view_state(state)},
                         ensure_ascii=False, indent=2)
    _atomic_write(path, lambda stream: stream.write(encoded.encode("utf-8")))


def load_views(path):
    path = Path(path)
    if path.stat().st_size > 1000000: raise ValueError("Слишком большой файл настроек вида.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "stereo-cloud-views" or not isinstance(data.get("views"),dict):
        raise ValueError("Это не файл настроек 3D-вида.")
    return view_state(data["views"])
