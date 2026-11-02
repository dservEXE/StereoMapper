"""Full-resolution point-cloud editing, independent of the preview's point budget.

Point identities are ``(layer_index, original_vertex_index)``. Edits change boolean
masks and rigid transforms, never the immutable source coordinates. Project files
are self-contained NumPy ZIP containers (normally ``.cloudproj``), without pickle.
"""
from dataclasses import dataclass
from itertools import islice
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .map_store import VoxelMap


_FORMAT = "stereo-cloud-project"
_VERSION = 1


def _units(value):
    value = str(value).strip()
    aliases = {"m": "metres", "meters": "metres", "meter": "metres", "metre": "metres",
               "mm": "millimetres", "millimeters": "millimetres", "millimeter": "millimetres",
               "millimetre": "millimetres"}
    value = aliases.get(value.lower(), value.lower())
    if not value or len(value) > 100 or any(ord(c) < 32 for c in value):
        raise ValueError("Некорректные единицы измерения облака.")
    return value


def _data(points, colors):
    if np.iscomplexobj(points) or np.iscomplexobj(colors):
        raise ValueError("Координаты и цвета должны быть вещественными числами.")
    points = np.array(points, dtype=np.float64, copy=True)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape:
        raise ValueError("Координаты и цвета должны иметь одинаковую форму N × 3.")
    if not np.isfinite(points).all():
        raise ValueError("Облако содержит нечисловые или бесконечные координаты.")
    if not np.issubdtype(colors.dtype, np.number) or not np.isfinite(colors).all():
        raise ValueError("Некорректные цвета точек.")
    if ((colors < 0) | (colors > 255)).any():
        raise ValueError("Компоненты цвета должны находиться в диапазоне 0–255.")
    colors = np.array(np.rint(colors), dtype=np.uint8, copy=True)
    points.flags.writeable = colors.flags.writeable = False
    return points, colors


def _rigid(transform):
    if np.iscomplexobj(transform):
        raise ValueError("Матрица преобразования должна содержать вещественные числа.")
    transform = np.array(transform, dtype=np.float64, copy=True)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0, 0, 0, 1), rtol=0, atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, rtol=0, atol=1e-6)):
        raise ValueError("Ожидается жёсткое преобразование 4 × 4: поворот и перенос без изменения масштаба.")
    transform.flags.writeable = False
    return transform


def _source_path(source):
    return str(Path(source).expanduser().resolve()) if source is not None else None


def _same_path(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _atomic_write(path, write, binary=True):
    """Keep an existing output intact if serialization or replacement fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        kwargs = {} if binary else {"encoding": "ascii", "newline": "\n"}
        with tempfile.NamedTemporaryFile(mode="w+b" if binary else "w", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False,
                                         **kwargs) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass(eq=False)
class CloudLayer:
    name: str
    points: np.ndarray
    colors: np.ndarray
    transform: np.ndarray
    active: np.ndarray
    source: str | None
    selected: np.ndarray


class CloudDocument:
    """Editable layers; all public point arrays use the document's stated units.

    ``select`` accepts a boolean mask aligned with ``arrays()``. Its results stay
    attached to original vertices after other selections and transformations.
    ``apply_transform`` sets an absolute layer-to-world transform. To incrementally
    align a layer, pass ``increment @ document.layers[index].transform``.
    """
    def __init__(self, points, colors, units="metres", source=None):
        self.units = _units(units)
        self.layers = [self._layer(points, colors, source)]
        self.project_path = None
        self.view_state = None
        self._undo, self._redo = [], []

    @staticmethod
    def _layer(points, colors, source):
        points, colors = _data(points, colors)
        source = _source_path(source)
        name = Path(source).stem if source else "Облако"
        return CloudLayer(name, points, colors, _rigid(np.eye(4)), np.ones(len(points), bool),
                          source, np.zeros(len(points), bool))

    @property
    def count(self):
        return sum(int(np.count_nonzero(layer.active)) for layer in self.layers)

    @property
    def total_count(self):
        return sum(len(layer.points) for layer in self.layers)

    @property
    def selection_count(self):
        return sum(int(np.count_nonzero(layer.selected & layer.active)) for layer in self.layers)

    @property
    def selection(self):
        return np.concatenate([layer.selected[layer.active] for layer in self.layers])

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def layer_points(self, index, world=True):
        layer = self.layers[index]
        points = layer.points[layer.active]
        if world:
            points = points @ layer.transform[:3, :3].T + layer.transform[:3, 3]
        return points, layer.colors[layer.active]

    def arrays(self, include_ids=False):
        pieces = [self.layer_points(index) for index in range(len(self.layers))]
        points = np.concatenate([piece[0] for piece in pieces], axis=0)
        colors = np.concatenate([piece[1] for piece in pieces], axis=0)
        if not include_ids:
            return points, colors
        ids = np.concatenate([np.column_stack((np.full(np.count_nonzero(layer.active), index, np.int64),
                                              np.flatnonzero(layer.active)))
                              for index, layer in enumerate(self.layers)], axis=0)
        return points, colors, ids

    def _snapshot(self):
        # Packed masks make history small; immutable points/colors are shared.
        return tuple((layer, np.packbits(layer.active), np.packbits(layer.selected), layer.transform)
                     for layer in self.layers)

    def _restore(self, snapshot):
        self.layers = [item[0] for item in snapshot]
        for layer, active, selected, transform in snapshot:
            layer.active = np.unpackbits(active, count=len(layer.points)).astype(bool)
            layer.selected = np.unpackbits(selected, count=len(layer.points)).astype(bool)
            layer.transform = transform

    def _record(self):
        self._undo.append(self._snapshot())
        self._redo.clear()

    def select(self, mask, mode="replace"):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (self.count,):
            raise ValueError("Маска выделения должна соответствовать всем активным точкам облака.")
        if mode not in ("replace", "add", "subtract"):
            raise ValueError("Неизвестный способ выделения.")
        offset = 0
        for layer in self.layers:
            count = int(np.count_nonzero(layer.active))
            part = mask[offset:offset + count]
            previous = layer.selected[layer.active]
            layer.selected[layer.active] = part if mode == "replace" else (previous | part if mode == "add" else previous & ~part)
            layer.selected[~layer.active] = False
            offset += count
        return self.selection_count

    def clear_selection(self):
        for layer in self.layers:
            layer.selected[:] = False

    def invert_selection(self):
        for layer in self.layers:
            layer.selected = layer.active & ~layer.selected
        return self.selection_count

    def _edit_selection(self, keep, scope):
        selected = self.selection
        if scope is None:
            affected = np.ones(selected.shape, bool)
        else:
            affected = np.asarray(scope)
            if affected.dtype != np.bool_ or affected.shape != selected.shape:
                raise ValueError("Маска области редактирования должна соответствовать всем активным точкам облака.")
        if keep and not np.any(selected & affected):
            raise ValueError("Сначала выделите точки в области, которую нужно оставить.")
        remove = affected & (~selected if keep else selected)
        removed = int(np.count_nonzero(remove))
        if not removed:
            return 0
        self._record()
        offset = 0
        for layer in self.layers:
            count = int(np.count_nonzero(layer.active))
            end = offset + count
            # Selection outside a scoped edit stays attached to its vertices.
            # Assign selection before changing the active mask used as its index.
            layer.selected[layer.active] = layer.selected[layer.active] & ~affected[offset:end]
            layer.active[layer.active] = ~remove[offset:end]
            offset = end
        return removed

    def delete_selected(self, scope=None):
        """Delete selected points, optionally restricted by an active-array mask."""
        return self._edit_selection(False, scope)

    def keep_selected(self, scope=None):
        """Crop within scope; preserve every point and selection outside it."""
        return self._edit_selection(True, scope)

    def add_cloud(self, points, colors, units, source=None):
        if _units(units) != self.units:
            raise ValueError(f"Нельзя объединить облака с разными единицами: {self.units} и {units}. "
                             "Нужен общий физический масштаб.")
        layer = self._layer(points, colors, source)
        self._record()
        self.layers.append(layer)
        self.clear_selection()
        return len(self.layers) - 1

    def apply_transform(self, layer_index, transform):
        transform = _rigid(transform)
        layer = self.layers[layer_index]
        if np.array_equal(transform, layer.transform):
            return False
        self._record()
        layer.transform = transform
        return True

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def _protect_sources(self, path):
        for layer in self.layers:
            if layer.source is not None and _same_path(path, layer.source):
                raise ValueError("Исходный файл защищён: сохраните результат под новым именем.")

    def export_ply(self, path):
        self._protect_sources(path)
        if self.project_path is not None and _same_path(path, self.project_path):
            raise ValueError("Нельзя заменить проект файлом PLY: выберите отдельный файл экспорта.")
        count = self.count
        if not count:
            raise ValueError("Нет активных точек для экспорта.")

        def write(stream):
            stream.write(f"ply\nformat ascii 1.0\ncomment units {self.units}; edited cloud\n"
                         f"element vertex {count}\nproperty double x\nproperty double y\nproperty double z\n"
                         "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for layer in self.layers:
                for start in range(0, len(layer.points), 50000):
                    active = layer.active[start:start + 50000]
                    points = layer.points[start:start + 50000][active]
                    if not len(points):
                        continue
                    points = points @ layer.transform[:3, :3].T + layer.transform[:3, 3]
                    colors = layer.colors[start:start + 50000][active]
                    np.savetxt(stream, np.column_stack((points, colors)), fmt="%.17g %.17g %.17g %d %d %d")

        _atomic_write(path, write, binary=False)
        return count

    def save_project(self, path):
        from .cloud_orientation import view_state
        self._protect_sources(path)
        metadata = {"format": _FORMAT, "version": _VERSION, "units": self.units,
                    "views": view_state(self.view_state),
                    "layers": [{"name": layer.name, "source": layer.source} for layer in self.layers]}
        arrays = {"metadata": np.asarray(json.dumps(metadata, ensure_ascii=False))}
        for index, layer in enumerate(self.layers):
            for name in ("points", "colors", "transform", "active", "selected"):
                arrays[f"layer_{index}_{name}"] = getattr(layer, name)
        _atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))
        self.project_path = str(Path(path).resolve())
        return self.project_path

    @classmethod
    def load_project(cls, path):
        try:
            with np.load(path, allow_pickle=False) as saved:
                encoded = saved["metadata"]
                if encoded.shape != () or encoded.dtype.kind not in ("U", "S") or encoded.nbytes > 1000000:
                    raise ValueError("Некорректные метаданные проекта.")
                encoded_value = encoded.item()
                metadata = json.loads(encoded_value.decode("utf-8") if isinstance(encoded_value, bytes) else encoded_value)
                if metadata.get("format") != _FORMAT or metadata.get("version") != _VERSION:
                    raise ValueError("Неизвестный формат или версия проекта облака.")
                descriptions = metadata["layers"]
                if not isinstance(descriptions, list) or not descriptions:
                    raise ValueError("В проекте отсутствуют слои.")
                result = None
                for index, description in enumerate(descriptions):
                    prefix = f"layer_{index}_"
                    points, colors = saved[prefix + "points"], saved[prefix + "colors"]
                    if result is None:
                        result = cls(points, colors, metadata["units"], description.get("source"))
                    else:
                        result.layers.append(cls._layer(points, colors, description.get("source")))
                    layer = result.layers[-1]
                    layer.name = str(description["name"])
                    layer.transform = _rigid(saved[prefix + "transform"])
                    for key in ("active", "selected"):
                        mask = saved[prefix + key]
                        if mask.dtype != np.bool_ or mask.shape != (len(layer.points),):
                            raise ValueError("Маска точек в проекте повреждена.")
                        setattr(layer, key, mask.copy())
                    if (layer.selected & ~layer.active).any():
                        raise ValueError("Выделение проекта содержит удалённые точки.")
                from .cloud_orientation import view_state
                result.view_state = view_state(metadata.get("views"))
                result.project_path = str(Path(path).resolve())
                return result
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError("Повреждённый проект облака точек.") from error


def _read_ply(path):
    with Path(path).open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
            raise ValueError("Поддерживаются текстовые файлы ASCII PLY.")
        elements, units, current = [], "unknown", None
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "end_header":
                break
            if line.startswith("comment units "):
                units = line.split("comment units ", 1)[1].split(";")[0].strip()
            elif fields[0] == "element":
                if len(fields) != 3 or int(fields[2]) < 0:
                    raise ValueError("Некорректный заголовок PLY.")
                current = {"name": fields[1], "count": int(fields[2]), "properties": []}
                elements.append(current)
            elif fields[0] == "property" and current is not None:
                current["properties"].append(fields[1:])
        else:
            raise ValueError("Заголовок PLY не завершён.")
        vertices = [element for element in elements if element["name"] == "vertex"]
        if len(vertices) != 1 or vertices[0]["count"] == 0:
            raise ValueError("В PLY нет вершин.")
        vertex = vertices[0]
        if any(len(prop) != 2 for prop in vertex["properties"]):
            raise ValueError("Списковые свойства вершин PLY не поддерживаются.")
        properties = [prop[-1] for prop in vertex["properties"]]
        if len(set(properties)) != len(properties) or not all(key in properties for key in ("x", "y", "z")):
            raise ValueError("Некорректные свойства координат PLY.")
        for element in elements:
            if element is vertex:
                break
            if sum(1 for _ in islice(stream, element["count"])) != element["count"]:
                raise ValueError("Файл PLY оборван.")
        points = np.empty((vertex["count"], 3), np.float64)
        colors = np.full(points.shape, 190., np.float64)
        xyz = [properties.index(key) for key in ("x", "y", "z")]
        rgb = ([properties.index(key) for key in ("red", "green", "blue")]
               if all(key in properties for key in ("red", "green", "blue")) else None)
        # Retain all vertices without retaining a second whole cloud as text.
        for start in range(0, vertex["count"], 50000):
            size = min(50000, vertex["count"] - start)
            rows = list(islice(stream, size))
            if len(rows) != size:
                raise ValueError("Число вершин не совпадает с заголовком PLY.")
            data = np.loadtxt(rows, ndmin=2)
            if data.shape != (size, len(properties)):
                raise ValueError("Данные вершин не совпадают с заголовком PLY.")
            points[start:start + size] = data[:, xyz]
            if rgb is not None:
                colors[start:start + size] = data[:, rgb]
        points, colors = _data(points, colors)
        return points, colors, _units(units)


def load_cloud(path):
    """Load every stored point, with no preview budget or implicit downsampling.

    Returns ``(float64_points, uint8_RGB, units, absolute_source_path)``. Project
    inputs are flattened here; use ``CloudDocument.load_project`` to keep layers.
    """
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        signature = stream.read(16)
    if signature.startswith(b"SQLite format 3\0"):
        archive = VoxelMap(path=path, readonly=True)
        try:
            points, colors = archive.arrays()
            points, colors = _data(points, colors)
        finally:
            archive.close()
        units = "metres"
    elif signature.startswith(b"PK"):
        document = CloudDocument.load_project(path)
        points, colors = document.arrays()
        units = document.units
    else:
        points, colors, units = _read_ply(path)
    return points, colors, units, str(path)
