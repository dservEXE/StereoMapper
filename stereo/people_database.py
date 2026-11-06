"""Local database of measured people with their real height and weight.

Each record pairs what the camera measured with what is known to be true, so the estimate can
be checked and later improved from data. It stays on this computer. A label can be a code
rather than a name, and no images are stored.
"""
from contextlib import closing
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

from .body_measure import SLICES

FOLDER = "people"
DATABASE_FILE = "people.sqlite"
SEXES = ("не указан", "женский", "мужской")
CLOTHING = ("облегающая", "обычная", "свободная")
MAX_LABEL_LENGTH = 40
HEIGHT_RANGE_CM = (50., 250.)
WEIGHT_RANGE_KG = (3., 300.)
# Below this many different people an average error says little about other people.
ENOUGH_PEOPLE = 5
MEASURED_PARTS = ("torso", "head", "upper_arm", "forearm", "thigh", "shin")
# A spreadsheet runs a text cell starting with one of these as a formula.
SPREADSHEET_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")

_SCHEMA = """CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    label TEXT NOT NULL,
    sex TEXT NOT NULL,
    real_height_cm REAL NOT NULL,
    real_weight_kg REAL NOT NULL,
    weight_from_scales INTEGER NOT NULL,
    clothing TEXT NOT NULL,
    camera TEXT NOT NULL,
    app_version TEXT NOT NULL,
    measurement TEXT NOT NULL)"""
_COLUMNS = ("id, created, label, sex, real_height_cm, real_weight_kg, weight_from_scales, "
            "clothing, camera, app_version, measurement")


@dataclass(frozen=True)
class PersonEntry:
    id: int | None
    created: str
    label: str
    sex: str
    real_height_cm: float
    real_weight_kg: float
    weight_from_scales: bool
    clothing: str
    camera: str
    app_version: str
    measurement: dict

    @property
    def estimated_weight_kg(self) -> float | None:
        return self.measurement.get("mass_kg")

    @property
    def estimated_height_cm(self) -> float | None:
        height = self.measurement.get("height_m")
        return height * 100 if height else None

    @property
    def weight_error_kg(self) -> float | None:
        estimate = self.estimated_weight_kg
        return estimate - self.real_weight_kg if estimate is not None else None

    @property
    def height_error_cm(self) -> float | None:
        estimate = self.estimated_height_cm
        return estimate - self.real_height_cm if estimate is not None else None


@dataclass(frozen=True)
class DatabaseSummary:
    records: int
    people: int
    compared: int
    from_scales: int
    mean_error_kg: float | None
    mean_absolute_error_kg: float | None
    mean_absolute_error_percent: float | None
    height_absolute_error_cm: float | None


def parse_number(text, name: str, limits: tuple) -> float:
    """A number typed by a person, accepting a decimal comma, within believable limits."""
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"{name}: введите число.") from None
    if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
        raise ValueError(f"{name}: допустимо от {limits[0]:g} до {limits[1]:g}.")
    return value


def clean_label(text: str) -> str:
    label = " ".join(str(text).split())
    if len(label) > MAX_LABEL_LENGTH:
        raise ValueError(f"Метка: не длиннее {MAX_LABEL_LENGTH} знаков.")
    return label


def finite_only(value):
    """JSON has no NaN: an unknown number is stored as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_only(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_only(item) for item in value]
    return value


def make_entry(measurement, label, height_text, weight_text, sex, clothing, from_scales, camera, app_version) -> PersonEntry:
    """Check what a person typed. Raises ValueError with a message meant for them."""
    if not measurement or not measurement.get("samples"):
        raise ValueError("Сначала выберите замер, в котором есть измерения.")
    if sex not in SEXES:
        raise ValueError("Пол: выберите значение из списка.")
    if clothing not in CLOTHING:
        raise ValueError("Одежда: выберите значение из списка.")
    return PersonEntry(None, datetime.now().isoformat(timespec="seconds"), clean_label(label), sex,
                       parse_number(height_text, "Рост", HEIGHT_RANGE_CM),
                       parse_number(weight_text, "Вес", WEIGHT_RANGE_KG),
                       bool(from_scales), clothing, camera or "", app_version, finite_only(measurement))


class PeopleDatabase:
    """One short connection per call: only the interface thread uses it, and rarely."""

    def __init__(self, root):
        self.path = Path(root) / "data" / FOLDER / DATABASE_FILE

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        with connection:
            connection.execute(_SCHEMA)
        return connection

    def add(self, entry: PersonEntry) -> int:
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO people (created, label, sex, real_height_cm, real_weight_kg, weight_from_scales, "
                "clothing, camera, app_version, measurement) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry.created, entry.label, entry.sex, entry.real_height_cm, entry.real_weight_kg,
                 int(entry.weight_from_scales), entry.clothing, entry.camera, entry.app_version,
                 json.dumps(entry.measurement, ensure_ascii=False, allow_nan=False)))
            return int(cursor.lastrowid)

    def entries(self) -> list:
        """Newest first. No file yet simply means no records."""
        if not self.path.exists():
            return []
        with closing(self.connect()) as connection:
            rows = connection.execute(f"SELECT {_COLUMNS} FROM people ORDER BY id DESC").fetchall()
        return [_entry_from_row(row) for row in rows]

    def delete(self, entry_id: int) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM people WHERE id = ?", (int(entry_id),))


def _entry_from_row(row) -> PersonEntry:
    try:
        measurement = json.loads(row[10])
    except ValueError:
        measurement = {}
    return PersonEntry(row[0], row[1], row[2], row[3], row[4], row[5], bool(row[6]), row[7], row[8], row[9],
                       measurement if isinstance(measurement, dict) else {})


def summarize(entries) -> DatabaseSummary:
    """How far the camera is from the truth over all records that have an estimate."""
    labels = {entry.label for entry in entries if entry.label}
    people = len(labels) + sum(1 for entry in entries if not entry.label)
    weights = [(entry.weight_error_kg, entry.real_weight_kg) for entry in entries if entry.weight_error_kg is not None]
    heights = [entry.height_error_cm for entry in entries if entry.height_error_cm is not None]
    count = len(weights)
    return DatabaseSummary(
        len(entries), people, count, sum(1 for entry in entries if entry.weight_from_scales),
        sum(error for error, _ in weights) / count if count else None,
        sum(abs(error) for error, _ in weights) / count if count else None,
        sum(abs(error) / real * 100 for error, real in weights) / count if count else None,
        sum(abs(error) for error in heights) / len(heights) if heights else None)


def summary_text(summary: DatabaseSummary) -> str:
    if not summary.records:
        return "База пока пуста. Сделайте замер и введите настоящие рост и вес человека."
    parts = [f"Записей: {summary.records}, людей: {summary.people}, с весов: {summary.from_scales}."]
    if summary.compared:
        parts.append(f"Средняя ошибка веса {summary.mean_absolute_error_kg:.1f} кг "
                     f"({summary.mean_absolute_error_percent:.1f}%), смещение {summary.mean_error_kg:+.1f} кг.")
    if summary.height_absolute_error_cm is not None:
        parts.append(f"Средняя ошибка роста {summary.height_absolute_error_cm:.1f} см.")
    if summary.people < ENOUGH_PEOPLE:
        parts.append(f"Людей пока мало: для честной оценки нужно хотя бы {ENOUGH_PEOPLE}.")
    return " ".join(parts)


def safe_cell(text: str) -> str:
    """Keep a typed label from running as a formula when the file is opened in a spreadsheet."""
    return "'" + text if text.startswith(SPREADSHEET_FORMULA_START) else text


def _rounded(value, digits=4):
    return round(value, digits) if isinstance(value, (int, float)) and not isinstance(value, bool) else ""


def export_csv(entries, path) -> int:
    """One row per record with every measured number, for analysis or fitting a model."""
    fieldnames = ["id", "created", "label", "sex", "real_height_cm", "real_weight_kg", "weight_from_scales",
                  "clothing", "camera", "app_version", "samples", "side_observed", "height_estimate_cm",
                  "weight_estimate_kg", "weight_low_kg", "weight_high_kg", "volume_l", "weight_error_kg",
                  "height_error_cm"]
    for kind in ("front_width_m", "side_width_m", "length_m"):
        fieldnames += [f"{part}_{kind}" for part in MEASURED_PARTS]
    fieldnames += [f"slice_{index:02d}_front_m" for index in range(SLICES)]
    fieldnames += [f"slice_{index:02d}_side_m" for index in range(SLICES)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Semicolons and a byte order mark are what Excel expects on a Russian Windows.
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames, delimiter=";")
        writer.writeheader()
        for entry in entries:
            measured = entry.measurement
            row = dict(id=entry.id, created=entry.created, label=safe_cell(entry.label), sex=entry.sex,
                       real_height_cm=entry.real_height_cm, real_weight_kg=entry.real_weight_kg,
                       weight_from_scales=int(entry.weight_from_scales), clothing=entry.clothing,
                       camera=entry.camera, app_version=entry.app_version, samples=measured.get("samples", ""),
                       side_observed=int(bool(measured.get("side_observed"))),
                       height_estimate_cm=_rounded(entry.estimated_height_cm, 2),
                       weight_estimate_kg=_rounded(entry.estimated_weight_kg, 2),
                       weight_low_kg=_rounded(measured.get("mass_low_kg"), 2),
                       weight_high_kg=_rounded(measured.get("mass_high_kg"), 2),
                       volume_l=_rounded(measured.get("volume_l"), 2),
                       weight_error_kg=_rounded(entry.weight_error_kg, 2),
                       height_error_cm=_rounded(entry.height_error_cm, 2))
            for kind, key in (("front_width_m", "widths_m"), ("side_width_m", "side_widths_m"), ("length_m", "lengths_m")):
                values = measured.get(key) or {}
                row.update({f"{part}_{kind}": _rounded(values.get(part)) for part in MEASURED_PARTS})
            for side, key in (("front", "slices_front_m"), ("side", "slices_side_m")):
                values = measured.get(key) or []
                row.update({f"slice_{index:02d}_{side}_m": _rounded(values[index]) if index < len(values) else ""
                            for index in range(SLICES)})
            writer.writerow(row)
    return len(entries)
