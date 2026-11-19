"""Local database of pigs weighed on scales, with what the camera measured of the same animal.

Each record pairs the estimate at the scale the pictures themselves gave with the weight on
scales. The pictures of one barn and one camera are wrong in much the same way every time, so
the records give a correction for the next videos taken there: the cube root of the median ratio
of true to estimated weight, applied to every size. It stays on this computer; no images are kept.
"""
from contextlib import closing
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
from typing import NamedTuple

from .people_database import SPREADSHEET_FORMULA_START, finite_only

FOLDER = "pigs"
DATABASE_FILE = "pigs.sqlite"
WEIGHT_RANGE_KG = (.3, 400.)
MAX_LABEL_LENGTH = 40
# Below this many weighed pigs the correction rests on too little to trust its spread.
ENOUGH_PIGS = 3
# A correction outside these limits means a record of another camera or a typing slip.
FACTOR_LIMITS = (.2, 5.)

_SCHEMA = """CREATE TABLE IF NOT EXISTS pigs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    label TEXT NOT NULL,
    real_weight_kg REAL NOT NULL,
    scale TEXT NOT NULL,
    source TEXT NOT NULL,
    app_version TEXT NOT NULL,
    measurement TEXT NOT NULL)"""
_COLUMNS = "id, created, label, real_weight_kg, scale, source, app_version, measurement"


@dataclass(frozen=True)
class PigEntry:
    id: int | None
    created: str
    label: str
    real_weight_kg: float
    # Where the scale of the estimate came from: the picture alone or a stereo pair.
    scale: str
    source: str
    app_version: str
    measurement: dict

    @property
    def estimated_weight_kg(self) -> float | None:
        return self.measurement.get("mass_kg")

    @property
    def length_factor(self) -> float | None:
        """How many times every size must grow for the estimate to match the scales."""
        estimate = self.estimated_weight_kg
        if not estimate or estimate <= 0:
            return None
        return (self.real_weight_kg / estimate) ** (1 / 3)


class Correction(NamedTuple):
    factor: float
    count: int
    spread: float


def parse_weight(text) -> float:
    try:
        value = float(str(text).strip().replace(",", "."))
    except ValueError:
        raise ValueError("Вес: введите число.") from None
    if not math.isfinite(value) or not WEIGHT_RANGE_KG[0] <= value <= WEIGHT_RANGE_KG[1]:
        raise ValueError(f"Вес: допустимо от {WEIGHT_RANGE_KG[0]:g} до {WEIGHT_RANGE_KG[1]:g} кг.")
    return value


def measurement_of(estimate) -> dict:
    return finite_only(dict(mass_kg=estimate.mass_kg, mass_low_kg=estimate.mass_low_kg,
                            mass_high_kg=estimate.mass_high_kg, length_m=estimate.length_m, girth_m=estimate.girth_m,
                            volume_l=estimate.volume_l, samples=estimate.samples))


def make_entry(row, weight_text, label, source, app_version) -> PigEntry:
    """A record of one pig from the tab's table. Raises ValueError with a message meant for the farmer."""
    if row is None or row.raw is None:
        raise ValueError("Эта свинья ещё не измерена: выберите номер, у которого в таблице есть вес.")
    label = " ".join(str(label).split())[:MAX_LABEL_LENGTH]
    return PigEntry(None, datetime.now().isoformat(timespec="seconds"), label, parse_weight(weight_text),
                    row.raw_source, source or "", app_version, measurement_of(row.raw))


class PigDatabase:
    """One short connection per call: only the interface thread uses it, and rarely."""

    def __init__(self, root):
        self.path = Path(root) / "data" / FOLDER / DATABASE_FILE

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        with connection:
            connection.execute(_SCHEMA)
        return connection

    def add(self, entry: PigEntry) -> int:
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO pigs (created, label, real_weight_kg, scale, source, app_version, measurement) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entry.created, entry.label, entry.real_weight_kg, entry.scale, entry.source, entry.app_version,
                 json.dumps(entry.measurement, ensure_ascii=False, allow_nan=False)))
            return int(cursor.lastrowid)

    def entries(self) -> list:
        """Newest first. No file yet simply means no records."""
        if not self.path.exists():
            return []
        with closing(self.connect()) as connection:
            rows = connection.execute(f"SELECT {_COLUMNS} FROM pigs ORDER BY id DESC").fetchall()
        result = []
        for row in rows:
            try:
                measurement = json.loads(row[7])
            except ValueError:
                measurement = {}
            result.append(PigEntry(*row[:7], measurement if isinstance(measurement, dict) else {}))
        return result

    def delete(self, entry_id: int) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM pigs WHERE id = ?", (int(entry_id),))


def correction(entries, scale) -> Correction | None:
    """The correction for estimates whose scale came from `scale`, or None without records."""
    factors = sorted(factor for factor in (entry.length_factor for entry in entries if entry.scale == scale)
                     if factor is not None and FACTOR_LIMITS[0] <= factor <= FACTOR_LIMITS[1])
    if not factors:
        return None
    middle = factors[len(factors) // 2] if len(factors) % 2 else (factors[len(factors) // 2 - 1] + factors[len(factors) // 2]) / 2
    spread = (factors[-1] - factors[0]) / (2 * middle) if len(factors) > 1 else 0.
    return Correction(middle, len(factors), spread)


def summary_text(entries, scale) -> str:
    """What the database holds and how far the camera was from the scales, for the tab."""
    if not entries:
        return "В базе пока нет свиней. Взвесьте одну на весах, введите её номер и вес и нажмите «Добавить в базу»."
    errors = [abs(entry.estimated_weight_kg - entry.real_weight_kg) / entry.real_weight_kg * 100
              for entry in entries if entry.estimated_weight_kg]
    parts = [f"В базе свиней: {len(entries)}."]
    found = correction(entries, scale)
    if found is not None:
        parts.append(f"Без поправки камера ошибалась в среднем на {sum(errors) / len(errors):.0f} %; "
                     f"поправка размеров ×{found.factor:.2f}".replace(".", ",") +
                     (f", разброс ±{found.spread * 100:.0f} %." if found.count > 1 else "."))
        if found.count < ENOUGH_PIGS:
            parts.append(f"Для надёжной поправки взвесьте хотя бы {ENOUGH_PIGS} свиньи.")
    return " ".join(parts)


def export_csv(entries, path) -> int:
    """One row per pig, semicolons and a byte order mark as Excel expects on a Russian Windows."""
    fieldnames = ["id", "created", "label", "real_weight_kg", "scale", "source", "app_version", "samples",
                  "weight_estimate_kg", "volume_l", "length_m", "girth_m", "length_factor"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames, delimiter=";")
        writer.writeheader()
        for entry in entries:
            measured = entry.measurement
            label = "'" + entry.label if entry.label.startswith(SPREADSHEET_FORMULA_START) else entry.label
            writer.writerow(dict(id=entry.id, created=entry.created, label=label, real_weight_kg=entry.real_weight_kg,
                                 scale=entry.scale, source=entry.source, app_version=entry.app_version,
                                 samples=measured.get("samples", ""), weight_estimate_kg=measured.get("mass_kg", ""),
                                 volume_l=measured.get("volume_l", ""), length_m=measured.get("length_m", ""),
                                 girth_m=measured.get("girth_m", ""),
                                 length_factor="" if entry.length_factor is None else round(entry.length_factor, 4)))
    return len(entries)
