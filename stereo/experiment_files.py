"""The result.json of one weighing experiment: the camera's numbers, the setup, and the real data typed later.

Everything needed to judge the error of a measurement stays in its own folder, next to the numbers.
"""
from datetime import datetime
import json
import math
import os
from pathlib import Path

from .version import VERSION

RESULT_FILE = "result.json"
UNITS = "metres, litres, kilograms"


def _finite(value):
    """JSON has no NaN: an unknown number is stored as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def experiment_setup(config, calibration) -> dict:
    """Camera mode and calibration quality at the moment of the measurement; missing parts are left out."""
    setup = {"saved": datetime.now().isoformat(timespec="seconds")}
    mode = getattr(config, "mode", None)
    if mode:
        setup["camera"] = str(mode)
    if calibration is not None:
        values = {"rms_px": getattr(calibration, "rms", None),
                  "vertical_error_px": getattr(calibration, "epipolar_error", None)}
        try:
            values["baseline_m"] = calibration.baseline
        except (AttributeError, TypeError, ValueError):
            pass
        size = getattr(calibration, "size", None)
        if size is not None and len(size) == 2:
            values["image_size_px"] = [int(size[0]), int(size[1])]
        values = {key: value if key == "image_size_px" else _number(value) for key, value in values.items()}
        setup["calibration"] = {key: value for key, value in values.items() if value is not None}
    return setup


def write_experiment(path, people, setup=None) -> int:
    """A new result.json for the people measured; returns how many were written."""
    data = dict(units=UNITS, validated_against_scales=False, app_version=VERSION,
                setup=setup or {}, people=list(people))
    _write(Path(path), data)
    return len(data["people"])


def attach_real_data(path, person_label, entry) -> None:
    """Store what was typed for one person next to the camera's numbers for them, with the errors.

    Raises OSError or ValueError when the file cannot be read or has no such person.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    people = data.get("people") if isinstance(data, dict) else None
    if not isinstance(people, list):
        raise ValueError("В файле замера нет списка людей.")
    index = next((i for i, person in enumerate(people) if isinstance(person, dict)
                  and str(person.get("label")) == str(person_label)), None)
    if index is None:
        raise ValueError(f"В файле замера нет человека №{person_label}.")
    real = dict(entered=entry.created, label=entry.label, sex=entry.sex, height_cm=entry.real_height_cm,
                weight_kg=entry.real_weight_kg, weight_from_scales=entry.weight_from_scales,
                clothing=entry.clothing, camera=entry.camera, app_version=entry.app_version,
                estimated_height_cm=entry.estimated_height_cm, estimated_weight_kg=entry.estimated_weight_kg,
                height_error_cm=entry.height_error_cm, weight_error_kg=entry.weight_error_kg)
    if real["weight_error_kg"] is not None and entry.real_weight_kg:
        real["weight_error_percent"] = real["weight_error_kg"] / entry.real_weight_kg * 100
    updated_people = [dict(person, real=real) if i == index else person for i, person in enumerate(people)]
    validated = any(isinstance(person, dict) and (person.get("real") or {}).get("weight_from_scales")
                    for person in updated_people)
    _write(path, dict(data, people=updated_people, validated_against_scales=validated))


def _write(path: Path, data: dict) -> None:
    """Write through a temporary file, so a failure never leaves half a result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_finite(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
