from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .constants import ALLOWED_SUFFIXES, LABEL_KEYS, RAW_TEXT_KEYS, TARGET_ALIASES


def collect_log_files(logs_dir: Path = Path("logs"), target: str = "vulhab") -> list[Path]:
    target_lc = target.lower()
    aliases = set(TARGET_ALIASES.get(target_lc, {target_lc}))

    files: list[Path] = []
    for path in logs_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        name_lc = path.name.lower()
        if any(alias in name_lc for alias in aliases):
            files.append(path)
    return sorted(files)


def load_records(files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            records.extend(_load_jsonl(path))
        elif suffix == ".json":
            records.extend(_load_json(path))
        elif suffix == ".csv":
            records.extend(_load_csv(path))
        else:
            records.extend(_load_txt_like(path))
    return records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                item = raw
            normalized = _normalize_record(item)
            if normalized is not None:
                out.append(normalized)
    return out


def _load_json(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        iterable = data
    elif isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            iterable = data["records"]
        elif "data" in data and isinstance(data["data"], list):
            iterable = data["data"]
        else:
            iterable = [data]
    else:
        iterable = [data]

    for item in iterable:
        normalized = _normalize_record(item)
        if normalized is not None:
            out.append(normalized)
    return out


def _load_csv(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames:
            for row in reader:
                normalized = _normalize_record(row)
                if normalized is not None:
                    out.append(normalized)
            return out

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            normalized = _normalize_record(row)
            if normalized is not None:
                out.append(normalized)
    return out


def _load_txt_like(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                item = raw
            normalized = _normalize_record(item)
            if normalized is not None:
                out.append(normalized)
    return out


def _normalize_record(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None

    raw_value: Any = ""
    label_value: Any = 0

    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return None
        raw_value = item[0]
        if len(item) > 1:
            label_value = item[1]
    elif isinstance(item, dict):
        lower_map = {str(key).lower(): key for key in item.keys()}

        for key in RAW_TEXT_KEYS:
            if key in lower_map:
                raw_value = item[lower_map[key]]
                break
        else:
            first_text = next((value for value in item.values() if isinstance(value, str)), None)
            if first_text is not None:
                raw_value = first_text
            else:
                raw_value = json.dumps(item, ensure_ascii=False, sort_keys=True)

        for key in LABEL_KEYS:
            if key in lower_map:
                label_value = item[lower_map[key]]
                break
    elif isinstance(item, str):
        raw_value = item
    else:
        raw_value = json.dumps(item, ensure_ascii=False, sort_keys=True)

    if not isinstance(raw_value, str):
        raw_value = json.dumps(raw_value, ensure_ascii=False, sort_keys=True)

    raw_value = raw_value.strip()
    if not raw_value:
        return None

    return {"raw": raw_value, "label": _coerce_label(label_value)}


def _coerce_label(value: Any) -> Any:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "anomaly", "anomalous", "attack", "malicious"}:
        return 1
    if text in {"0", "false", "no", "normal", "benign"}:
        return 0
    return str(value).strip()
