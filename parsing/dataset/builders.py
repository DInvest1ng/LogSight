from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import STRUCTURED_COLUMNS, TEMPLATE_COLUMNS
from .line_parser import extract_content, parse_raw_line
from .template import make_event_id, normalize_to_template


def build_structured_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_id, record in enumerate(records, start=1):
        raw_line = str(record.get("raw", "")).strip()
        parsed = parse_raw_line(raw_line)
        content = extract_content(parsed)
        event_template = normalize_to_template(content)
        event_id = make_event_id(event_template)

        row = {
            "LineId": line_id,
            "Label": record.get("label", 0),
            "Id": hashlib.md5(raw_line.encode("utf-8")).hexdigest(),
            "Date": parsed.get("Date", "-") or "-",
            "Admin": parsed.get("Admin", "-") or "-",
            "Month": parsed.get("Month", "-") or "-",
            "Day": parsed.get("Day", "-") or "-",
            "Time": parsed.get("Time", "-") or "-",
            "AdminAddr": parsed.get("AdminAddr", "-") or "-",
            "Content": content if content else "-",
            "EventId": event_id,
            "EventTemplate": event_template,
        }
        rows.append(row)
    return rows


def build_templates_table(structured_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["EventTemplate"] for row in structured_rows)
    table = [
        {
            "EventId": make_event_id(template),
            "EventTemplate": template,
            "Occurrences": occurrences,
        }
        for template, occurrences in counts.items()
    ]
    table.sort(key=lambda item: (-item["Occurrences"], item["EventId"]))
    return table


def save_structured_outputs(
    structured_rows: list[dict[str, Any]],
    templates_rows: list[dict[str, Any]],
    output_root: Path,
    dataset_title: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    structured_path = output_root / f"{dataset_title}.log_structured.csv"
    templates_path = output_root / f"{dataset_title}.log_templates.csv"

    with structured_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=STRUCTURED_COLUMNS)
        writer.writeheader()
        writer.writerows(structured_rows)

    with templates_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(templates_rows)


def load_structured_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def save_merged_outputs(
    structured_rows: list[dict[str, Any]],
    templates_rows: list[dict[str, Any]],
    output_root: Path,
    dataset_title: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    structured_path = output_root / f"{dataset_title}.log_structured.csv"
    templates_path = output_root / f"{dataset_title}.log_templates.csv"

    with structured_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=STRUCTURED_COLUMNS)
        writer.writeheader()
        writer.writerows(structured_rows)

    with templates_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(templates_rows)
