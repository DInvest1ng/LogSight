from __future__ import annotations

from pathlib import Path
from typing import Any

from .builders import (
    build_structured_rows,
    build_templates_table,
    load_structured_rows,
    save_merged_outputs,
    save_structured_outputs,
)
from .constants import DATASET_TITLE_OVERRIDES
from .loaders import collect_log_files, load_records
from .template import finalize_training_template, make_event_id, normalize_to_template, training_pre_normalize


def dataset_title(target: str) -> str:
    target_lc = target.lower()
    return DATASET_TITLE_OVERRIDES.get(target_lc, target_lc[:1].upper() + target_lc[1:])


def prepare_dataset(
    target: str = "vulhab",
    logs_dir: Path = Path("logs"),
    output_dir: Path = Path("output"),
) -> dict[str, Any]:
    target_lc = target.lower()
    title = dataset_title(target_lc)
    output_root = output_dir / target_lc

    log_files = collect_log_files(logs_dir=logs_dir, target=target_lc)
    if not log_files:
        raise FileNotFoundError(f"No log files found in {logs_dir}/ for target '{target_lc}'.")

    records = load_records(log_files)
    structured_rows = build_structured_rows(records)
    templates_rows = build_templates_table(structured_rows)
    save_structured_outputs(structured_rows, templates_rows, output_root=output_root, dataset_title=title)

    return {
        "target": target_lc,
        "dataset_title": title,
        "files_count": len(log_files),
        "records_count": len(records),
        "structured_path": str(output_root / f"{title}.log_structured.csv"),
        "templates_path": str(output_root / f"{title}.log_templates.csv"),
    }


def merge_dvwa_vulhab(
    dvwa_structured_path: Path = Path("output/dvwa/DVWA.log_structured.csv"),
    vulhab_structured_path: Path = Path("output/vulhab/Vulhab.log_structured.csv"),
    output_root: Path = Path("output/dvwa_vulhab"),
    dataset_title_name: str = "DVWA_Vulhab",
) -> dict[str, Any]:
    if not dvwa_structured_path.exists():
        raise FileNotFoundError(f"Missing file: {dvwa_structured_path}")
    if not vulhab_structured_path.exists():
        raise FileNotFoundError(f"Missing file: {vulhab_structured_path}")

    dvwa_rows = load_structured_rows(dvwa_structured_path)
    vulhab_rows = load_structured_rows(vulhab_structured_path)
    combined_rows = _build_combined_rows(dvwa_rows, vulhab_rows)
    templates_rows = build_templates_table(combined_rows)
    save_merged_outputs(combined_rows, templates_rows, output_root=output_root, dataset_title=dataset_title_name)

    return {
        "dvwa_rows": len(dvwa_rows),
        "vulhab_rows": len(vulhab_rows),
        "combined_rows": len(combined_rows),
        "templates_count": len(templates_rows),
        "structured_path": str(output_root / f"{dataset_title_name}.log_structured.csv"),
        "templates_path": str(output_root / f"{dataset_title_name}.log_templates.csv"),
    }


def _build_combined_rows(*dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for rows in dataset_rows:
        all_rows.extend(rows)

    combined: list[dict[str, Any]] = []
    for line_id, row in enumerate(all_rows, start=1):
        content = (row.get("Content") or "-").strip()
        normalized_input = training_pre_normalize(content)
        event_template = normalize_to_template(normalized_input)
        event_template = finalize_training_template(event_template)
        event_id = make_event_id(event_template)

        combined_row = {
            "LineId": line_id,
            "Label": row.get("Label", "0"),
            "Id": row.get("Id", ""),
            "Date": row.get("Date", "-") or "-",
            "Admin": row.get("Admin", "-") or "-",
            "Month": row.get("Month", "-") or "-",
            "Day": row.get("Day", "-") or "-",
            "Time": row.get("Time", "-") or "-",
            "AdminAddr": row.get("AdminAddr", "-") or "-",
            "Content": content or "-",
            "EventId": event_id,
            "EventTemplate": event_template,
        }
        combined.append(combined_row)
    return combined
