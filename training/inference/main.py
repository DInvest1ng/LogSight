from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable, List

from .predictor import LogBERTPredictor, LogBERTPredictorConfig


LOGGER = logging.getLogger(__name__)


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        head = handle.readline()
        handle.seek(0)
        if "Content" in head and "Date" in head and "Time" in head:
            reader = csv.DictReader(handle)
            rows: List[str] = []
            for row in reader:
                content = (row.get("Content") or "").strip()
                date = (row.get("Date") or "").strip()
                time = (row.get("Time") or "").strip()
                if not content:
                    continue
                if date and time:
                    rows.append(f"{date} {time} {content}")
                else:
                    rows.append(content)
            return rows
        return [line.strip() for line in handle if line.strip()]


def _write_results(results: Iterable[dict], output: Path | None) -> None:
    payload = list(results)
    if output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LogBERT predictions on a log file.")
    parser.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Path to a text/CSV log file with one log line per row.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("weights/best_bert.pth"),
        help="Path to LogBERT checkpoint.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("weights/vocab.pkl"),
        help="Path to vocab.pkl from training.",
    )
    parser.add_argument(
        "--center",
        type=Path,
        default=Path("weights/best_center.pt"),
        help="Optional path to center checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--group-by-minute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group contiguous log lines by minute before inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for predictions.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()

    logs = _read_lines(args.logs)
    if not logs:
        raise SystemExit(f"No logs found in {args.logs}")

    config = LogBERTPredictorConfig(device=args.device)
    center_path = args.center if args.center and args.center.exists() else None

    predictor = LogBERTPredictor(
        state_path=args.state,
        vocab_path=args.vocab,
        center_path=center_path,
        config=config,
    )

    results = predictor.predict(logs, group_by_minute=args.group_by_minute)
    _write_results(results, args.output)
    LOGGER.info("Predicted %d log lines", len(results))


if __name__ == "__main__":
    main()
