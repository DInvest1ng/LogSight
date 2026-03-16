#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare raw logs in LogPAI-like format for LogBERT fine-tuning."
    )
    parser.add_argument("--target", default="vulhab", help="Dataset name (e.g. vulhab, dvwa).")
    parser.add_argument("--logs-dir", default="logs", help="Directory with raw logs.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated CSV files.")
    args = parser.parse_args()

    stats = prepare_dataset(
        target=args.target,
        logs_dir=Path(args.logs_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"Target: {stats['target']}")
    print(f"Found files: {stats['files_count']}")
    print(f"Loaded records: {stats['records_count']}")
    print(f"Saved: {stats['structured_path']}")
    print(f"Saved: {stats['templates_path']}")


if __name__ == "__main__":
    main()
