#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import merge_dvwa_vulhab


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge DVWA and Vulhab structured logs into one training-ready dataset."
    )
    parser.add_argument(
        "--dvwa-structured",
        default="output/dvwa/DVWA.log_structured.csv",
        help="Path to DVWA structured csv.",
    )
    parser.add_argument(
        "--vulhab-structured",
        default="output/vulhab/Vulhab.log_structured.csv",
        help="Path to Vulhab structured csv.",
    )
    parser.add_argument("--output-dir", default="output/dvwa_vulhab", help="Output directory.")
    parser.add_argument(
        "--dataset-title",
        default="DVWA_Vulhab",
        help="Output file prefix (<title>.log_structured.csv).",
    )
    args = parser.parse_args()

    stats = merge_dvwa_vulhab(
        dvwa_structured_path=Path(args.dvwa_structured),
        vulhab_structured_path=Path(args.vulhab_structured),
        output_root=Path(args.output_dir),
        dataset_title_name=args.dataset_title,
    )
    print(f"DVWA rows: {stats['dvwa_rows']}")
    print(f"Vulhab rows: {stats['vulhab_rows']}")
    print(f"Combined rows: {stats['combined_rows']}")
    print(f"Templates: {stats['templates_count']}")
    print(f"Saved: {stats['structured_path']}")
    print(f"Saved: {stats['templates_path']}")


if __name__ == "__main__":
    main()
