# scripts/logsight.py


import os
import sys
import argparse
from pathlib import Path
from typing import Optional


def setup_project_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)


setup_project_path()

import asyncio
from scripts.log_analyzer import log_responce


async def main():
    parser = argparse.ArgumentParser(description="Log analyzer using LLM")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the log file for analysis",
    )
    args = parser.parse_args()

    result = await log_responce(args.file)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
