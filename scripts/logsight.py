import argparse
import asyncio
import os
import sys


def setup_project_path() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)


setup_project_path()

from clients.log_analyzer import log_response


async def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze logs with YandexGPT")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the log file for analysis",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML prompt config (default: configs/log_analysis_prompt.yaml)",
    )
    args = parser.parse_args()

    result = await log_response(args.file, config_file=args.config)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
