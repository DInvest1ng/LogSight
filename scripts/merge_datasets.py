#!/usr/bin/env python3
import os
import sys


def setup_project_path() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)


setup_project_path()

from parsing.dataset.cli_merge import main


if __name__ == "__main__":
    main()
