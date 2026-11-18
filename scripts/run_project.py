"""Run the declared numerical study, or reuse verified frozen outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.workflow import run_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true", help="Repeat the same frozen plan; this is not a new independent test.")
    arguments = parser.parse_args()
    run_experiments(reuse=not arguments.reproduce)
