"""Recreate the individual dataset data/AI_A2_2527747.csv (AI-A2-7747).

Usage:
    python data/generate_data.py --seed 7747
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data_generation import write_dataset  # noqa: E402
from src.validation import DataValidationError, validate_seed  # noqa: E402


def main() -> int:
    """Generate the CSV and fingerprint for the given seed."""
    parser = argparse.ArgumentParser(description="Generate the individual synthetic dataset.")
    parser.add_argument("--seed", type=int, default=config.LLLL,
                        help=f"Personal seed (must be {config.LLLL}).")
    args = parser.parse_args()
    try:
        validate_seed(args.seed)
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    summary = write_dataset(args.seed)
    print(f"Verification code : {config.VERIFICATION_CODE}")
    print(f"Base rows         : {summary['base_rows']}")
    print(f"Rows in CSV       : {summary['total_rows_in_csv']}")
    print(f"CSV               : data/{summary['csv_file']}")
    print(f"SHA-256           : {summary['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
