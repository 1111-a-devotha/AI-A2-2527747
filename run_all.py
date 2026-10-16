"""Reproducible entry point: recreate data, clean, train all tasks, regenerate artifacts.

Usage:
    python run_all.py --student-id ug2527747@ines.ac.rw --seed 7747
"""
from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.pipeline import artifact_list, run
from src.validation import DataValidationError


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline and print the evidence summary."""
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service - full pipeline")
    parser.add_argument("--student-id", required=True,
                        help=f"Institutional username, e.g. {config.INSTITUTIONAL_USERNAME}")
    parser.add_argument("--seed", type=int, required=True,
                        help=f"Personal seed = last four registration digits ({config.LLLL})")
    args = parser.parse_args(argv)

    print("=" * 66)
    print(f" Musanze Harvest Intelligence Service   {config.VERIFICATION_CODE}")
    print(f" {config.STUDENT_NAME} | {config.REGISTRATION_NUMBER} | {args.student_id}")
    print(f" Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 66)
    try:
        s = run(args.student_id, args.seed)
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Seed                : {s['seed']}")
    print(f"Base rows           : {s['base_rows']}  (240 + {s['seed']} mod 41)")
    print(f"Rows in CSV         : {s['rows_in_csv']}  (incl. 3 exact duplicates)")
    print(f"Missing values      : {s['missing_values']}")
    print(f"Duplicates removed  : {s['duplicates_removed']}")
    print(f"Rows after cleaning : {s['rows_after_cleaning']}  (train {s['train_rows']} / test {s['test_rows']})")
    print(f"Dataset SHA-256     : {s['sha256']}")
    r, c, k = s["regression"], s["classification"], s["clustering"]
    print(f"Regression          : {r['selected']}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  R2={r['r2']:.4f}")
    print(f"Classification      : {c['selected']}  P={c['precision']:.4f}  R={c['recall']:.4f}  "
          f"F1={c['f1']:.4f}  AUC={c['roc_auc']:.4f}")
    print(f"Clustering          : K={k['selected_k']}  silhouette={k['silhouette']:.4f}")
    print("-" * 66)
    print("Generated artifacts:")
    for path in artifact_list():
        print(f"  {path}")
    print("-" * 66)
    print(f"Done. Verification code: {config.VERIFICATION_CODE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
