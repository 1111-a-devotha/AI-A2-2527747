"""Command-line prediction for one farm record (works offline).

Usage:
    python predict.py --json samples/valid.json
    python predict.py --json '{"rainfall_mm": 1000, ...}'

Exit codes: 0 = success, 2 = invalid input, 3 = models missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src import config
from src.inference import ModelsNotAvailableError, predict_record
from src.schema import FarmRecord


def load_json_argument(value: str) -> dict:
    """Read a JSON object from a file path or from an inline JSON string."""
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.exists() else value
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("The JSON input must be one object (one farm record), not a list.")
    return data


def format_errors(exc: ValidationError) -> list[dict]:
    """Turn pydantic errors into short, readable field messages."""
    return [{"field": ".".join(str(p) for p in e["loc"]) or "record",
             "problem": e["msg"], "received": None if e["type"] == "missing" else e.get("input")} for e in exc.errors()]


def main(argv: list[str] | None = None) -> int:
    """Validate the record, run the saved models and print the JSON result."""
    parser = argparse.ArgumentParser(description=f"Predict harvest outcome ({config.VERIFICATION_CODE})")
    parser.add_argument("--json", required=True, help="Path to a JSON file or an inline JSON object")
    args = parser.parse_args(argv)

    try:
        payload = load_json_argument(args.json)
        record = FarmRecord.model_validate(payload)
    except json.JSONDecodeError as exc:
        print(json.dumps({"verification_code": config.VERIFICATION_CODE, "status": "invalid_input",
                          "errors": [{"field": "record", "problem": f"Malformed JSON: {exc}"}]},
                         indent=2), file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(json.dumps({"verification_code": config.VERIFICATION_CODE, "status": "invalid_input",
                          "errors": format_errors(exc)}, indent=2, default=str), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(json.dumps({"verification_code": config.VERIFICATION_CODE, "status": "invalid_input",
                          "errors": [{"field": "record", "problem": str(exc)}]}, indent=2),
              file=sys.stderr)
        return 2

    try:
        result = predict_record(record)
    except ModelsNotAvailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({"status": "ok", **result.model_dump()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
