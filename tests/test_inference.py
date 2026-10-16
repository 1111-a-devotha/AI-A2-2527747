"""Inference and input-validation tests (CLI + shared schema)."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from src import config
from src.inference import predict_record
from src.schema import FarmRecord
from tests.conftest import ROOT


def test_valid_inference_returns_all_fields(valid_record):
    """A valid record gives yield, risk, probability, cluster and the verification code."""
    result = predict_record(FarmRecord(**valid_record))
    assert result.verification_code == config.VERIFICATION_CODE
    assert result.yield_estimate_tons > 0
    assert result.high_risk_prediction in (0, 1)
    assert 0.0 <= result.high_risk_probability <= 1.0
    assert 0 <= result.cluster < 6


def test_prediction_is_deterministic(valid_record):
    """The same record always gives the same answer."""
    a = predict_record(FarmRecord(**valid_record))
    b = predict_record(FarmRecord(**valid_record))
    assert a == b


def test_missing_field_is_rejected(valid_record):
    del valid_record["soil_moisture_pct"]
    with pytest.raises(ValidationError) as exc:
        FarmRecord(**valid_record)
    assert exc.value.errors()[0]["type"] == "missing"
    assert exc.value.errors()[0]["loc"] == ("soil_moisture_pct",)


@pytest.mark.parametrize("field,value", [
    ("rainfall_mm", "a lot"),        # text instead of number
    ("avg_temp_c", "18"),            # numeric string is not silently converted
    ("cooperative_support", True),   # boolean is not accepted as 1
    ("season", 1),                   # number instead of A/B
])
def test_wrong_type_is_rejected(valid_record, field, value):
    valid_record[field] = value
    with pytest.raises(ValidationError):
        FarmRecord(**valid_record)


@pytest.mark.parametrize("field,value", [
    ("rainfall_mm", 1400.1), ("rainfall_mm", 599.9), ("avg_temp_c", 30.0),
    ("soil_moisture_pct", 10.0), ("fertilizer_kg", -1.0), ("field_size_ha", 0.0),
    ("season", "C"), ("cooperative_support", 2),
])
def test_out_of_range_is_rejected(valid_record, field, value):
    valid_record[field] = value
    with pytest.raises(ValidationError):
        FarmRecord(**valid_record)


def test_target_column_in_input_is_rejected(valid_record):
    """Supplying the target as an input is refused (leakage guard at inference)."""
    valid_record["yield_tons"] = 10.0
    with pytest.raises(ValidationError):
        FarmRecord(**valid_record)


def test_cli_valid_and_invalid():
    """predict.py exits 0 on the valid sample and 2 on the invalid sample."""
    ok = subprocess.run([sys.executable, "predict.py", "--json", "samples/valid.json"],
                        cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    out = json.loads(ok.stdout)
    assert out["verification_code"] == config.VERIFICATION_CODE
    bad = subprocess.run([sys.executable, "predict.py", "--json", "samples/invalid.json"],
                         cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode == 2
    fields = {e["field"] for e in json.loads(bad.stderr)["errors"]}
    assert {"rainfall_mm", "avg_temp_c", "soil_moisture_pct", "season"} <= fields
