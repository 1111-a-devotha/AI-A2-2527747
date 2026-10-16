"""Reproducibility, data validation and leakage tests."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config
from src.data_generation import dataframe_to_csv_bytes, generate_dataframe, sha256_bytes
from src.validation import DataValidationError, validate_and_clean, validate_student_id


def test_dataset_is_reproducible_and_matches_fingerprint():
    """Two generations give identical bytes, matching the saved fingerprint and CSV."""
    first = dataframe_to_csv_bytes(generate_dataframe(config.LLLL))
    second = dataframe_to_csv_bytes(generate_dataframe(config.LLLL))
    assert first == second
    digest = sha256_bytes(first)
    assert f"sha256: {digest}" in config.FINGERPRINT_TXT.read_text(encoding="utf-8")
    assert config.DATA_CSV.read_bytes() == first


def test_individual_data_signature():
    """279 base rows, 14 missing per column, 3 duplicates, all from seed 7747."""
    df = generate_dataframe(config.LLLL)
    assert config.base_row_count(config.LLLL) == 279
    assert len(df) == 279 + 3
    assert df["soil_moisture_pct"].isna().sum() == 14
    assert df["fertilizer_kg"].isna().sum() == 14
    assert df.duplicated().sum() == 3
    assert df["yield_tons"].min() >= 0.5


def test_cleaning_removes_duplicates_and_keeps_missing_for_pipeline():
    raw = pd.read_csv(config.DATA_CSV)
    clean, report = validate_and_clean(raw)
    assert report["exact_duplicate_rows_found"] == 3
    assert len(clean) == 279
    assert clean.duplicated(subset=config.REQUIRED_COLUMNS).sum() == 0
    assert all(v == 0 for v in report["range_violations"].values())
    assert report["leakage"]["status"] == "passed"


@pytest.mark.parametrize("student_id,seed", [
    ("ug2527747@ines.ac.rw", 4671),   # wrong seed
    ("ug2424671@ines.ac.rw", 4671),   # somebody else's ID
    ("no-digits@ines.ac.rw", 7747),
])
def test_wrong_identity_or_seed_is_rejected(student_id, seed):
    with pytest.raises(DataValidationError):
        validate_student_id(student_id, seed)


def test_missing_column_is_rejected():
    raw = pd.read_csv(config.DATA_CSV).drop(columns=["season"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_and_clean(raw)


def test_no_target_leakage_in_saved_models():
    """The saved classifier and regressor only see the seven input features."""
    import joblib
    for path in (config.CLASSIFICATION_MODEL_PATH, config.REGRESSION_MODEL_PATH):
        model = joblib.load(path)
        used = list(model.feature_names_in_)
        assert used == config.INPUT_FEATURES
        assert not set(used) & set(config.FORBIDDEN_FEATURES)


def test_metrics_files_have_required_content():
    reg = json.loads((config.ARTIFACTS_DIR / "regression_metrics.json").read_text())
    clf = json.loads((config.ARTIFACTS_DIR / "classification_metrics.json").read_text())
    clu = json.loads((config.ARTIFACTS_DIR / "clustering_report.json").read_text())
    for block in [reg["baseline"]["mean_baseline"], *reg["candidates"].values()]:
        assert {"mae", "rmse", "r2"} <= block.keys()
    assert len(clf["candidates"]) >= 2 and "confusion_matrix" in clf["selected_model_test_metrics"]
    assert set(clu["k_comparison"]) == {"2", "3", "4", "5", "6"}
    assert clu["targets_used_for_fitting"] is False
    for d in (reg, clf, clu):
        assert d["verification_code"] == config.VERIFICATION_CODE
