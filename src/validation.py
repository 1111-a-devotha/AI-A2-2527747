"""Identity, schema, range, missing-value, duplicate and leakage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


class DataValidationError(ValueError):
    """Raised when the identity, seed or dataset breaks a required rule."""


def validate_seed(seed: int) -> None:
    """Reject any seed that is not the personal LLLL value."""
    if seed != config.LLLL:
        raise DataValidationError(
            f"Seed {seed} rejected: the personal seed for {config.VERIFICATION_CODE} "
            f"must be {config.LLLL} (last four digits of {config.REGISTRATION_NUMBER})."
        )


def validate_student_id(student_id: str, seed: int) -> None:
    """Check the student ID belongs to this submission and matches the seed."""
    try:
        digits = config.digits_from_student_id(student_id)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc
    if digits != config.REG_DIGITS:
        raise DataValidationError(
            f"Student ID '{student_id}' does not match registration "
            f"{config.REGISTRATION_NUMBER} ({config.INSTITUTIONAL_USERNAME})."
        )
    if int(digits[-4:]) != seed:
        raise DataValidationError(
            f"Seed {seed} does not match the last four digits of '{student_id}' "
            f"({digits[-4:]})."
        )
    validate_seed(seed)


def check_schema(df: pd.DataFrame) -> None:
    """Raise if any required column is missing or has an impossible type."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    for col in config.NUMERIC_FEATURES + [config.REGRESSION_TARGET]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric, got {df[col].dtype}.")
    bad_season = set(df["season"].dropna().unique()) - set(config.SEASON_VALUES)
    if bad_season:
        raise DataValidationError(f"Unexpected season values: {sorted(bad_season)}")
    for col in ["cooperative_support", config.CLASSIFICATION_TARGET]:
        bad = set(df[col].dropna().unique()) - {0, 1}
        if bad:
            raise DataValidationError(f"Column '{col}' must be 0/1, found {sorted(bad)}")


def range_violations(df: pd.DataFrame) -> dict[str, int]:
    """Count non-missing values outside the configured valid ranges."""
    out = {}
    for col, (low, high) in config.RANGES.items():
        values = df[col].dropna()
        out[col] = int(((values < low) | (values > high)).sum())
    return out


def leakage_report(df: pd.DataFrame, features: list[str]) -> dict:
    """Confirm no target (or target-derived) column is used as a feature.

    Also flags any numeric feature that is almost perfectly correlated with a
    target, which would indicate hidden leakage.
    """
    forbidden_used = [f for f in features if f in config.FORBIDDEN_FEATURES]
    if forbidden_used:
        raise DataValidationError(f"Target leakage: forbidden features used {forbidden_used}")
    correlations = {}
    for target in config.TARGET_COLUMNS:
        for col in config.NUMERIC_FEATURES:
            pair = df[[col, target]].dropna()
            correlations[f"{col}~{target}"] = round(float(pair[col].corr(pair[target])), 4)
    suspicious = {k: v for k, v in correlations.items() if abs(v) > 0.98}
    if suspicious:
        raise DataValidationError(f"Possible leakage, near-perfect correlation: {suspicious}")
    return {
        "features_used": features,
        "forbidden_features": config.FORBIDDEN_FEATURES,
        "forbidden_features_used": [],
        "max_abs_feature_target_correlation": max(abs(v) for v in correlations.values()),
        "feature_target_correlations": correlations,
        "status": "passed",
    }


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate the raw dataset, remove exact duplicates and invalid rows.

    Missing values are *not* filled here: imputation is learned inside the
    model pipelines from the training split only.

    Returns the cleaned DataFrame (with a ``row_id`` column holding the
    original CSV row position) and a validation report.
    """
    check_schema(df)
    report: dict = {"verification_code": config.VERIFICATION_CODE}
    report["rows_loaded"] = int(len(df))
    report["required_columns_present"] = True
    report["missing_values_before_cleaning"] = {
        c: int(v) for c, v in df[config.REQUIRED_COLUMNS].isna().sum().items()
    }

    dup_mask = df.duplicated(keep="first")
    report["exact_duplicate_rows_found"] = int(dup_mask.sum())
    report["duplicate_row_positions"] = [int(i) for i in np.flatnonzero(dup_mask)]
    as_text = df.astype(str)  # NaN-safe row comparison
    duplicated_of = []
    for pos in np.flatnonzero(dup_mask):
        same = np.flatnonzero((as_text == as_text.iloc[pos]).all(axis=1))
        duplicated_of.append(int(same[0]))
    report["duplicates_are_copies_of_rows"] = duplicated_of

    clean = df.loc[~dup_mask].copy()
    clean.insert(0, "row_id", clean.index.astype(int))

    violations = range_violations(clean)
    report["range_violations"] = violations
    bad_rows = pd.Series(False, index=clean.index)
    for col, (low, high) in config.RANGES.items():
        v = clean[col]
        bad_rows |= v.notna() & ((v < low) | (v > high))
    target_missing = clean[config.TARGET_COLUMNS].isna().any(axis=1)
    report["rows_dropped_out_of_range"] = int(bad_rows.sum())
    report["rows_dropped_missing_target"] = int(target_missing.sum())
    clean = clean.loc[~bad_rows & ~target_missing].reset_index(drop=True)

    report["rows_after_cleaning"] = int(len(clean))
    report["missing_values_after_cleaning"] = {
        c: int(v) for c, v in clean[config.INPUT_FEATURES].isna().sum().items()
    }
    report["missing_value_strategy"] = (
        "Kept as NaN in the cleaned data; median imputation is fitted on the "
        "training split only inside each scikit-learn Pipeline."
    )
    ypha = clean[config.REGRESSION_TARGET] / clean["field_size_ha"]
    recomputed = (ypha < ypha.median()).astype(int)
    report["high_risk_rule_agreement"] = round(float((recomputed == clean[config.CLASSIFICATION_TARGET]).mean()), 4)
    report["class_balance"] = {
        str(k): int(v) for k, v in clean[config.CLASSIFICATION_TARGET].value_counts().sort_index().items()
    }
    report["leakage"] = leakage_report(clean, config.INPUT_FEATURES)
    return clean, report
