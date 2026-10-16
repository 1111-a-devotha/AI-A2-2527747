"""Central configuration: student identity, paths, column schema and valid ranges.

Every other module reads its constants from here, so a rule such as a valid
range only has to be changed in one place (it then applies to data validation,
the CLI and the API at the same time).
"""
from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Student identity
# ---------------------------------------------------------------------------
STUDENT_NAME = "TUYISINGIZE Devotha"
REGISTRATION_NUMBER = "25/27747"
REG_DIGITS = REGISTRATION_NUMBER.replace("/", "")          # "2527747"
INSTITUTIONAL_USERNAME = "ug2527747@ines.ac.rw"
LLLL = int(REG_DIGITS[-4:])                                  # 7747
VERIFICATION_CODE = f"AI-A2-{LLLL:04d}"                      # "AI-A2-7747"

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, built with pathlib)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
DATA_CSV = DATA_DIR / f"AI_A2_{REG_DIGITS}.csv"
FINGERPRINT_TXT = ARTIFACTS_DIR / "data_fingerprint.txt"

REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.joblib"
CLUSTERING_MODEL_PATH = MODELS_DIR / "clustering_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

REGRESSION_TARGET = "yield_tons"
CLASSIFICATION_TARGET = "high_risk"
TARGET_COLUMNS = [REGRESSION_TARGET, CLASSIFICATION_TARGET]
# Anything derived from the targets must never be used as an input feature.
FORBIDDEN_FEATURES = TARGET_COLUMNS + ["yield_per_ha", "yield_tons_per_ha"]
REQUIRED_COLUMNS = INPUT_FEATURES + TARGET_COLUMNS

# Clustering groups farms by growing conditions only (no targets).
CLUSTER_FEATURES = list(NUMERIC_FEATURES)

# Valid input ranges (inclusive). Used by data validation, CLI and API.
RANGES: dict[str, tuple[float, float]] = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
SEASON_VALUES = ("A", "B")
SUPPORT_VALUES = (0, 1)

# Agronomic optimum used for the engineered temperature-stress feature.
OPTIMAL_TEMP_C = 18.0

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
CV_FOLDS = 5
K_RANGE = range(2, 7)          # K = 2 ... 6
FLOAT_FORMAT = "%.4f"


def base_row_count(seed: int) -> int:
    """Return the individual base row count, 240 + (LLLL mod 41)."""
    return 240 + (seed % 41)


def digits_from_student_id(student_id: str) -> str:
    """Extract the registration digits from a username or registration number.

    ``ug2527747@ines.ac.rw`` and ``25/27747`` both give ``"2527747"``.
    """
    local_part = student_id.split("@")[0]
    digits = "".join(re.findall(r"\d", local_part))
    if len(digits) < 4:
        raise ValueError(
            f"Student ID '{student_id}' does not contain a registration number "
            "with at least four digits."
        )
    return digits
