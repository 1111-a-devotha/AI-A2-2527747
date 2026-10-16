"""Individual synthetic dataset generator (seed = LLLL).

The generator follows the assignment specification exactly:
* base rows = 240 + (LLLL mod 41), drawn with ``numpy.random.default_rng(LLLL)``
* yield formula with season effect, Normal(0, 0.80) noise and a 0.50 t floor
* ``high_risk`` = yield per hectare strictly below its median
* m = round(0.05 x base rows) missing values in soil moisture and in fertilizer
  (positions chosen by the same generator), then three exact duplicate rows.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Generate the raw (dirty) dataset for ``seed`` as a DataFrame.

    The order of random draws is fixed, so the same seed always produces the
    same rows on any computer.
    """
    rng = np.random.default_rng(seed)
    n = config.base_row_count(seed)

    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    field_size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), size=n)
    support = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temp - config.OPTIMAL_TEMP_C)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    yield_per_ha = yield_tons / field_size
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fertilizer,
            "field_size_ha": field_size,
            "season": season,
            "cooperative_support": support.astype(int),
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    # Deliberate data problems ------------------------------------------------
    m = int(round(0.05 * n))
    soil_missing = rng.choice(n, size=m, replace=False)
    fert_missing = rng.choice(n, size=m, replace=False)
    df.loc[soil_missing, "soil_moisture_pct"] = np.nan
    df.loc[fert_missing, "fertilizer_kg"] = np.nan

    duplicate_positions = rng.choice(n, size=3, replace=False)
    df = pd.concat([df, df.iloc[duplicate_positions]], ignore_index=True)
    return df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise the dataset with 4 decimals and Unix line endings.

    Fixing ``lineterminator`` keeps the SHA-256 fingerprint identical on
    Windows, macOS and Linux.
    """
    text = df.to_csv(index=False, float_format=config.FLOAT_FORMAT, lineterminator="\n")
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def write_dataset(seed: int, csv_path: Path = config.DATA_CSV,
                  fingerprint_path: Path = config.FINGERPRINT_TXT) -> dict:
    """Generate the dataset, write the CSV and its fingerprint file.

    Returns a summary dictionary (row counts, hash, paths).
    """
    df = generate_dataframe(seed)
    data = dataframe_to_csv_bytes(df)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(data)

    digest = sha256_bytes(csv_path.read_bytes())
    n = config.base_row_count(seed)
    summary = {
        "verification_code": config.VERIFICATION_CODE,
        "seed": seed,
        "base_rows": n,
        "missing_per_column": int(round(0.05 * n)),
        "duplicate_rows_appended": 3,
        "total_rows_in_csv": int(len(df)),
        "csv_file": csv_path.name,
        "sha256": digest,
    }
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(
        f"verification_code: {config.VERIFICATION_CODE}\n"
        f"student: {config.STUDENT_NAME} ({config.REGISTRATION_NUMBER})\n"
        f"seed: {seed}\n"
        f"csv_file: data/{csv_path.name}\n"
        f"rows_in_csv: {len(df)}\n"
        f"sha256: {digest}\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary
