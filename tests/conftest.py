"""Shared fixtures: make the project importable and ensure models exist."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

VALID_RECORD = {
    "rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0,
    "fertilizer_kg": 120.0, "field_size_ha": 1.5, "season": "A", "cooperative_support": 1,
}


@pytest.fixture(scope="session", autouse=True)
def trained_models():
    """Train once if the saved models are missing (e.g. on a fresh checkout)."""
    if not config.REGRESSION_MODEL_PATH.exists():
        from src.pipeline import run
        run(config.INSTITUTIONAL_USERNAME, config.LLLL)
    yield


@pytest.fixture()
def valid_record() -> dict:
    """A fresh copy of a valid farm record."""
    return dict(VALID_RECORD)


@pytest.fixture(scope="session")
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)
