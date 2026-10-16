"""Load saved artifacts and score one validated farm record."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config
from src.schema import FarmRecord, PredictionResponse

ADVISORY = ("Decision-support estimate from a classroom prototype trained on synthetic "
            "data. A field officer or agronomist must review it before any action.")


class ModelsNotAvailableError(RuntimeError):
    """Raised when saved models are missing (run run_all.py first)."""


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    """Load the saved regression, classification and clustering pipelines once."""
    paths = [config.REGRESSION_MODEL_PATH, config.CLASSIFICATION_MODEL_PATH,
             config.CLUSTERING_MODEL_PATH, config.MODEL_METADATA_PATH]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise ModelsNotAvailableError(
            f"Saved models not found ({', '.join(missing)}). Run: "
            f"python run_all.py --student-id {config.INSTITUTIONAL_USERNAME} --seed {config.LLLL}"
        )
    return {
        "regressor": joblib.load(config.REGRESSION_MODEL_PATH),
        "classifier": joblib.load(config.CLASSIFICATION_MODEL_PATH),
        "clusterer": joblib.load(config.CLUSTERING_MODEL_PATH),
        "metadata": json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8")),
    }


def predict_record(record: FarmRecord) -> PredictionResponse:
    """Return yield estimate, risk class, risk probability and cluster for one farm."""
    art = load_artifacts()
    X = pd.DataFrame([record.model_dump()])[config.INPUT_FEATURES]
    yield_est = float(art["regressor"].predict(X)[0])
    prob = float(art["classifier"].predict_proba(X)[0, 1])
    risk = int(art["classifier"].predict(X)[0])
    cluster = int(art["clusterer"].predict(X[config.CLUSTER_FEATURES])[0])
    profiles = art["metadata"]["clustering"]["profiles"]
    return PredictionResponse(
        verification_code=config.VERIFICATION_CODE,
        yield_estimate_tons=round(yield_est, 4),
        high_risk_prediction=risk,
        high_risk_label="high risk" if risk == 1 else "not high risk",
        high_risk_probability=round(prob, 4),
        cluster=cluster,
        cluster_profile=profiles[cluster]["plain_language_profile"],
        advisory=ADVISORY,
    )
