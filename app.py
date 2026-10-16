"""FastAPI service for the Musanze Harvest Intelligence Service (AI-A2-7747).

Run:
    uvicorn app:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src import config
from src.inference import ModelsNotAvailableError, load_artifacts, predict_record
from src.schema import FarmRecord, PredictionResponse

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    version="1.0.0",
    description=(
        f"{config.VERIFICATION_CODE} - {config.STUDENT_NAME} ({config.REGISTRATION_NUMBER}). "
        "Classroom decision-support prototype trained on synthetic data. "
        "It does not replace an agronomist and must not be used to decide access to "
        "finance, insurance or public services."
    ),
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a clear 422 listing every invalid field."""
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
               "problem": e["msg"], "received": None if e["type"] == "missing" else e.get("input")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={
        "verification_code": config.VERIFICATION_CODE,
        "status": "invalid_input",
        "message": "The farm record failed validation. Fix the fields below and retry.",
        "errors": errors,
    })


@app.exception_handler(ModelsNotAvailableError)
async def models_missing_handler(request: Request, exc: ModelsNotAvailableError) -> JSONResponse:
    """Return 503 when the models have not been trained yet."""
    return JSONResponse(status_code=503, content={
        "verification_code": config.VERIFICATION_CODE, "status": "models_unavailable",
        "message": str(exc)})


@app.get("/health")
def health() -> dict:
    """Liveness check that also reports whether the saved models load."""
    try:
        load_artifacts()
        loaded = True
    except ModelsNotAvailableError:
        loaded = False
    return {"status": "ok" if loaded else "degraded", "models_loaded": loaded,
            "verification_code": config.VERIFICATION_CODE}


@app.get("/model-info")
def model_info() -> dict:
    """Identity, data fingerprint, selected models, metrics, features and valid ranges."""
    return load_artifacts()["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord) -> PredictionResponse:
    """Predict yield, high-risk class/probability and cluster for one farm."""
    return predict_record(record)
