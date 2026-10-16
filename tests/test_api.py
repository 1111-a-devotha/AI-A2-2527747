"""FastAPI endpoint tests (happy and failure paths)."""
from __future__ import annotations

from src import config


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is True
    assert body["verification_code"] == config.VERIFICATION_CODE


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["verification_code"] == config.VERIFICATION_CODE
    assert info["registration_number"] == config.REGISTRATION_NUMBER
    assert len(info["data_sha256"]) == 64


def test_predict_valid(client, valid_record):
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 200
    body = r.json()
    assert body["verification_code"] == config.VERIFICATION_CODE
    assert {"yield_estimate_tons", "high_risk_prediction", "high_risk_probability",
            "cluster"} <= body.keys()


def test_predict_missing_field_returns_422(client, valid_record):
    del valid_record["field_size_ha"]
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 422
    assert r.json()["errors"][0]["field"] == "field_size_ha"


def test_predict_wrong_type_and_range_return_422(client, valid_record):
    valid_record["avg_temp_c"] = "hot"
    valid_record["rainfall_mm"] = 5000
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 422
    fields = {e["field"] for e in r.json()["errors"]}
    assert fields == {"avg_temp_c", "rainfall_mm"}


def test_predict_malformed_json_returns_422(client):
    r = client.post("/predict", content="{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
