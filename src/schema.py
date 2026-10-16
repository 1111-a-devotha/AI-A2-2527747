"""Typed input/output contracts shared by predict.py and the FastAPI app.

Ranges come from ``config.RANGES``, so changing a range rule in one place
updates the CLI, the API and the data validation together.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config

_R = config.RANGES


class FarmRecord(BaseModel):
    """One farm record. Strict types: strings are not silently turned into numbers."""

    model_config = ConfigDict(
        extra="forbid",   # unknown fields (e.g. yield_tons) are rejected
        strict=True,
        json_schema_extra={"example": {
            "rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0, "field_size_ha": 1.5, "season": "A",
            "cooperative_support": 1,
        }},
    )

    rainfall_mm: float = Field(..., ge=_R["rainfall_mm"][0], le=_R["rainfall_mm"][1],
                               description="Seasonal rainfall in mm")
    avg_temp_c: float = Field(..., ge=_R["avg_temp_c"][0], le=_R["avg_temp_c"][1],
                              description="Average temperature in C")
    soil_moisture_pct: float = Field(..., ge=_R["soil_moisture_pct"][0],
                                     le=_R["soil_moisture_pct"][1],
                                     description="Soil moisture in %")
    fertilizer_kg: float = Field(..., ge=_R["fertilizer_kg"][0], le=_R["fertilizer_kg"][1],
                                 description="Fertilizer applied in kg")
    field_size_ha: float = Field(..., ge=_R["field_size_ha"][0], le=_R["field_size_ha"][1],
                                 description="Field size in hectares")
    season: Literal["A", "B"] = Field(..., description="Rwandan growing season A or B")
    cooperative_support: Literal[0, 1] = Field(..., description="1 if the farm receives cooperative support")

    @field_validator("rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
                     "field_size_ha", "cooperative_support", mode="before")
    @classmethod
    def reject_booleans(cls, value):
        """JSON true/false must not be accepted as 1/0."""
        if isinstance(value, bool):
            raise ValueError("Input should be a number, not true/false")
        return value


class PredictionResponse(BaseModel):
    """Prediction returned by the CLI and POST /predict."""

    verification_code: str
    yield_estimate_tons: float
    high_risk_prediction: int
    high_risk_label: str
    high_risk_probability: float
    cluster: int
    cluster_profile: str
    advisory: str
