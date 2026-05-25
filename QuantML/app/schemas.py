"""
schemas.py
==========
Pydantic models for FastAPI request validation and response serialization.
Pydantic auto-validates types and generates Swagger docs at /docs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class MetricsResponse(BaseModel):
    R2:                   float
    RMSE:                 float
    MAE:                  float
    MAPE:                 float
    Directional_Accuracy: float


class PredictionResponse(BaseModel):
    model_name: str
    dates:      List[str]
    actual:     List[float]
    predicted:  List[float]
    metrics:    MetricsResponse


class ForecastResponse(BaseModel):
    model_name:       str
    dates:            List[str]
    forecast:         List[float]
    upper:            List[float]
    lower:            List[float]
    last_known_price: float
    price_change_pct: float
    trend:            str
