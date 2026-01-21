# project/app/models/pydantic.py

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SinglePrediction(BaseModel):
    label: str
    confidence: float


class PredictionResponseSchema(BaseModel):
    id: int
    filename: str


class PredictionDetailSchema(BaseModel):
    id: int
    filename: str
    image_path: str
    top_prediction: Optional[str] = None
    confidence: Optional[float] = None
    all_predictions: Optional[List[SinglePrediction]] = None
    created_at: datetime


class PredictionUpdateSchema(BaseModel):
    """For manual updates if needed"""

    top_prediction: str
    confidence: float


# Alias for backward compatibility
PredictionSchema = PredictionDetailSchema
