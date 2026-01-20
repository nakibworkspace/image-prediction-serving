# project/app/models/pydantic.py

from typing import Dict, List, Optional

from pydantic import BaseModel


class PredictionResponseSchema(BaseModel):
    id: int
    filename: str


class PredictionDetailSchema(BaseModel):
    id: int
    filename: str
    image_path: str
    top_prediction: Optional[str] = None
    confidence: Optional[float] = None
    all_predictions: Optional[List[Dict[str, float]]] = None
    created_at: str


class PredictionUpdateSchema(BaseModel):
    """For manual updates if needed"""

    top_prediction: str
    confidence: float
