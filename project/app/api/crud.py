# project/app/api/crud.py

from typing import List, Union
from app.models.tortoise import ImagePrediction


async def create(filename: str, image_path: str) -> int:
    """Create a new prediction record"""
    prediction = ImagePrediction(
        filename=filename,
        image_path=image_path,
        top_prediction=None,
        confidence=None,
    )
    await prediction.save()
    return prediction.id


async def get(id: int) -> Union[dict, None]:
    """Get a single prediction by ID"""
    prediction = await ImagePrediction.filter(id=id).first().values()
    if prediction:
        return prediction
    return None


async def get_all() -> List:
    """Get all predictions"""
    predictions = await ImagePrediction.all().values()
    return predictions


async def delete(id: int) -> int:
    """Delete a prediction by ID"""
    prediction = await ImagePrediction.filter(id=id).first().delete()
    return prediction


async def update(id: int, top_prediction: str, confidence: float) -> Union[dict, None]:
    """Update a prediction (for manual corrections if needed)"""
    prediction = await ImagePrediction.filter(id=id).update(
        top_prediction=top_prediction,
        confidence=confidence
    )
    if prediction:
        updated_prediction = await ImagePrediction.filter(id=id).first().values()
        return updated_prediction
    return None