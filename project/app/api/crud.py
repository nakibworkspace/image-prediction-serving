# project/app/api/crud.py

from typing import List, Union, Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ImagePrediction


async def create(session: AsyncSession, filename: str, image_path: str) -> int:
    prediction = ImagePrediction(
        filename=filename,
        image_path=image_path,
        top_prediction=None,
        confidence=None,
        all_predictions=None,
    )
    session.add(prediction)
    await session.commit()
    await session.refresh(prediction)
    return prediction.id


async def get(session: AsyncSession, id: int) -> Union[dict, None]:
    result = await session.execute(select(ImagePrediction).where(ImagePrediction.id == id))
    prediction = result.scalar_one_or_none()
    if prediction:
        return {
            "id": prediction.id,
            "filename": prediction.filename,
            "image_path": prediction.image_path,
            "top_prediction": prediction.top_prediction,
            "confidence": prediction.confidence,
            "all_predictions": prediction.all_predictions,
            "created_at": prediction.created_at,
        }
    return None


async def get_all(session: AsyncSession) -> List:
    result = await session.execute(select(ImagePrediction))
    predictions = result.scalars().all()
    return [
        {
            "id": p.id,
            "filename": p.filename,
            "image_path": p.image_path,
            "top_prediction": p.top_prediction,
            "confidence": p.confidence,
            "all_predictions": p.all_predictions,
            "created_at": p.created_at,
        }
        for p in predictions
    ]


async def delete(session: AsyncSession, id: int) -> int:
    result = await session.execute(select(ImagePrediction).where(ImagePrediction.id == id))
    prediction = result.scalar_one_or_none()
    if prediction:
        await session.delete(prediction)
        await session.commit()
        return id
    return 0


async def update(
    session: AsyncSession,
    id: int,
    top_prediction: str,
    confidence: float,
    all_predictions: Optional[List[Dict[str, Any]]]
) -> Union[dict, None]:
    result = await session.execute(select(ImagePrediction).where(ImagePrediction.id == id))
    prediction = result.scalar_one_or_none()

    if prediction:
        prediction.top_prediction = top_prediction
        prediction.confidence = confidence
        prediction.all_predictions = all_predictions
        await session.commit()
        await session.refresh(prediction)

        return {
            "id": prediction.id,
            "filename": prediction.filename,
            "image_path": prediction.image_path,
            "top_prediction": prediction.top_prediction,
            "confidence": prediction.confidence,
            "all_predictions": prediction.all_predictions,
            "created_at": prediction.created_at,
        }
    return None
