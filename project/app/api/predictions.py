# project/app/api/predictions.py

import os
from typing import List
import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Path, UploadFile, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import crud
from app.classifier import classify_image
from app.models.pydantic import PredictionResponseSchema, PredictionUpdateSchema
from app.models.pydantic import PredictionSchema
from app.db import get_session

router = APIRouter()

@router.post("/", response_model=PredictionResponseSchema, status_code=201)
async def create_prediction(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),  # ADD THIS
) -> PredictionResponseSchema:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    upload_path = f"/usr/src/app/uploads/{file.filename}"
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    prediction_id = await crud.create(session, file.filename, upload_path)  # ADD session
    background_tasks.add_task(classify_image, upload_path, prediction_id)
    
    return {"id": prediction_id, "filename": file.filename}

@router.get("/{id}/", response_model=PredictionSchema)
async def read_prediction(
    id: int = Path(..., gt=0),
    session: AsyncSession = Depends(get_session),  # ADD THIS
) -> PredictionSchema:
    prediction = await crud.get(session, id)  # ADD session
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@router.get("/", response_model=List[PredictionSchema])
async def read_all_predictions(
    session: AsyncSession = Depends(get_session),  # ADD THIS
) -> List[PredictionSchema]:
    return await crud.get_all(session)  # ADD session

@router.delete("/{id}/", response_model=PredictionResponseSchema)
async def delete_prediction(
    id: int = Path(..., gt=0),
    session: AsyncSession = Depends(get_session),  # ADD THIS
) -> PredictionResponseSchema:
    prediction = await crud.get(session, id)  # ADD session
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if os.path.exists(prediction["image_path"]):
        os.remove(prediction["image_path"])
    
    await crud.delete(session, id)  # ADD session
    return {"id": prediction["id"], "filename": prediction["filename"]}

@router.put("/{id}/", response_model=PredictionSchema)
async def update_prediction(
    payload: PredictionUpdateSchema,
    id: int = Path(..., gt=0),
    session: AsyncSession = Depends(get_session),  # ADD THIS
) -> PredictionSchema:
    prediction = await crud.get(session, id)  # ADD session
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    updated = await crud.update(session, id, payload.top_prediction, payload.confidence, None)  # ADD session
    return updated