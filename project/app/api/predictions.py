# project/app/api/predictions.py

import os
import shutil
from typing import List
from fastapi import APIRouter, HTTPException, Path, BackgroundTasks, UploadFile, File

from app.api import crud
from app.classifier import classify_image
from app.models.pydantic import (
    PredictionResponseSchema,
    PredictionDetailSchema,
    PredictionUpdateSchema,
)
from app.models.tortoise import PredictionSchema


router = APIRouter()

# Directory to store uploaded images
UPLOAD_DIR = "/usr/src/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=PredictionResponseSchema, status_code=201)
async def create_prediction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> PredictionResponseSchema:
    """
    Upload an image for classification
    
    The image will be processed in the background and results
    will be available via the GET endpoint
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create database record
    prediction_id = await crud.create(filename=file.filename, image_path=file_path)
    
    # Schedule background classification
    background_tasks.add_task(classify_image, prediction_id, file_path)
    
    return {"id": prediction_id, "filename": file.filename}


@router.get("/{id}/", response_model=PredictionSchema)
async def read_prediction(id: int = Path(..., gt=0)) -> PredictionSchema:
    """Get prediction results by ID"""
    prediction = await crud.get(id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return prediction


@router.get("/", response_model=List[PredictionSchema])
async def read_all_predictions() -> List[PredictionSchema]:
    """Get all predictions"""
    return await crud.get_all()


@router.delete("/{id}/", response_model=PredictionResponseSchema)
async def delete_prediction(id: int = Path(..., gt=0)) -> PredictionResponseSchema:
    """Delete a prediction by ID"""
    prediction = await crud.get(id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Delete image file if it exists
    if os.path.exists(prediction["image_path"]):
        os.remove(prediction["image_path"])
    
    await crud.delete(id)
    
    return {"id": id, "filename": prediction["filename"]}


@router.put("/{id}/", response_model=PredictionSchema)
async def update_prediction(
    payload: PredictionUpdateSchema,
    id: int = Path(..., gt=0),
) -> PredictionSchema:
    """Update prediction results (for manual corrections)"""
    prediction = await crud.update(id, payload.top_prediction, payload.confidence)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return prediction