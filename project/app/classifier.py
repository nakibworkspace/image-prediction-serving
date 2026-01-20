# project/app/classifier.py

import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

from app.models.tortoise import ImagePrediction

# Load model once at module level for efficiency
model = None


def get_model():
    """Lazy load the model"""
    global model
    if model is None:
        model = MobileNetV2(weights="imagenet")
    return model


async def classify_image(prediction_id: int, image_path: str) -> None:
    """
    Classify an image using MobileNetV2 and update the database

    Args:
        prediction_id: Database ID of the prediction record
        image_path: Path to the image file
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))  # MobileNetV2 input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get model and predict
        classifier = get_model()
        predictions = classifier.predict(img_array)

        # Decode predictions
        decoded = decode_predictions(predictions, top=5)[0]

        # Format results
        all_predictions = [
            {"label": label, "confidence": float(conf)} for (_, label, conf) in decoded
        ]

        top_label = decoded[0][1]
        top_confidence = float(decoded[0][2])

        # Update database
        await ImagePrediction.filter(id=prediction_id).update(
            top_prediction=top_label,
            confidence=top_confidence,
            all_predictions=all_predictions,
        )

    except Exception as e:
        # Log error and update with error status
        print(f"Error classifying image: {e}")
        await ImagePrediction.filter(id=prediction_id).update(
            top_prediction="Error",
            confidence=0.0,
        )
