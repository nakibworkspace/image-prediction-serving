import os

import numpy as np
from PIL import Image

# Skip TensorFlow in test environment
if os.getenv("TESTING") == "1":
    model = None

    def get_model():
        return None

    async def classify_image(prediction_id: int, image_path: str) -> None:
        return None
else:
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        decode_predictions,
        preprocess_input,
    )
    from sqlalchemy import select

    from app.db import async_session_maker
    from app.models.models import ImagePrediction

    model = None

    def get_model():
        global model
        if model is None:
            model = MobileNetV2(weights="imagenet")
        return model

    async def classify_image(prediction_id: int, image_path: str) -> None:
        """Classify an image using MobileNetV2 and update the database"""
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            classifier = get_model()
            predictions = classifier.predict(img_array)
            decoded = decode_predictions(predictions, top=5)[0]

            all_predictions = [
                {"label": label, "confidence": float(conf)}
                for (_, label, conf) in decoded
            ]

            top_label = decoded[0][1]
            top_confidence = float(decoded[0][2])

            async with async_session_maker() as session:
                result = await session.execute(
                    select(ImagePrediction).where(ImagePrediction.id == prediction_id)
                )
                prediction = result.scalar_one_or_none()
                if prediction:
                    prediction.top_prediction = top_label
                    prediction.confidence = top_confidence
                    prediction.all_predictions = all_predictions
                    await session.commit()

        except Exception as e:
            print(f"Error classifying image: {e}")
            async with async_session_maker() as session:
                result = await session.execute(
                    select(ImagePrediction).where(ImagePrediction.id == prediction_id)
                )
                prediction = result.scalar_one_or_none()
                if prediction:
                    prediction.top_prediction = "Error"
                    prediction.confidence = 0.0
                    await session.commit()
