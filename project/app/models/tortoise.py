# project/app/models/tortoise.py

from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator


class ImagePrediction(models.Model):
    """
    Stores image classification results
    """
    filename = fields.CharField(max_length=255)
    image_path = fields.TextField()
    top_prediction = fields.CharField(max_length=255, null=True)
    confidence = fields.FloatField(null=True)
    all_predictions = fields.JSONField(null=True)  # Store top 5 predictions
    created_at = fields.DatetimeField(auto_now_add=True)

    def __str__(self):
        return self.filename


PredictionSchema = pydantic_model_creator(ImagePrediction)