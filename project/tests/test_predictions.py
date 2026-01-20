# project/tests/test_predictions.py

import json
import io
from datetime import datetime
from PIL import Image
import pytest

from app.api import crud, predictions


# In tests/test_predictions.py, update test_create_prediction:

def test_create_prediction(test_app, monkeypatch):
    """Test image upload endpoint"""
    async def mock_create(filename, image_path):
        return 1

    monkeypatch.setattr(crud, "create", mock_create)
    
    # ADD THIS - mock the classify_image function
    def mock_classify(prediction_id, image_path):
        return None
    
    monkeypatch.setattr(predictions, "classify_image", mock_classify)

    # Create a dummy image file
    image = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = test_app.post(
        "/predictions/",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "test.jpg"


def test_create_prediction_invalid_file(test_app):
    """Test uploading non-image file"""
    response = test_app.post(
        "/predictions/",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image"


def test_read_prediction(test_app, monkeypatch):
    """Test getting a single prediction"""
    test_data = {
        "id": 1,
        "filename": "test.jpg",
        "image_path": "/path/to/test.jpg",
        "top_prediction": "golden_retriever",
        "confidence": 0.95,
        "all_predictions": [
            {"label": "golden_retriever", "confidence": 0.95},
            {"label": "labrador", "confidence": 0.03},
        ],
        "created_at": datetime.utcnow().isoformat(),
    }

    async def mock_get(id):
        return test_data

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.get("/predictions/1/")
    assert response.status_code == 200
    assert response.json()["top_prediction"] == "golden_retriever"


def test_read_prediction_incorrect_id(test_app, monkeypatch):
    """Test getting non-existent prediction"""
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.get("/predictions/999/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"

    response = test_app.get("/predictions/0/")
    assert response.status_code == 422


def test_read_all_predictions(test_app, monkeypatch):
    """Test getting all predictions"""
    test_data = [
        {
            "id": 1,
            "filename": "test1.jpg",
            "image_path": "/path/to/test1.jpg",
            "top_prediction": "golden_retriever",
            "confidence": 0.95,
            "all_predictions": [],
            "created_at": datetime.utcnow().isoformat(),
        },
        {
            "id": 2,
            "filename": "test2.jpg",
            "image_path": "/path/to/test2.jpg",
            "top_prediction": "tabby_cat",
            "confidence": 0.89,
            "all_predictions": [],
            "created_at": datetime.utcnow().isoformat(),
        },
    ]

    async def mock_get_all():
        return test_data

    monkeypatch.setattr(crud, "get_all", mock_get_all)

    response = test_app.get("/predictions/")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_remove_prediction(test_app, monkeypatch):
    """Test deleting a prediction"""
    test_data = {
        "id": 1,
        "filename": "test.jpg",
        "image_path": "/tmp/test.jpg",
        "top_prediction": "golden_retriever",
        "confidence": 0.95,
        "all_predictions": [],
        "created_at": datetime.utcnow().isoformat(),
    }

    async def mock_get(id):
        return test_data

    async def mock_delete(id):
        return id

    monkeypatch.setattr(crud, "get", mock_get)
    monkeypatch.setattr(crud, "delete", mock_delete)

    response = test_app.delete("/predictions/1/")
    assert response.status_code == 200
    assert response.json()["id"] == 1


def test_remove_prediction_incorrect_id(test_app, monkeypatch):
    """Test deleting non-existent prediction"""
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.delete("/predictions/999/")
    assert response.status_code == 404

    response = test_app.delete("/predictions/0/")
    assert response.status_code == 422


def test_update_prediction(test_app, monkeypatch):
    """Test updating a prediction"""
    test_response = {
        "id": 1,
        "filename": "test.jpg",
        "image_path": "/path/to/test.jpg",
        "top_prediction": "labrador",
        "confidence": 0.98,
        "all_predictions": [],
        "created_at": datetime.utcnow().isoformat(),
    }

    async def mock_update(id, top_prediction, confidence):
        return test_response

    monkeypatch.setattr(crud, "update", mock_update)

    response = test_app.put(
        "/predictions/1/",
        data=json.dumps({"top_prediction": "labrador", "confidence": 0.98}),
    )
    assert response.status_code == 200
    assert response.json()["top_prediction"] == "labrador"