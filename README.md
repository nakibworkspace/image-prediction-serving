# Production ML Pipeline: Image Classification Service on AWS

## Part 1: Local Development Setup

In this hands-on lab, you'll learn how to set up a local development environment for an image classification service with Python, FastAPI, TensorFlow, and Docker. The service will be exposed via a RESTful API that allows users to upload images and receive AI-generated predictions using MobileNetV2.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Docker Setup](#docker-setup)
4. [Database Configuration](#database-configuration)
5. [FastAPI Application Setup](#fastapi-application-setup)
6. [Configuration Management](#configuration-management)
7. [Database Integration with Tortoise ORM](#database-integration-with-tortoise-orm)
8. [Data Models](#data-models)
9. [ML Classifier Implementation](#ml-classifier-implementation)
10. [RESTful API Routes](#restful-api-routes)
11. [Running the Application](#running-the-application)
12. [Testing with curl](#testing-with-curl)

---

## Project Overview

### Task Description

A RESTful API service that:
- Accepts image uploads via POST requests
- Classifies images using MobileNetV2 neural network (trained on ImageNet)
- Stores predictions in a PostgreSQL database
- Provides full CRUD operations (Create, Read, Update, Delete) for predictions

### Tool Prerequisites

| Technology | Purpose |
|------------|---------|
| **FastAPI** | Modern, fast web framework for building APIs |
| **Docker** | Containerization for consistent development environments |
| **PostgreSQL** | Relational database for storing predictions |
| **Tortoise ORM** | Async ORM for database operations |
| **TensorFlow** | ML framework for image classification |
| **MobileNetV2** | Pre-trained neural network for image recognition |
| **Pillow** | Image processing library |

---

## Project Structure

```
image-prediction-serving/
├── project/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Configuration management
│   │   ├── db.py                # Database connection setup
│   │   ├── classifier.py        # ML model & prediction logic
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── ping.py          # Health check endpoint
│   │   │   ├── predictions.py   # Prediction CRUD endpoints
│   │   │   └── crud.py          # Database operations
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── tortoise.py      # Tortoise ORM models
│   │       └── pydantic.py      # Pydantic schemas
│   ├── db/
│   │   ├── Dockerfile           # PostgreSQL container config
│   │   └── create.sql           # Database initialization
│   ├── migrations/
│   │   └── models/
│   │       └── 0_20260120200937_init.py
│   ├── uploads/                 # Uploaded image storage
│   ├── Dockerfile               # Web application container
│   ├── entrypoint.sh            # Container startup script
│   ├── requirements.txt         # Production dependencies
│   └── pyproject.toml           # Aerich migration config
├── docker-compose.yml           # Multi-container orchestration
└── README.md                    # This documentation
```

---

## Step by Step Implementation

## Step 1: Environment Setup

### 1.1 Update the environment

```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Clone the repository

```bash
git clone <repository-url>
cd image-prediction-serving
```

---

## Step 2: Docker Setup

### 2.1 Docker Compose Configuration

This `docker-compose.yml` defines a two-container environment consisting of a FastAPI/Python web application with TensorFlow and a PostgreSQL database. The web service builds from local code, runs with hot-reloading for development, and connects to the database service via an internal network.

**docker-compose.yml**
```yaml
services:

  web:
    build: ./project
    command: uvicorn app.main:app --reload --workers 1 --host 0.0.0.0 --port 8000
    volumes:
      - ./project:/usr/src/app
    ports:
      - 8004:8000
    environment:
      - ENVIRONMENT=dev
      - TESTING=0
      - DATABASE_URL=postgres://postgres:postgres@web-db:5432/web_cv_dev
      - DATABASE_TEST_URL=postgres://postgres:postgres@web-db:5432/web_cv_test
    depends_on:
      - web-db

  web-db:
    build:
      context: ./project/db
      dockerfile: Dockerfile
    expose:
      - 5432
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
```

**Key Configuration Points:**

| Setting | Description |
|---------|-------------|
| `build: ./project` | Builds the web service from the project directory |
| `--reload` | Enables hot reloading during development |
| `volumes` | Mounts local code for live updates |
| `ports: 8004:8000` | Maps container port 8000 to host port 8004 |
| `depends_on` | Ensures database starts before web service |
| `expose: 5432` | Makes PostgreSQL available to linked containers |

### 2.2 Web Application Dockerfile

This Dockerfile sets up a Python environment optimized for TensorFlow and image processing, with database connectivity for the ML classification service.

**project/Dockerfile**
```dockerfile
# pull official base image
FROM python:3.11-slim-bookworm

# set working directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
  && apt-get -y install netcat-openbsd gcc g++ libpq-dev \
     libhdf5-dev libglib2.0-0 libsm6 libxext6 libgomp1 \
  && apt-get clean

# install python dependencies
RUN pip install --upgrade pip setuptools wheel
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# create uploads directory
RUN mkdir -p /usr/src/app/uploads

# add app
COPY . .

# add entrypoint.sh
COPY ./entrypoint.sh .
RUN chmod +x /usr/src/app/entrypoint.sh

# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
```

**Environment Variables Explained:**

| Variable | Purpose |
|----------|---------|
| `PYTHONDONTWRITEBYTECODE=1` | Prevents Python from writing `.pyc` files |
| `PYTHONUNBUFFERED=1` | Ensures Python output is sent directly to terminal |

**System Dependencies:**

| Package | Purpose |
|---------|---------|
| `libhdf5-dev` | Required for TensorFlow model loading |
| `libglib2.0-0` | Image processing support |
| `libgomp1` | OpenMP for parallel processing |

### 2.3 Container Entrypoint Script

The entrypoint script ensures the database is ready before starting the application. This script uses `netcat` to poll the database port until PostgreSQL is accepting connections, preventing race conditions during startup.

**project/entrypoint.sh**
```bash
#!/bin/sh

echo "Waiting for postgres..."

while ! nc -z web-db 5432; do
  sleep 0.1
done

echo "PostgreSQL started"

exec "$@"
```

### 2.4 Docker Ignore File

`.dockerignore` file prevents bulky or sensitive files from being sent to the Docker daemon, which speeds up builds.

**project/.dockerignore**
```
env
.dockerignore
Dockerfile
Dockerfile.prod
__pycache__
*.pyc
uploads/*
```

---

## Step 3: Database Configuration

### 3.1 PostgreSQL Dockerfile

Files placed in `/docker-entrypoint-initdb.d` are automatically executed when the PostgreSQL container initializes.

**project/db/Dockerfile**
```dockerfile
# pull official base image
FROM postgres:17

# run create.sql on init
ADD create.sql /docker-entrypoint-initdb.d
```

### 3.2 Database Initialization Script

**project/db/create.sql**
```sql
CREATE DATABASE web_cv_dev;
CREATE DATABASE web_cv_test;
```

This creates two separate databases:
- `web_cv_dev` - For development
- `web_cv_test` - For running tests in isolation

---

## Step 4: FastAPI Application Setup

### 4.1 Main Application Entry Point

The script creates a modular FastAPI setup that uses the factory pattern to initialize the application and register specific functionality.

**project/app/main.py**
```python
import logging

from fastapi import FastAPI

from app.api import ping, predictions
from app.db import init_db

log = logging.getLogger("uvicorn")


def create_application() -> FastAPI:
    application = FastAPI(title="Image Classification API", version="1.0.0")
    application.include_router(ping.router)
    application.include_router(
        predictions.router, prefix="/predictions", tags=["predictions"]
    )

    return application


app = create_application()

init_db(app)
```

**Core Components:**

| Component | Purpose |
|-----------|---------|
| `create_application()` | Factory function for creating the FastAPI instance |
| `include_router(ping.router)` | Registers the health check endpoint at root level |
| `include_router(predictions.router)` | Registers prediction endpoints under `/predictions` prefix |
| `tags=["predictions"]` | Groups endpoints in OpenAPI documentation |
| `init_db(app)` | Initializes database connection on startup |

---

## Step 5: Configuration Management

### 5.1 Settings with Pydantic

**project/app/config.py**
```python
import logging
from functools import lru_cache

from pydantic import AnyUrl
from pydantic_settings import BaseSettings

log = logging.getLogger("uvicorn")


class Settings(BaseSettings):
    environment: str = "dev"
    testing: bool = False
    database_url: AnyUrl = None


@lru_cache()
def get_settings() -> BaseSettings:
    log.info("Loading config settings from the environment...")
    return Settings()
```

**Key Features:**

| Feature | Description |
|---------|-------------|
| `BaseSettings` | Automatically reads from environment variables |
| `@lru_cache()` | Caches settings to avoid repeated environment reads |
| `AnyUrl` | Validates that `database_url` is a proper URL format |

Environment variables are automatically mapped to settings:
- `ENVIRONMENT` → `settings.environment`
- `TESTING` → `settings.testing`
- `DATABASE_URL` → `settings.database_url`

---

## Step 6: Database Integration with Tortoise ORM

This script configures the Tortoise ORM to manage database connections for a FastAPI app and includes a utility to programmatically generate database tables.

**Why Tortoise ORM?**

Tortoise ORM is an easy-to-use async ORM (Object-Relational Mapper) inspired by Django, specifically designed to handle database interactions in modern asynchronous Python frameworks like FastAPI.

### 6.1 Database Connection Setup

**project/app/db.py**
```python
import logging
import os

from fastapi import FastAPI
from tortoise import Tortoise, run_async
from tortoise.contrib.fastapi import register_tortoise

log = logging.getLogger("uvicorn")


TORTOISE_ORM = {
    "connections": {"default": os.environ.get("DATABASE_URL")},
    "apps": {
        "models": {
            "models": ["app.models.tortoise", "aerich.models"],
            "default_connection": "default",
        },
    },
}


def init_db(app: FastAPI) -> None:
    register_tortoise(
        app,
        db_url=os.environ.get("DATABASE_URL"),
        modules={"models": ["app.models.tortoise"]},
        generate_schemas=False,
        add_exception_handlers=True,
    )


async def generate_schema() -> None:
    log.info("Initializing Tortoise...")

    await Tortoise.init(
        db_url=os.environ.get("DATABASE_URL"),
        modules={"models": ["models.tortoise"]},
    )
    log.info("Generating database schema via Tortoise...")
    await Tortoise.generate_schemas()
    await Tortoise.close_connections()


if __name__ == "__main__":
    run_async(generate_schema())
```

**Configuration Explained:**

| Setting | Purpose |
|---------|---------|
| `TORTOISE_ORM` | Configuration dict used by Aerich for migrations |
| `register_tortoise()` | Integrates Tortoise with FastAPI lifecycle |
| `generate_schemas=False` | Disables auto-schema generation (using migrations instead) |
| `add_exception_handlers=True` | Adds proper error handling for DB exceptions |

### 6.2 Aerich Migration Configuration

This configuration file tells Aerich exactly where to find your database settings and where to store the resulting migration files.

**project/pyproject.toml**
```toml
[tool.aerich]
tortoise_orm = "app.db.TORTOISE_ORM"
location = "./migrations"
src_folder = "./."
```

---

## Step 7: Data Models

### 7.1 Tortoise ORM Model

This code defines a database model for storing image predictions and uses `pydantic_model_creator` to automatically convert that database model into a validation schema for your API.

**project/app/models/tortoise.py**
```python
from tortoise import fields, models
from tortoise.contrib.pydantic import pydantic_model_creator


class ImagePrediction(models.Model):
    id = fields.IntField(pk=True)
    filename = fields.CharField(max_length=255)
    image_path = fields.TextField()
    top_prediction = fields.CharField(max_length=255, null=True)
    confidence = fields.FloatField(null=True)
    all_predictions = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    def __str__(self):
        return self.filename

    class Meta:
        table = "imageprediction"


PredictionSchema = pydantic_model_creator(ImagePrediction)
```

**Model Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer (auto) | Primary key, auto-generated |
| `filename` | CharField | Original uploaded filename |
| `image_path` | TextField | Full path to stored image |
| `top_prediction` | CharField | Best classification label (nullable until processed) |
| `confidence` | FloatField | Confidence score between 0 and 1 |
| `all_predictions` | JSONField | Top-5 predictions with labels and scores |
| `created_at` | DatetimeField | Timestamp, auto-set on creation |

### 7.2 Pydantic Schemas

**project/app/models/pydantic.py**
```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class PredictionResponseSchema(BaseModel):
    id: int
    filename: str


class PredictionDetailSchema(BaseModel):
    id: int
    filename: str
    image_path: str
    top_prediction: Optional[str] = None
    confidence: Optional[float] = None
    all_predictions: Optional[List[Dict[str, Any]]] = None
    created_at: str


class PredictionUpdateSchema(BaseModel):
    top_prediction: str
    confidence: float
```

**Schema Purposes:**

| Schema | Purpose |
|--------|---------|
| `PredictionResponseSchema` | POST response (returns id and filename) |
| `PredictionDetailSchema` | GET response (full prediction details) |
| `PredictionUpdateSchema` | PUT request body (manual correction) |

---

## Step 8: ML Classifier Implementation

The core functionality of the application is classifying images using a pre-trained MobileNetV2 neural network.

**project/app/classifier.py**
```python
import os
from PIL import Image
import numpy as np

from app.api import crud

# Skip TensorFlow import during testing
if os.environ.get("TESTING") != "1":
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input,
        decode_predictions,
    )

_model = None


def get_model():
    """Lazy load and cache the ML model (singleton pattern)."""
    global _model
    if _model is None:
        _model = MobileNetV2(weights="imagenet")
    return _model


async def classify_image(image_path: str, prediction_id: int) -> None:
    """Classify an image and update the database with results."""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Run inference
        model = get_model()
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=5)[0]

        # Format results
        all_predictions = [
            {"label": label, "confidence": float(conf)}
            for (_, label, conf) in decoded
        ]
        top_prediction = decoded[0][1]
        confidence = float(decoded[0][2])

        # Update database
        await crud.update(
            prediction_id, top_prediction, confidence, all_predictions
        )
    except Exception as e:
        await crud.update(prediction_id, "Error", 0.0, [{"error": str(e)}])
```

**How It Works:**

| Step | Description |
|------|-------------|
| 1. Image Loading | Opens image file and converts to RGB |
| 2. Preprocessing | Resizes to 224x224 and normalizes pixel values |
| 3. Model Loading | Lazy loads MobileNetV2 (singleton pattern) |
| 4. Inference | Runs prediction through neural network |
| 5. Decoding | Converts model output to human-readable labels |
| 6. Database Update | Stores top-5 predictions in database |

**Key Features:**

| Feature | Purpose |
|---------|---------|
| Conditional Import | Skips TensorFlow when `TESTING=1` for faster tests |
| Singleton Pattern | Loads model once and reuses for performance |
| Background Task | Runs asynchronously without blocking API response |
| Error Handling | Gracefully handles classification failures |

---

## Step 9: RESTful API Routes

### 9.1 Health Check Endpoint

**project/app/api/ping.py**
```python
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings

router = APIRouter()


@router.get("/ping")
async def pong(settings: Settings = Depends(get_settings)):
    return {
        "ping": "pong",
        "environment": settings.environment,
        "testing": settings.testing,
    }
```

This endpoint:
- Verifies the service is running
- Shows the current environment configuration
- Demonstrates FastAPI's dependency injection with `Depends(get_settings)`

### 9.2 CRUD Operations

**project/app/api/crud.py**
```python
from typing import List, Union, Optional, Dict, Any

from app.models.tortoise import ImagePrediction


async def create(filename: str, image_path: str) -> int:
    prediction = ImagePrediction(
        filename=filename,
        image_path=image_path,
        top_prediction=None,
        confidence=None,
        all_predictions=None,
    )
    await prediction.save()
    return prediction.id


async def get(id: int) -> Union[dict, None]:
    prediction = await ImagePrediction.filter(id=id).first().values()
    if prediction:
        return prediction
    return None


async def get_all() -> List:
    predictions = await ImagePrediction.all().values()
    return predictions


async def delete(id: int) -> int:
    prediction = await ImagePrediction.filter(id=id).first().delete()
    return prediction


async def update(
    id: int,
    top_prediction: str,
    confidence: float,
    all_predictions: Optional[List[Dict[str, Any]]]
) -> Union[dict, None]:
    await ImagePrediction.filter(id=id).update(
        top_prediction=top_prediction,
        confidence=confidence,
        all_predictions=all_predictions,
    )
    return await ImagePrediction.filter(id=id).first().values()
```

**CRUD Functions:**

| Function | Operation | Returns |
|----------|-----------|---------|
| `create()` | Create new prediction record | ID of created record |
| `get()` | Read single prediction | Prediction dict or None |
| `get_all()` | Read all predictions | List of prediction dicts |
| `delete()` | Delete a prediction | Deletion count |
| `update()` | Update prediction with ML results | Updated prediction dict |

### 9.3 Prediction API Endpoints

**project/app/api/predictions.py**
```python
import os
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Path, UploadFile

from app.api import crud
from app.classifier import classify_image
from app.models.pydantic import (
    PredictionResponseSchema,
    PredictionDetailSchema,
    PredictionUpdateSchema,
)
from app.models.tortoise import PredictionSchema

router = APIRouter()


@router.post("/", response_model=PredictionResponseSchema, status_code=201)
async def create_prediction(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> PredictionResponseSchema:
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    upload_path = f"/usr/src/app/uploads/{file.filename}"
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Create database record
    prediction_id = await crud.create(file.filename, upload_path)

    # Queue background classification
    background_tasks.add_task(classify_image, upload_path, prediction_id)

    return {"id": prediction_id, "filename": file.filename}


@router.get("/{id}/", response_model=PredictionSchema)
async def read_prediction(id: int = Path(..., gt=0)) -> PredictionSchema:
    prediction = await crud.get(id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.get("/", response_model=List[PredictionSchema])
async def read_all_predictions() -> List[PredictionSchema]:
    return await crud.get_all()


@router.delete("/{id}/", response_model=PredictionResponseSchema)
async def delete_prediction(id: int = Path(..., gt=0)) -> PredictionResponseSchema:
    prediction = await crud.get(id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Delete image file
    if os.path.exists(prediction["image_path"]):
        os.remove(prediction["image_path"])

    await crud.delete(id)
    return {"id": prediction["id"], "filename": prediction["filename"]}


@router.put("/{id}/", response_model=PredictionSchema)
async def update_prediction(
    payload: PredictionUpdateSchema, id: int = Path(..., gt=0)
) -> PredictionSchema:
    prediction = await crud.get(id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    updated = await crud.update(
        id, payload.top_prediction, payload.confidence, None
    )
    return updated
```

**API Endpoints Summary:**

| Method | Endpoint | Description | Response Code |
|--------|----------|-------------|---------------|
| POST | `/predictions/` | Upload image for classification | 201 Created |
| GET | `/predictions/{id}/` | Get a specific prediction | 200 OK |
| GET | `/predictions/` | Get all predictions | 200 OK |
| DELETE | `/predictions/{id}/` | Delete a prediction | 200 OK |
| PUT | `/predictions/{id}/` | Update a prediction manually | 200 OK |

**Key Features:**

| Feature | Description |
|---------|-------------|
| **File Validation** | Ensures uploaded file is an image |
| **Background Tasks** | Classification runs asynchronously after response |
| **Path Validation** | `Path(..., gt=0)` ensures IDs are positive integers |
| **HTTP Exceptions** | Proper 404 responses when resources aren't found |

---

## Step 10: Running the Application

### 10.1 Dependencies

**project/requirements.txt**
```
aerich==0.8.2
aiosqlite==0.19.0
asyncpg==0.30.0
fastapi==0.115.12
gunicorn==22.0.0
httpx==0.28.1
pydantic-settings==2.8.1
tortoise-orm==0.25.0
uvicorn==0.34.1
tensorflow==2.17.0
pillow==10.4.0
numpy==1.26.4
python-multipart==0.0.20
aiofiles==24.1.0
```

### 10.2 Build and Run

```bash
# Build and start the containers
docker-compose up -d --build

# View logs
docker-compose logs -f

# Apply database migrations
docker-compose exec web aerich upgrade

# Create tables if the migration fails
docker-compose exec web python app/db.py

# Verify tables exist
docker-compose exec web-db psql -U postgres -c "\c web_cv_dev" -c "\dt"
```

### 10.3 Stopping the Application

```bash
# Stop and remove containers
docker-compose down

# Stop and remove containers including volumes (clears database)
docker-compose down -v
```

---

## Step 11: Testing with curl

This section demonstrates how to interact with all API endpoints using curl commands.

### 11.1 Health Check Endpoint

**Request:**

```bash
curl -X GET http://localhost:8004/ping
```

**Response:**

```json
{
  "ping": "pong",
  "environment": "dev",
  "testing": false
}
```

### 11.2 Upload an Image for Classification

**Request:**

```bash
curl -X POST http://localhost:8004/predictions/ \
  -F "file=@/path/to/your/image.jpg"
```

**Example with a sample image:**

```bash
curl -X POST http://localhost:8004/predictions/ \
  -F "file=@project/uploads/download.jpeg"
```

**Response:**

```json
{
  "id": 1,
  "filename": "download.jpeg"
}
```

The image is uploaded and classification begins in the background. The response returns immediately with the prediction ID.

### 11.3 Get a Single Prediction

**Request:**

```bash
curl -X GET http://localhost:8004/predictions/1/
```

**Response (before classification completes):**

```json
{
  "id": 1,
  "filename": "download.jpeg",
  "image_path": "/usr/src/app/uploads/download.jpeg",
  "top_prediction": null,
  "confidence": null,
  "all_predictions": null,
  "created_at": "2026-01-21T10:30:00.000000Z"
}
```

**Response (after classification completes):**

```json
{
  "id": 1,
  "filename": "download.jpeg",
  "image_path": "/usr/src/app/uploads/download.jpeg",
  "top_prediction": "golden_retriever",
  "confidence": 0.9234,
  "all_predictions": [
    {"label": "golden_retriever", "confidence": 0.9234},
    {"label": "Labrador_retriever", "confidence": 0.0512},
    {"label": "cocker_spaniel", "confidence": 0.0123},
    {"label": "Irish_setter", "confidence": 0.0089},
    {"label": "tennis_ball", "confidence": 0.0042}
  ],
  "created_at": "2026-01-21T10:30:00.000000Z"
}
```

### 11.4 Get All Predictions

**Request:**

```bash
curl -X GET http://localhost:8004/predictions/
```

**Response:**

```json
[
  {
    "id": 1,
    "filename": "download.jpeg",
    "image_path": "/usr/src/app/uploads/download.jpeg",
    "top_prediction": "golden_retriever",
    "confidence": 0.9234,
    "all_predictions": [...],
    "created_at": "2026-01-21T10:30:00.000000Z"
  },
  {
    "id": 2,
    "filename": "cat.png",
    "image_path": "/usr/src/app/uploads/cat.png",
    "top_prediction": "tabby",
    "confidence": 0.8756,
    "all_predictions": [...],
    "created_at": "2026-01-21T10:35:00.000000Z"
  }
]
```

### 11.5 Update a Prediction (Manual Correction)

If the ML model misclassified an image, you can manually update the prediction:

**Request:**

```bash
curl -X PUT http://localhost:8004/predictions/1/ \
  -H "Content-Type: application/json" \
  -d '{"top_prediction": "labrador_retriever", "confidence": 1.0}'
```

**Response:**

```json
{
  "id": 1,
  "filename": "download.jpeg",
  "image_path": "/usr/src/app/uploads/download.jpeg",
  "top_prediction": "labrador_retriever",
  "confidence": 1.0,
  "all_predictions": [...],
  "created_at": "2026-01-21T10:30:00.000000Z"
}
```

### 11.6 Delete a Prediction

**Request:**

```bash
curl -X DELETE http://localhost:8004/predictions/1/
```

**Response:**

```json
{
  "id": 1,
  "filename": "download.jpeg"
}
```

This removes both the database record and the uploaded image file.

### 11.7 Error Handling Examples

**Invalid file type:**

```bash
curl -X POST http://localhost:8004/predictions/ \
  -F "file=@document.pdf"
```

**Response (400 Bad Request):**

```json
{
  "detail": "File must be an image"
}
```

**Prediction not found:**

```bash
curl -X GET http://localhost:8004/predictions/999/
```

**Response (404 Not Found):**

```json
{
  "detail": "Prediction not found"
}
```

### 11.8 Interactive API Documentation

FastAPI provides automatic interactive documentation. With the application running, visit:

- **Swagger UI**: http://localhost:8004/docs
- **ReDoc**: http://localhost:8004/redoc

These interfaces allow you to explore and test all endpoints directly in your browser.

---

## Conclusion

In this lab, you have successfully built a Dockerized FastAPI environment integrated with PostgreSQL and Tortoise ORM for efficient asynchronous database management. You implemented a MobileNetV2-based image classification system that processes uploads in the background while providing immediate API responses. By leveraging Pydantic models and dependency injection, you established a robust foundation for building and validating RESTful API endpoints. This local setup ensures a scalable and consistent development workflow, ready for testing, CI/CD integration, and cloud deployment.

---

---

# Production ML Pipeline: Image Classification Service on AWS

## Part 2: Testing, Code Quality & AWS EC2 Deployment

In this section, you'll learn about comprehensive testing strategies, code quality tools, CI/CD with GitHub Actions, and deploying the application to AWS EC2.

## Table of Contents

1. [Project Structure for Production](#project-structure-for-production)
2. [Production Dockerfile](#production-dockerfile)
3. [Code Quality Tools](#code-quality-tools)
4. [Testing with Pytest](#testing-with-pytest)
5. [GitHub Actions Workflows](#github-actions-workflows)
6. [AWS EC2 Setup](#aws-ec2-setup)
7. [Deploying the Application](#deploying-the-application)
8. [Verification After Deployment](#verification-after-deployment)

---

## Project Structure for Production

```
image-prediction-serving/
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD pipeline configuration
├── project/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application entry point
│   │   ├── config.py             # Configuration management
│   │   ├── db.py                 # Database connection setup
│   │   ├── classifier.py         # ML model & prediction logic
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── ping.py           # Health check endpoint
│   │   │   ├── predictions.py    # Prediction CRUD endpoints
│   │   │   └── crud.py           # Database operations
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── tortoise.py       # Tortoise ORM models
│   │       └── pydantic.py       # Pydantic schemas
│   ├── db/
│   │   ├── Dockerfile            # PostgreSQL container config
│   │   └── create.sql            # Database initialization
│   ├── migrations/
│   │   └── models/
│   │       └── 0_20260120200937_init.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py           # Pytest fixtures
│   │   ├── test_ping.py          # Health check tests
│   │   └── test_predictions.py   # Prediction endpoint tests
│   ├── uploads/                  # Uploaded image storage
│   ├── Dockerfile                # Development container
│   ├── Dockerfile.prod           # Production container (multi-stage)
│   ├── entrypoint.sh             # Container startup script
│   ├── requirements.txt          # Production dependencies
│   ├── requirements-dev.txt      # Development dependencies
│   ├── pyproject.toml            # Aerich migration config
│   ├── setup.cfg                 # Flake8 & isort configuration
│   ├── .coveragerc               # Test coverage configuration
│   └── .dockerignore             # Docker build exclusions
├── docker-compose.yml            # Multi-container orchestration
└── README.md                     # This documentation
```

**Key Production Files:**

| File | Purpose |
|------|---------|
| `.github/workflows/main.yml` | CI/CD pipeline for automated testing and deployment |
| `Dockerfile.prod` | Multi-stage production build for smaller, secure images |
| `tests/` | Comprehensive test suite with pytest |
| `requirements-dev.txt` | Development tools (linting, testing) |
| `setup.cfg` | Code quality tool configuration |
| `.coveragerc` | Test coverage configuration |

---

## Step 1: Environment Setup

### 1.1 Update the environment

```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Clone the repository

```bash
git clone <repository-url>
cd image-prediction-serving
```

---

## Step 2: Production Dockerfile

The production Dockerfile uses multi-stage builds for smaller, more secure images.

**project/Dockerfile.prod**
```dockerfile
###########
# BUILDER #
###########

FROM python:3.11-slim-bookworm AS builder

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
  && apt-get -y install netcat-openbsd gcc g++ libpq-dev \
     libhdf5-dev libglib2.0-0 libsm6 libxext6 libgomp1 \
  && apt-get clean

# Install python dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# Lint
COPY . .
RUN pip install black==25.1.0 flake8==7.2.0 isort==6.0.1
RUN flake8 .
RUN black --exclude=migrations . --check
RUN isort . --check-only


#########
# FINAL #
#########

FROM python:3.11-slim-bookworm

# Create directory for the app user
RUN mkdir -p /home/app

# Create the app user
RUN addgroup --system app && adduser --system --group app

# Create the appropriate directories
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/uploads
WORKDIR $APP_HOME

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV ENVIRONMENT prod
ENV TESTING 0

# Install system dependencies
RUN apt-get update \
  && apt-get -y install netcat-openbsd gcc libpq5 \
     libhdf5-103-1 libglib2.0-0 libsm6 libxext6 libgomp1 \
  && apt-get clean

# Install python dependencies
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*
RUN pip install "gunicorn==22.0.0"

# Add app
COPY . .

# Chown all the files to the app user
RUN chown -R app:app $APP_HOME

# Change to the app user
USER app

# Run gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app.main:app -k uvicorn.workers.UvicornWorker
```

**Multi-Stage Build Explanation:**

| Stage | Purpose |
|-------|---------|
| **Builder** | Compiles dependencies, runs linting checks |
| **Final** | Contains only runtime dependencies, smaller image |

**Security Features:**

| Feature | Description |
|---------|-------------|
| Non-root user | Runs as `app` user instead of root |
| Minimal dependencies | Only production packages in final image |
| Pre-built wheels | Dependencies compiled in builder stage |

---

## Step 3: Code Quality Tools

### 3.1 Overview

| Tool | Purpose | Command |
|------|---------|---------|
| **Black** | Code formatter | `black .` |
| **Flake8** | Linter (style guide enforcement) | `flake8 .` |
| **isort** | Import sorter | `isort .` |

### 3.2 Black - Code Formatter

Black is an opinionated code formatter that enforces a consistent style.

```bash
# Format all files
docker-compose exec web black .

# Check without modifying (CI mode)
docker-compose exec web black . --check

# Exclude directories
docker-compose exec web black --exclude=migrations .
```

### 3.3 Flake8 - Linter

Flake8 checks code against PEP 8 style guide and finds common errors.

```bash
# Run linting
docker-compose exec web flake8 .

# Show specific error codes
docker-compose exec web flake8 . --show-source
```

**Configuration in setup.cfg:**

```ini
[flake8]
max-line-length = 119
exclude = migrations
```

### 3.4 isort - Import Sorter

isort automatically sorts and organizes imports.

```bash
# Sort imports
docker-compose exec web isort .

# Check without modifying
docker-compose exec web isort . --check-only
```

**Configuration in setup.cfg:**

```ini
[isort]
profile = black
skip = migrations
line_length = 119
```

### 3.5 Running All Quality Checks

```bash
# Run all checks
docker-compose exec web flake8 .
docker-compose exec web black --exclude=migrations . --check
docker-compose exec web isort . --check-only
```

---

## Step 4: Testing with Pytest

### 4.1 Development Dependencies

**project/requirements-dev.txt**
```
black==25.1.0
flake8==7.2.0
isort==6.0.1
pytest==8.3.5
pytest-cov==6.1.1
pytest-xdist==3.6.1

-r requirements.txt
```

### 4.2 Test Configuration

**project/tests/conftest.py**
```python
import os

import pytest
from starlette.testclient import TestClient
from tortoise.contrib.fastapi import register_tortoise

from app.config import Settings, get_settings
from app.main import create_application


def get_settings_override():
    return Settings(testing=True, database_url=os.environ.get("DATABASE_TEST_URL"))


@pytest.fixture(scope="module")
def test_app():
    app = create_application()
    app.dependency_overrides[get_settings] = get_settings_override
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def test_app_with_db():
    app = create_application()
    app.dependency_overrides[get_settings] = get_settings_override
    register_tortoise(
        app,
        db_url=os.environ.get("DATABASE_TEST_URL"),
        modules={"models": ["app.models.tortoise"]},
        generate_schemas=True,
        add_exception_handlers=True,
    )
    with TestClient(app) as test_client:
        yield test_client
```

**Fixtures Explained:**

| Fixture | Purpose |
|---------|---------|
| `test_app` | Basic test client for unit tests (no database) |
| `test_app_with_db` | Test client with database for integration tests |

### 4.3 Health Check Test

**project/tests/test_ping.py**
```python
def test_ping(test_app):
    response = test_app.get("/ping")
    assert response.status_code == 200
    assert response.json() == {
        "environment": "dev",
        "ping": "pong",
        "testing": True
    }
```

### 4.4 Prediction Tests

**project/tests/test_predictions.py**
```python
import pytest
from unittest.mock import AsyncMock

from app.api import crud, predictions


def test_create_prediction(test_app, monkeypatch):
    """Test image upload and prediction creation."""
    async def mock_create(filename, path):
        return 1

    monkeypatch.setattr(crud, "create", mock_create)
    monkeypatch.setattr(predictions, "classify_image", AsyncMock())

    response = test_app.post(
        "/predictions/",
        files={"file": ("test.jpg", b"fake image data", "image/jpeg")}
    )

    assert response.status_code == 201
    assert response.json()["id"] == 1
    assert response.json()["filename"] == "test.jpg"


def test_create_prediction_invalid_file(test_app):
    """Test rejection of non-image files."""
    response = test_app.post(
        "/predictions/",
        files={"file": ("test.txt", b"text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_read_prediction(test_app, monkeypatch):
    """Test retrieving a single prediction."""
    test_data = {
        "id": 1,
        "filename": "test.jpg",
        "image_path": "/uploads/test.jpg",
        "top_prediction": "cat",
        "confidence": 0.95,
        "all_predictions": [{"label": "cat", "confidence": 0.95}],
        "created_at": "2026-01-21T10:00:00Z"
    }

    async def mock_get(id):
        return test_data

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.get("/predictions/1/")
    assert response.status_code == 200
    assert response.json()["top_prediction"] == "cat"


def test_read_prediction_incorrect_id(test_app, monkeypatch):
    """Test 404 for non-existent prediction."""
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.get("/predictions/999/")
    assert response.status_code == 404


def test_read_all_predictions(test_app, monkeypatch):
    """Test retrieving all predictions."""
    test_data = [
        {"id": 1, "filename": "test1.jpg", "top_prediction": "cat"},
        {"id": 2, "filename": "test2.jpg", "top_prediction": "dog"},
    ]

    async def mock_get_all():
        return test_data

    monkeypatch.setattr(crud, "get_all", mock_get_all)

    response = test_app.get("/predictions/")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_delete_prediction(test_app, monkeypatch):
    """Test deleting a prediction."""
    test_data = {
        "id": 1,
        "filename": "test.jpg",
        "image_path": "/tmp/test.jpg",
    }

    async def mock_get(id):
        return test_data

    async def mock_delete(id):
        return 1

    monkeypatch.setattr(crud, "get", mock_get)
    monkeypatch.setattr(crud, "delete", mock_delete)

    response = test_app.delete("/predictions/1/")
    assert response.status_code == 200


def test_update_prediction(test_app, monkeypatch):
    """Test updating a prediction."""
    test_data = {
        "id": 1,
        "filename": "test.jpg",
        "image_path": "/uploads/test.jpg",
        "top_prediction": "dog",
        "confidence": 1.0,
    }

    async def mock_get(id):
        return {"id": 1, "filename": "test.jpg"}

    async def mock_update(id, top_prediction, confidence, all_predictions):
        return test_data

    monkeypatch.setattr(crud, "get", mock_get)
    monkeypatch.setattr(crud, "update", mock_update)

    response = test_app.put(
        "/predictions/1/",
        json={"top_prediction": "dog", "confidence": 1.0}
    )
    assert response.status_code == 200
```

**Key Testing Concepts:**

| Concept | Description |
|---------|-------------|
| `monkeypatch` | Replaces real functions with mocks for isolation |
| `AsyncMock` | Mocks async functions like the classifier |
| Unit tests | Test individual components without database |
| Integration tests | Test with real database connections |

### 4.5 Coverage Configuration

**project/.coveragerc**
```ini
[run]
omit = tests/*
branch = True
```

### 4.6 Running Tests

```bash
# Run all tests
docker-compose exec web python -m pytest

# Run with coverage report
docker-compose exec web python -m pytest --cov=app --cov-report=term-missing

# Run tests in parallel
docker-compose exec web python -m pytest -n auto

# Run specific test file
docker-compose exec web python -m pytest tests/test_predictions.py -v
```

---

## Step 5: GitHub Actions Workflows

GitHub Actions automates testing and deployment on every push.

**.github/workflows/main.yml**
```yaml
name: Continuous Integration and Delivery

on: [push]

env:
  IMAGE: ghcr.io/${{ github.repository }}/classifier

jobs:

  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          ref: main

      - name: Log in to GitHub Packages
        run: echo ${GITHUB_TOKEN} | docker login -u ${GITHUB_ACTOR} --password-stdin ghcr.io
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Pull image
        run: |
          docker pull ${{ env.IMAGE }}:latest || true

      - name: Build image
        run: |
          docker build \
            --cache-from ${{ env.IMAGE }}:latest \
            --tag ${{ env.IMAGE }}:latest \
            --file ./project/Dockerfile.prod \
            "./project"

      - name: Push image
        run: |
          docker push ${{ env.IMAGE }}:latest

  test:
    name: Test Docker Image
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          ref: main

      - name: Log in to GitHub Packages
        run: echo ${GITHUB_TOKEN} | docker login -u ${GITHUB_ACTOR} --password-stdin ghcr.io
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Pull image
        run: |
          docker pull ${{ env.IMAGE }}:latest || true

      - name: Run container
        run: |
          docker run \
            -d \
            --name fastapi-classifier \
            -e PORT=8765 \
            -e ENVIRONMENT=dev \
            -e TESTING=1 \
            -e DATABASE_URL=sqlite://sqlite.db \
            -e DATABASE_TEST_URL=sqlite://sqlite.db \
            -p 5003:8765 \
            ${{ env.IMAGE }}:latest

      - name: Install requirements
        run: docker exec fastapi-classifier pip install -r requirements-dev.txt

      - name: Pytest
        run: docker exec fastapi-classifier python -m pytest .

      - name: Flake8
        run: docker exec fastapi-classifier python -m flake8 .

      - name: Black
        run: docker exec fastapi-classifier python -m black . --check

      - name: isort
        run: docker exec fastapi-classifier python -m isort . --check-only

  deploy:
    name: Deploy to EC2
    runs-on: ubuntu-latest
    needs: [build, test]
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Deploy to EC2
        env:
          EC2_HOST: ${{ secrets.EC2_HOST }}
          EC2_SSH_KEY: ${{ secrets.EC2_SSH_KEY }}
          DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
        run: |
          echo "$EC2_SSH_KEY" > key.pem
          chmod 600 key.pem
          ssh -o StrictHostKeyChecking=no -i key.pem ubuntu@$EC2_HOST << 'EOF'
            cd /home/ubuntu/app
            docker-compose pull
            docker-compose up -d
            docker-compose exec -T web aerich upgrade
          EOF
```

**Workflow Stages:**

| Stage | Description |
|-------|-------------|
| **Build** | Builds Docker image and pushes to GitHub Container Registry |
| **Test** | Runs pytest, flake8, black, and isort checks |
| **Deploy** | SSH to EC2 and deploy (only on main branch) |

---

## Step 6: AWS EC2 Setup

### 6.1 VPC and Networking

1. Create a VPC named `ml-vpc` with IPv4 CIDR block `10.0.0.0/16`
2. Create a public subnet named `ml-subnet` with IPv4 CIDR block `10.0.1.0/24`
3. Create a route table and associate it with the public subnet
4. Create an internet gateway and attach it to the VPC
5. Add a route with destination `0.0.0.0/0` pointing to the internet gateway

### 6.2 Security Group

Create a security group with the following inbound rules:

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| HTTP | 80 | 0.0.0.0/0 | Web traffic |
| Custom TCP | 8000 | 0.0.0.0/0 | FastAPI application |

### 6.3 Launch EC2 Instance

1. Choose Ubuntu 22.04 AMI
2. Choose instance type (t2.medium recommended for TensorFlow)
3. Select the VPC and subnet created above
4. Assign a public IP
5. Select the security group
6. Create or select an SSH key pair

### 6.4 Install Docker on EC2

```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt install docker-compose -y

# Log out and back in
exit
```

---

## Step 7: Deploying the Application

### 7.1 Configure GitHub Secrets

Add these secrets in GitHub repository settings (Settings → Secrets → Actions):

| Secret Name | Value Description |
|-------------|-------------------|
| `EC2_HOST` | Your EC2 public IP |
| `EC2_SSH_KEY` | Contents of your `.pem` file |
| `DB_PASSWORD` | PostgreSQL password |

### 7.2 Manual Deployment

If deploying manually without CI/CD:

```bash
# On your local machine
# Build the production image
docker build -t image-classifier:latest -f project/Dockerfile.prod project/

# Save and compress the image
docker save image-classifier:latest | gzip > image.tar.gz

# Transfer to EC2
scp -i your-key.pem image.tar.gz ubuntu@your-ec2-ip:/home/ubuntu/

# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Load the image
docker load < image.tar.gz

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  web:
    image: image-classifier:latest
    ports:
      - "80:8000"
    environment:
      - PORT=8000
      - DATABASE_URL=postgres://postgres:yourpassword@db:5432/web_cv_prod
      - ENVIRONMENT=prod
    depends_on:
      - db
  db:
    image: postgres:17
    environment:
      - POSTGRES_PASSWORD=yourpassword
      - POSTGRES_DB=web_cv_prod
    volumes:
      - postgres_data:/var/lib/postgresql/data
volumes:
  postgres_data:
EOF

# Start the application
docker-compose up -d

# Wait for database and run migrations
sleep 15
docker-compose exec web aerich upgrade
```

---

## Step 8: Verification After Deployment

### 8.1 Health Check

```bash
curl http://your-ec2-ip/ping
```

**Expected Response:**

```json
{
  "ping": "pong",
  "environment": "prod",
  "testing": false
}
```

### 8.2 Functional Testing

```bash
# Upload an image
curl -X POST http://your-ec2-ip/predictions/ \
  -F "file=@test-image.jpg"

# Get prediction (wait a few seconds for classification)
curl http://your-ec2-ip/predictions/1/

# List all predictions
curl http://your-ec2-ip/predictions/

# Delete prediction
curl -X DELETE http://your-ec2-ip/predictions/1/
```

### 8.3 Monitoring Logs

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# View application logs
docker-compose logs web

# Follow logs in real-time
docker-compose logs -f web
```

### 8.4 Container Health

```bash
docker-compose ps

# Expected output:
# NAME                    STATUS
# app-web-1              Up (healthy)
# app-db-1               Up
```

---

## Troubleshooting

### Database Migration Fails

```bash
# Create tables manually
docker-compose exec web python app/db.py

# Verify tables exist
docker-compose exec web-db psql -U postgres -c "\c web_cv_prod" -c "\dt"
```

### Docker Permission Denied

```bash
sudo usermod -aG docker ubuntu
# Log out and back in
```

### SSH Key Issues

Ensure the entire key is pasted in GitHub Secrets:
```
-----BEGIN RSA PRIVATE KEY-----
...all content...
-----END RSA PRIVATE KEY-----
```

### TensorFlow Memory Issues

If the EC2 instance runs out of memory:
- Use a larger instance type (t2.medium or higher)
- Or add swap space:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Conclusion

In Part 2, you transitioned from local development to production readiness by implementing comprehensive testing with pytest and establishing code quality tools (black, flake8, isort). You created a CI/CD pipeline with GitHub Actions that automates testing and deployment on every push. You mastered multi-stage Docker builds for optimized production images and successfully deployed the ML-powered image classification API to AWS EC2 with proper networking and security configurations.

### Key Takeaways

1. **Testing Strategy**: Unit tests with mocks for isolation, integration tests with database
2. **Code Quality**: Automated formatting and linting ensure consistent code style
3. **CI/CD Pipeline**: GitHub Actions automates the entire build-test-deploy workflow
4. **Production Docker**: Multi-stage builds create smaller, more secure images
5. **AWS Deployment**: Proper VPC, security groups, and EC2 configuration for production

### Next Steps

- Add authentication and authorization
- Implement rate limiting
- Set up monitoring with Prometheus and Grafana
- Configure auto-scaling for handling traffic spikes
- Add Redis for caching predictions
- Implement model versioning for A/B testing
