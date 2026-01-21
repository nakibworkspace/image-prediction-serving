# Production ML Pipeline: Image Classification Service on AWS Part 1

In this hands-on lab, you'll learn how to set up a local development environment for an image classification service with Python, FastAPI, TensorFlow, and Docker. The service will be exposed via a RESTful API that allows users to upload images and receive AI-generated predictions using MobileNetV2.

![dia1](https://raw.githubusercontent.com/poridhiEng/lab-asset/8fc17ca40d2fc6ec516f688becb7e7a0e1fa4e32/MLOps%20Lab/API%20Labs/Lab%2013/images/dia1.svg)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Docker Setup](#docker-setup)
4. [Database Configuration](#database-configuration)
5. [FastAPI Application Setup](#fastapi-application-setup)
6. [Configuration Management](#configuration-management)
7. [Database Integration with SQLAlchemy](#database-integration-with-tortoise-orm)
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
| **SQLAlchemy** | Async ORM for database operations |
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
git clone -b lab1 <repository-url>
cd image-prediction-serving
```


## Step 2: Docker Setup

### 2.1 Docker Compose Configuration

This `docker-compose.yml` defines a two-container environment consisting of a FastAPI/Python web application with TensorFlow and a PostgreSQL database.

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

* `build: ./project`: Builds the web service from the project directory
* `--reload`: Enables hot reloading during development
* `volumes`: Mounts local code for live updates
* `ports: 8004:8000`: Maps container port 8000 to host port 8004
* `depends_on`: Ensures database starts before web service

### 2.2 Web Application Dockerfile
This Dockerfile builds a Python 3.11 environment that installs essential system dependencies and project libraries. It then copies your source code, sets up an upload directory, and configures an entrypoint script to manage the container's startup process.

**project/Dockerfile**
```dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
  && apt-get -y install netcat-openbsd gcc g++ libpq-dev \
     libhdf5-dev libglib2.0-0 libsm6 libxext6 libgomp1 \
  && apt-get clean

RUN pip install --upgrade pip setuptools wheel
COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /usr/src/app/uploads

COPY . .

COPY ./entrypoint.sh .
RUN chmod +x /usr/src/app/entrypoint.sh

ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
```

### 2.3 Container Entrypoint Script
This script acts as a service gatekeeper, using netcat to pause execution until the PostgreSQL database is reachable on port 5432. Once the connection is confirmed, it uses exec "$@" to hand over control to the container's main command (like starting a FastAPI server).

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

---

## Step 3: Database Configuration

### 3.1 PostgreSQL Dockerfile
This Dockerfile extends the official PostgreSQL 17 image by automatically executing the create.sql script to initialize the database schema or data during the container's first startup.

**project/db/Dockerfile**
```dockerfile
FROM postgres:17

ADD create.sql /docker-entrypoint-initdb.d
```

### 3.2 Database Initialization Script
This SQL script initializes two separate databases, web_cv_dev and web_cv_test, to isolate your development and testing environments within the PostgreSQL container.

**project/db/create.sql**
```sql
CREATE DATABASE web_cv_dev;
CREATE DATABASE web_cv_test;
```

---

## Step 4: FastAPI Application Setup

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

---

## Step 5: Configuration Management
This Python module defines a pydantic-based configuration system that pulls environment variables into a structured Settings object, using @lru_cache() to ensure settings are loaded efficiently only once.

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

---

## Step 6: Database Integration with SQLAlchemy
This module configures the Tortoise ORM integration for FastAPI, defining the database connection settings and a helper function to register models and exception handlers during the application's startup.

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
```

---

## Step 7: Data Models

### 7.1 SQLAlchemy ORM Model
This defines a ORM model that stores image metadata and AI classification results in the database, automatically generating a Pydantic schema for easy data validation and API serialization.

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

### 7.2 Pydantic Schemas
This module defines Pydantic schemas used for data validation and API responses, separating the data structure for basic list views, detailed result objects, and update payloads.

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

---

## Step 8: ML Classifier Implementation
This module implements an asynchronous image classification pipeline using MobileNetV2, which resizes uploaded images to 224*224 pixels and extracts the top 5 predictions to update the database record. It uses a singleton pattern for the model to ensure the heavy neural network is only loaded into memory once.

**project/app/classifier.py**
```python
import os
from PIL import Image
import numpy as np

from app.api import crud

if os.environ.get("TESTING") != "1":
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input,
        decode_predictions,
    )

_model = None


def get_model():
    global _model
    if _model is None:
        _model = MobileNetV2(weights="imagenet")
    return _model


async def classify_image(image_path: str, prediction_id: int) -> None:
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        model = get_model()
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=5)[0]

        all_predictions = [
            {"label": label, "confidence": float(conf)}
            for (_, label, conf) in decoded
        ]
        top_prediction = decoded[0][1]
        confidence = float(decoded[0][2])

        await crud.update(prediction_id, top_prediction, confidence, all_predictions)
    except Exception as e:
        await crud.update(prediction_id, "Error", 0.0, [{"error": str(e)}])
```

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

### 9.2 CRUD Operations
This module provides an abstraction layer for database operations, handling the creation, retrieval, and updating of image prediction records using Tortoise ORM.


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

![dia3](https://raw.githubusercontent.com/poridhiEng/lab-asset/c9cdaf4c749baa8f606bb23c4e7515ed3b1cd485/MLOps%20Lab/API%20Labs/Lab%2011/images/dia3.svg)

| Function | Operation | Returns |
|----------|-----------|---------|
| `post()` | Create new summary | ID of created record |
| `get()` | Read single summary | Summary dict or None |
| `get_all()` | Read all summaries | List of summary dicts |
| `delete()` | Delete a summary | Deletion count |
| `put()` | Update a summary | Updated summary dict or None |

### 9.3 Prediction API Endpoints
This module defines the API routing logic for the prediction service, managing file uploads, background processing, and resource management.

**project/app/api/predictions.py**
```python
import os
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Path, UploadFile

from app.api import crud
from app.classifier import classify_image
from app.models.pydantic import PredictionResponseSchema, PredictionUpdateSchema
from app.models.tortoise import PredictionSchema

router = APIRouter()


@router.post("/", response_model=PredictionResponseSchema, status_code=201)
async def create_prediction(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> PredictionResponseSchema:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    upload_path = f"/usr/src/app/uploads/{file.filename}"
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    prediction_id = await crud.create(file.filename, upload_path)
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

    updated = await crud.update(id, payload.top_prediction, payload.confidence, None)
    return updated
```

**Prediction Flow**

![dia3](https://raw.githubusercontent.com/poridhiEng/lab-asset/0c6d1117f8858278da99909f10c78c98a0cdbd29/MLOps%20Lab/API%20Labs/Lab%2013/images/dia3.svg)


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
docker compose up -d --build

# View logs
docker compose logs -f

# Apply database migrations
docker compose exec web aerich upgrade

# Create tables if the migration fails
docker compose exec web python app/db.py

# Verify tables exist
docker compose exec web-db psql -U postgres -c "\c web_cv_dev" -c "\dt"
```

Check if the containers are running:
```bash
docker ps
```

![img1](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img1.png?raw=true)

### 10.3 Stopping the Application

```bash
docker compose down

# Including volumes
docker compose down -v
```

---

## Step 11: Testing and Verifications

### 11.1 Interactive API Documentation

Expose the UI using Load Balancer by getting the VM IP
```bash
ip addr show wt0
```

![img3](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img3.png?raw=true)

use the port 8004
![img4](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img4.png?raw=true)

Output:
![img5](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img5.png?raw=true)

**Prediction in Swagger UI**
* Uploaded a cat image and performed a POST request

![img6](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img6.png?raw=true)

* In GET execution the output is shown after 2-3 seconds

![img7](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img7.png?raw=true)

The predictions are visible

* Let's upload another image of a dog

![img8](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img8.png?raw=true)

* GET execution

![img9](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img9.png?raw=true)


### 11.2 Health Check

```bash
curl -X GET http://localhost:8004/ping
```

![img2](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img2.png?raw=true)

### 11.3 Upload an Image

```bash
curl -X POST http://localhost:8004/predictions/ \
  -F "file=@/path/to/your/image.jpg"
```

### 11.4 Get a Prediction

```bash
curl -X GET http://localhost:8004/predictions/1/
```

### 11.5 Get All Predictions

```bash
curl -X GET http://localhost:8004/predictions/
```

### 11.6 Update a Prediction

```bash
curl -X PUT http://localhost:8004/predictions/1/ \
  -H "Content-Type: application/json" \
  -d '{"top_prediction": "labrador_retriever", "confidence": 1.0}'
```

### 11.7 Delete a Prediction

```bash
curl -X DELETE http://localhost:8004/predictions/1/
```

**Curl prediction output**
![img10](https://github.com/poridhiEng/lab-asset/blob/main/MLOps%20Lab/API%20Labs/Lab%2013/images/img10.png?raw=true)

## Conclusion

In this lab, you have successfully built a Dockerized FastAPI environment integrated with PostgreSQL and Tortoise ORM for efficient asynchronous database management. You implemented a MobileNetV2-based image classification system that processes uploads in the background while providing immediate API responses. This local setup ensures a scalable and consistent development workflow, ready for testing, CI/CD integration, and cloud deployment in Part 2.
