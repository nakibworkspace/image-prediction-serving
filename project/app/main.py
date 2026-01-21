import logging

from fastapi import FastAPI

from app.api import ping, predictions
from app.db import engine
from app.models.models import Base

log = logging.getLogger("uvicorn")


def create_application() -> FastAPI:
    application = FastAPI(
        title="Image Classification API",
        description="Upload images and get ML-powered predictions using MobileNetV2",
        version="1.0.0",
    )

    application.include_router(ping.router)
    application.include_router(predictions.router, prefix="/predictions", tags=["predictions"])

    return application


app = create_application()


@app.on_event("startup")
def startup():
    log.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    log.info("Database tables created!")