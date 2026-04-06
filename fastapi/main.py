import os
import logging
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from transformers import pipeline

# 1. Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent / ".env")


# --- Схемы данных ---

class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class PredictRequest(BaseModel):
    # Валидация входа: ограничение длины текста
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Текст для анализа тональности",
        examples=["I love this project!"]
    )


class SentimentItem(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    results: list[SentimentItem]


# --- Логика ---

def _to_predict_response(raw) -> PredictResponse:
    if not raw:
        return PredictResponse(results=[])

    items = []
    for row in raw:
        if isinstance(row, dict) and "label" in row and "score" in row:
            items.append(SentimentItem(label=str(row["label"]), score=float(row["score"])))
    return PredictResponse(results=items)


def _build_classifier():
    model_name = os.getenv("SENTIMENT_MODEL", "").strip() or None
    try:
        if model_name:
            return pipeline("sentiment-analysis", model=model_name)
        return pipeline("sentiment-analysis")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


# --- Инициализация API ---

app = FastAPI(
    title="Sentiment Analysis API",
    description="API для классификации тональности текста с использованием Transformers",
    version="1.0.0"
)

classifier = _build_classifier()


# --- Middleware для наблюдаемости (Correlation ID) ---

@app.middleware("http")
async def add_process_time_and_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start_time = time.time()

    # Обработка запроса
    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"

    return response


# --- Эндпоинты ---

@app.get("/", tags=["System"])
def root():
    return {"message": "FastApi service started!"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return HealthResponse(status="ok")


@app.post(
    "/predict/",
    response_model=PredictResponse,
    tags=["ML Model"],
    summary="Анализ тональности",
    response_description="Список меток тональности с оценками уверенности"
)
def predict(item: PredictRequest):
    # Наблюдаемость: логируем длину текста и начало запроса
    logger.info(f"Processing request. Text length: {len(item.text)} characters.")

    if classifier is None:
        logger.error("Classifier is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is currently unavailable"
        )

    try:
        # Обработка сбоев модели
        raw = classifier(item.text)

        if not raw:
            logger.warning("Model returned empty response")
            raise HTTPException(
                status_code=status.HTTP_204_NO_CONTENT,
                detail="Model produced no results"
            )

        return _to_predict_response(raw)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Явный 500/503 без сырых traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during model inference"
        )