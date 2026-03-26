import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

load_dotenv(Path(__file__).resolve().parent / ".env")


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class Item(BaseModel):
    text: str


def _build_classifier():
    model_name = os.getenv("SENTIMENT_MODEL", "").strip() or None
    if model_name:
        return pipeline("sentiment-analysis", model=model_name)
    return pipeline("sentiment-analysis")


app = FastAPI()
classifier = _build_classifier()


@app.get("/")
def root():
    return {"FastApi service started!"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/{text}")
def get_params(text: str):
    return classifier(text)


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)
