"""Веб-интерфейс (Streamlit) для вызова FastAPI-сервиса анализа тональности."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").strip()
PREDICT_PATH = "/predict/"
REQUEST_TIMEOUT = 120.0    

def fetch_prediction(url: str, text: str):
    """Делает HTTP-запрос к API"""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(url, json={"text": text.strip()})
        response.raise_for_status()
        return response.json()

def handle_request(url: str, text: str):
    """Обработка ошибок при запросе"""
    try:
        return fetch_prediction(url, text)
    except httpx.HTTPStatusError as exc:
        st.error(f"HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        st.error(
            f"Не удалось достучаться до API ({exc!s}). "
            "Проверьте, что сервис запущен и URL верный."
        )
    return None

def render_prediction(data):
    """Отображение результата (метрики; формат API: {\"results\": [{label, score}, ...]})."""
    rows = None
    if isinstance(data, dict):
        inner = data.get("results")
        if isinstance(inner, list) and inner:
            rows = inner
    elif isinstance(data, list) and data:
        rows = data

    if rows:
        row = rows[0]
        if isinstance(row, dict):
            label = row.get("label")
            score = row.get("score")
            if label is not None:
                st.metric("Метка", str(label))
            if score is not None:
                st.metric("Уверенность", f"{float(score):.4f}")
            return

    if isinstance(data, dict):
        st.json(data)
        return
    st.write(data)


def render_sidebar():
    """UI сайдбара"""
    with st.sidebar:
        api_base = st.text_input("Базовый URL API", value=DEFAULT_API_BASE).strip()
        st.markdown(f"Запрос: `POST {PREDICT_PATH}` · тело: `{{\"text\": \"...\"}}`")
        st.caption("По умолчанию из переменной `API_BASE_URL` в `.env` (см. `.env.example`).")
    return api_base

def is_valid_text(text: str | None) -> bool:
    """Проверка текста"""
    return bool(text and text.strip())

def main() -> None:
    st.set_page_config(
        page_title="Тональность текста",
        page_icon="💬",
        layout="centered",
    )
    st.title("Анализ тональности")
    st.caption(
        "Сначала запустите API (см. корневой ReadMe.MD). "
        "Только Streamlit — см. README.md в этой папке; URL API задаётся в .env или в поле ниже."
    )

    api_base = render_sidebar()

    text = st.text_area(
        "Текст для анализа",
        height=140,
        placeholder="Например: I love this project!",
    )

    submitted = st.button("Отправить", type="primary")

    if submitted and not is_valid_text(text):
        st.warning("Введите непустой текст.")

    if submitted and is_valid_text(text):
        url = api_base.rstrip("/") + PREDICT_PATH

        payload = handle_request(url, text)
        if payload is None:
            return

        st.success("Ответ сервера")
        render_prediction(payload)
        with st.expander("Сырой JSON"):
            st.json(payload)


if __name__ == "__main__":
    main()
