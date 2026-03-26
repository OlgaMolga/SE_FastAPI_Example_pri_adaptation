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
    """Длеает HTTP-запрос к API"""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(url, json={"text": text.strip()})
        response.raise_for_status()
        return response.json()

def _format_prediction(data: object) -> None:
    """Показывает ответ Hugging Face pipeline (список словарей с label/score)."""
    if isinstance(data, list) and data:
        row = data[0]
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

    with st.sidebar:
        api_base = st.text_input("Базовый URL API", value=DEFAULT_API_BASE).strip()
        st.markdown(f"Запрос: `POST {PREDICT_PATH}` · тело: `{{\"text\": \"...\"}}`")
        st.caption("По умолчанию из переменной `API_BASE_URL` в `.env` (см. `.env.example`).")

    text = st.text_area(
        "Текст для анализа",
        height=140,
        placeholder="Например: I love this project!",
    )

    submitted = st.button("Отправить", type="primary")

    if submitted and not (text or "").strip():
        st.warning("Введите непустой текст.")

    if submitted and (text or "").strip():
        url = api_base.rstrip("/") + PREDICT_PATH
        try:
            payload = fetch_prediction(url, text)
        except httpx.HTTPStatusError as exc:
            st.error(f"HTTP {exc.response.status_code}")
            return
        except httpx.RequestError as exc:
            st.error(
                f"Не удалось достучаться до API ({exc!s}). "
                "Проверьте, что сервис запущен и URL верный."
            )
            return

        st.success("Ответ сервера")
        _format_prediction(payload)
        with st.expander("Сырой JSON"):
            st.json(payload)


if __name__ == "__main__":
    main()
