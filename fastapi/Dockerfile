FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py run.py ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Слушаем все интерфейсы — нужно для контейнера; переменные из run.py здесь не используются
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
