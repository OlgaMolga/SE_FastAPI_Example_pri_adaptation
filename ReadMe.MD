# SE FastAPI Example

Два сервиса в разных папках и с **разными** `requirements.txt`:


| Часть    | Папка        | Зависимости                  |
| -------- | ------------ | ---------------------------- |
| Бэкенд   | `fastapi/`   | `fastapi/requirements.txt`   |
| Фронтенд | `streamlit/` | `streamlit/requirements.txt` |


Переменные окружения: в каждой папке есть `**.env.example`** — скопируйте в `**.env**` и при необходимости поправьте.

## Быстрый старт: API и веб вместе

```bash
python -m venv .venv
```

**Windows:** `.venv\Scripts\activate`  
**Linux / macOS:** `source .venv/bin/activate`

После активации venv установите **оба** списка зависимостей **одной командой**:

```bash
pip install -r fastapi/requirements.txt -r streamlit/requirements.txt
```

Для **тестов и линтера** ( pytest, ruff ) добавьте `requirements-dev.txt`:

```bash
pip install -r fastapi/requirements.txt -r streamlit/requirements.txt -r requirements-dev.txt
pytest
ruff check fastapi streamlit tests
```

Опционально: `cp fastapi/.env.example fastapi/.env` и `cp streamlit/.env.example streamlit/.env` (на Windows — `copy`). В `**streamlit/.env**` у `**API_BASE_URL**` должны совпадать хост и порт с `**API_HOST**` и `**API_PORT**` в `**fastapi/.env**`.

1. **API** (первый терминал), из **корня** репозитория:
  ```bash
   python fastapi/run.py
  ```
   Хост, порт и режим reload задаются в `**fastapi/.env**`. Документация OpenAPI: обычно [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. **Streamlit** (второй терминал):
  ```bash
   streamlit run streamlit/app.py
  ```

URL API для фронта: `**streamlit/.env**` (`API_BASE_URL`) или поле в боковой панели.

## Docker (API + Streamlit)

Из **корня** репозитория:

```bash
docker compose up --build
```

- API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Streamlit: [http://127.0.0.1:8501](http://127.0.0.1:8501)

В compose для фронта задано `API_BASE_URL=http://api:8000` (имя сервиса `api`). Образы тянут `torch` и `transformers` — первая сборка может занять много времени и места на диске.

## Только бэкенд

См. `**fastapi/README.md**`.

## Только фронтенд

См. `**streamlit/README.md**`.