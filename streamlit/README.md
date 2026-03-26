# Веб-интерфейс (Streamlit)

Клиент для API анализа тональности: `POST /predict/` с телом `{"text": "..."}`.

Зависимости: **`requirements.txt` в этой папке**.

**Конфигурация:** скопируйте **`.env.example`** → **`.env`** и при необходимости задайте **`API_BASE_URL`** (базовый адрес API без завершающего `/`). Если `.env` нет, используется `http://127.0.0.1:8000`.

---

## Только Streamlit (без тяжёлых пакетов бэкенда)

Подходит, если API уже запущен **в другом окружении** или на другой машине.

Из каталога **`streamlit/`**:

```bash
python -m venv .venv
```

**Windows:** `.venv\Scripts\activate`  
**Linux / macOS:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
copy .env.example .env
```

(На Linux/macOS: `cp .env.example .env` — при необходимости поправьте `API_BASE_URL`.)

**Запуск из папки `streamlit/`:**

```bash
streamlit run app.py
```

**Запуск из корня репозитория:**

```bash
streamlit run streamlit/app.py
```

---

## Полный проект: бэкенд + фронт одним окружением

Из корня:

```bash
pip install -r fastapi/requirements.txt -r streamlit/requirements.txt
```

Дальше — **ReadMe.MD** в корне проекта.
