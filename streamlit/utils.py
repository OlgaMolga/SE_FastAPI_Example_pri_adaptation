from __future__ import annotations

from urllib.parse import urljoin, urlparse


def normalize_base_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url.rstrip("/") + "/"


def build_url(base: str, path: str) -> str | None:
    base = normalize_base_url(base)
    if not is_valid_url(base):
        return None
    return urljoin(base, path.lstrip("/"))


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def extract_fastapi_error(response) -> str:
    try:
        data = response.json()
        if "detail" in data:
            return str(data["detail"])
    except Exception:
        pass
    return f"HTTP {response.status_code}"