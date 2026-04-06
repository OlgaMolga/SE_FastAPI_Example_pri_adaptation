from fastapi.testclient import TestClient
from main import app, get_classifier


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    # Корень отдаёт одну строку в JSON как массив из одного элемента.
    assert "FastApi" in str(response.json())


def test_predict_mocked(client: TestClient):
    def fake_classifier(_text: str):
        return [{"label": "POSITIVE", "score": 0.99}]

    app.dependency_overrides[get_classifier] = lambda: fake_classifier
    try:
        response = client.post("/predict/", json={"text": "ok"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["label"] == "POSITIVE"
    assert body["results"][0]["score"] == 0.99


def test_predict_validation(client: TestClient):
    def stub(_t: str):
        return [{"label": "X", "score": 0.1}]

    app.dependency_overrides[get_classifier] = lambda: stub
    try:
        response = client.post("/predict/", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
