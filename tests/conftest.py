import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from main import app

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
