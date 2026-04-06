from unittest.mock import MagicMock, patch

from app import fetch_prediction, is_valid_text


def test_is_valid_text():
    assert is_valid_text("a") is True
    assert is_valid_text("  x  ") is True
    assert is_valid_text("") is False
    assert is_valid_text("   ") is False
    assert is_valid_text(None) is False


def test_fetch_prediction():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"label": "NEGATIVE", "score": 0.42}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_http = MagicMock()
    mock_http.post.return_value = mock_response
    mock_http.__enter__.return_value = mock_http
    mock_http.__exit__.return_value = False

    with patch("httpx.Client", return_value=mock_http):
        data = fetch_prediction("http://127.0.0.1:8000/predict/", "  hello  ")

    assert data["results"][0]["label"] == "NEGATIVE"
    assert data["results"][0]["score"] == 0.42
    mock_http.post.assert_called_once_with(
        "http://127.0.0.1:8000/predict/",
        json={"text": "hello"},
    )
