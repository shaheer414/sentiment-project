"""
tests/test_api.py
Basic smoke tests for the FastAPI service and text cleaning utility.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.data_loader import clean_text


def test_clean_text_strips_html():
    raw = "This movie was <br/>GREAT</br>!"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert "great" in cleaned


def test_root_endpoint():
    from deployment.app import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "endpoints" in response.json()


def test_health_endpoint():
    from deployment.app import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
