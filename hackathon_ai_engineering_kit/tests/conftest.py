"""Тесты не должны зависеть от сети и ключа: всегда работаем в demo-режиме.

Без этого `test_ask_returns_expected_shape` при заполненном `.env`
ходит в реальный провайдер: медленно, платно и падает без интернета.
"""
import pytest

from app import app


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client
