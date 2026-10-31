import importlib.util

import pytest

HAS_MODEL2VEC = importlib.util.find_spec("model2vec") is not None

def test_default_backend_is_lexical(client):
    response = client.post("/ask", json={"question": "Когда дедлайн отправки проектов?"})
    assert response.status_code == 200
    assert response.json()["retrieval"] == "lexical"

def test_unknown_backend_is_rejected(client):
    response = client.post("/ask", json={"question": "Когда дедлайн отправки?", "retrieval": "faiss"})
    assert response.status_code == 422

def test_health_reports_backends(client):
    body = client.get("/health").json()
    assert body["retrieval_backends"] == ["lexical", "semantic", "hybrid"]
    assert isinstance(body["semantic_available"], bool)

@pytest.mark.skipif(
    HAS_MODEL2VEC, reason="model2vec установлен — проверяем успешный путь ниже"
)
def test_semantic_without_dependency_returns_hint(client):
    response = client.post(
        "/ask", json={"question": "Можно ли прийти одному без команды?", "retrieval": "semantic"}
    )
    assert response.status_code == 503
    assert "model2vec" in response.json()["detail"]

@pytest.mark.skipif(not HAS_MODEL2VEC, reason="нет model2vec: pip install -r requirements-semantic.txt")
def test_hybrid_finds_paraphrase(client):
    # Слова «страховка» в документе нет — там «полис страхования». Лексика с обрезкой промахивается.
    response = client.post(
        "/ask", json={"question": "Нужна ли страховка для участия?", "retrieval": "hybrid"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval"] == "hybrid"
    assert "08_support_and_safety.md" in body["sources"]
