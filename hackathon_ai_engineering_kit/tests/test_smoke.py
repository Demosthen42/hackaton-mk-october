from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ask_returns_expected_shape():
    response = client.post("/ask", json={"question": "Во сколько стоп-кодинг?"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str)
    assert isinstance(body["sources"], list)
    assert body["mode"] in {"demo", "llm"}
