from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_liveness_health_check():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
