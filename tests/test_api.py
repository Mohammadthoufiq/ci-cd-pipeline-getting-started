from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json=[0.1]*30)
    assert response.status_code == 200