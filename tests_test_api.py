from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Загрузите изображение" in response.text

def test_history_page():
    response = client.get("/history/")
    assert response.status_code == 200
    assert "История предсказаний" in response.text

def test_predict_without_file():
    response = client.post("/predict/")
    assert response.status_code == 422  # FastAPI: ошибка валидации файла