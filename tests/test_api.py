from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PATIENT = {
    "gender": "Female",
    "age": 54,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 27.3,
    "HbA1c_level": 6.6,
    "blood_glucose_level": 140,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_index_serves_form():
    res = client.get("/")
    assert res.status_code == 200
    assert "<form" in res.text


def test_predict_returns_probability():
    body = client.post("/predict", json=PATIENT).json()
    assert 0 <= body["probability"] <= 1
    assert body["diabetes"] == (body["probability"] >= 0.5)


def test_high_risk_patient_scores_higher():
    healthy = {**PATIENT, "age": 25, "HbA1c_level": 4.8, "blood_glucose_level": 90, "bmi": 22}
    at_risk = {**PATIENT, "age": 70, "HbA1c_level": 8.8, "blood_glucose_level": 260, "hypertension": 1}
    low = client.post("/predict", json=healthy).json()["probability"]
    high = client.post("/predict", json=at_risk).json()["probability"]
    assert high > low


def test_rejects_invalid_input():
    assert client.post("/predict", json={**PATIENT, "gender": "unknown"}).status_code == 422
    assert client.post("/predict", json={**PATIENT, "bmi": -5}).status_code == 422
