from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "diabetes_model.joblib"
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
# Decision threshold for a positive prediction. Lower than 0.5 so fewer diabetic
# patients are missed; must match THRESHOLD in train.py.
THRESHOLD = 0.25

app = FastAPI(
    title="Diabetes Detection API",
    description="Predicts diabetes risk with a soft-voting ensemble "
                "(Decision Tree + XGBoost + Logistic Regression).",
    version="1.0.0",
)
model = joblib.load(MODEL_PATH)


class Patient(BaseModel):
    gender: Literal["Female", "Male", "Other"]
    age: float = Field(ge=0, le=120, examples=[54])
    hypertension: Literal[0, 1] = Field(examples=[0])
    heart_disease: Literal[0, 1] = Field(examples=[0])
    smoking_history: Literal["No Info", "current", "ever", "former", "never", "not current"]
    bmi: float = Field(gt=0, le=100, examples=[27.3])
    HbA1c_level: float = Field(gt=0, le=20, examples=[6.6])
    blood_glucose_level: float = Field(gt=0, le=500, examples=[140])


class Prediction(BaseModel):
    diabetes: bool
    probability: float


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient) -> Prediction:
    features = pd.DataFrame([patient.model_dump()])
    probability = float(model.predict_proba(features)[0, 1])
    return Prediction(diabetes=probability >= THRESHOLD, probability=round(probability, 4))
