# Diabetes Detection API

[![Tests](https://github.com/DidiUkomadu/diabetes-detection-api/actions/workflows/tests.yml/badge.svg)](https://github.com/DidiUkomadu/diabetes-detection-api/actions/workflows/tests.yml)

A machine learning model from my MSc Data Science thesis that predicts whether a patient has diabetes. It is served with FastAPI, containerized with Docker, and deployed on Render.

**Live demo:** https://diabetes-detection-api-hc2z.onrender.com


## Model

- **Algorithm:** soft-voting ensemble of Decision Tree, XGBoost and Logistic Regression
- **Data:** [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset), 100,000 patients
- **Features:** gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level
- **Preprocessing:** BMI, HbA1c and glucose values more than 3 standard deviations above the mean are replaced with the mean; categories are encoded and numeric features standardized inside a scikit-learn `Pipeline`, so the API accepts raw values.

### Results

Test set of 20,000 patients (80/20 stratified split, `random_state=42`). Run `python train.py` to reproduce.

| Class        | Precision | Recall | F1   | Support |
|--------------|-----------|--------|------|---------|
| No diabetes  | 0.97      | 1.00   | 0.98 | 18,300  |
| Diabetes     | 0.97      | 0.67   | 0.79 | 1,700   |

**Accuracy:** 0.970

|                       | Predicted no | Predicted yes |
|-----------------------|--------------|---------------|
| **Actual no**         | 18,266       | 34            |
| **Actual yes**        | 560          | 1,140         |

Only 8.5% of patients have diabetes, so accuracy alone is misleading. The model is precise when it predicts diabetes, but it misses about a third of diabetic patients (recall 0.67). For screening, lowering the decision threshold below 0.5 would trade some precision for higher recall.

## API

| Method | Path       | Description                      |
|--------|------------|----------------------------------|
| GET    | `/`        | Web form for trying the model    |
| POST   | `/predict` | Returns the prediction as JSON   |
| GET    | `/health`  | Health check                     |
| GET    | `/docs`    | Interactive Swagger docs         |

```bash
curl -X POST https://diabetes-detection-api-hc2z.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","age":54,"hypertension":0,"heart_disease":0,
       "smoking_history":"never","bmi":27.3,"HbA1c_level":6.6,"blood_glucose_level":140}'
# {"diabetes": false, "probability": 0.0878}
```

## Run locally

With Docker:

```bash
docker build -t diabetes-detection-api .
docker run -p 8000:8000 diabetes-detection-api
# open http://localhost:8000
```

Without Docker:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
python train.py
uvicorn app.main:app --reload
pytest
```

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, choose **New > Blueprint** and select the repository. Render reads `render.yaml` and builds the `Dockerfile`.
3. Every push to `main` redeploys automatically.

The Docker build trains the model in a separate stage, so no model file is stored in git.

## Project structure

```
app/main.py            FastAPI app
app/static/index.html  Web form
train.py               Training script (writes model/diabetes_model.joblib)
data/                  Training data
tests/                 API tests
Dockerfile             Multi-stage build: train, then serve
render.yaml            Render blueprint
```

_For educational use only. Not a medical diagnosis._
