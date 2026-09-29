"""Train the diabetes detection ensemble and save it to model/diabetes_model.joblib.

Ported from the thesis notebook "Diabetes Detection Model.ipynb", with one change:
the model is trained on the diabetes prediction dataset alone (100k patients).
The notebook concatenated three unrelated datasets side by side, which mixed rows
from different patients and leaked the other datasets' labels into the features.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "diabetes_prediction_dataset.csv"
MODEL_PATH = ROOT / "model" / "diabetes_model.joblib"

CATEGORICAL = ["gender", "smoking_history"]
NUMERIC = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "diabetes"

# Same category order the notebook's LabelEncoder produced (alphabetical).
GENDERS = ["Female", "Male", "Other"]
SMOKING = ["No Info", "current", "ever", "former", "never", "not current"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype={"blood_glucose_level": float})
    # Notebook cleaning: values more than 3 std above the mean are replaced with the mean.
    for col in ["bmi", "HbA1c_level", "blood_glucose_level"]:
        mean, std = df[col].mean(), df[col].std()
        df.loc[df[col] > mean + 3 * std, col] = mean
    return df


def build_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("cat", OrdinalEncoder(categories=[GENDERS, SMOKING]), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    ensemble = VotingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=10, random_state=42)),
            ("xgb", XGBClassifier(random_state=42, eval_metric="logloss")),
            ("lr", LogisticRegression(random_state=42, max_iter=1000)),
        ],
        voting="soft",
    )
    return Pipeline([("preprocess", preprocess), ("ensemble", ensemble)])


def main() -> None:
    df = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    model = build_model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
