import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, build_feature_matrix, latest_team_snapshots
from src.load_data import load_matches

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TEST_SEASONS = {"2324"}
LABELS = ["H", "D", "A"]


def _encode(y: pd.Series) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(LABELS)}
    return y.map(mapping).to_numpy()


def _form_baseline_probs(X: pd.DataFrame) -> np.ndarray:
    probs = []
    for _, row in X.iterrows():
        diff = row["ppg_diff_l5"]
        if diff > 0.4:
            probs.append([0.55, 0.25, 0.20])
        elif diff < -0.4:
            probs.append([0.20, 0.25, 0.55])
        else:
            probs.append([0.30, 0.40, 0.30])
    return np.array(probs)


def train_and_evaluate() -> dict:
    matches = load_matches()
    features = build_feature_matrix(matches)

    train_mask = ~features["Season"].isin(TEST_SEASONS)
    test_mask = features["Season"].isin(TEST_SEASONS)

    X_train = features.loc[train_mask, FEATURE_COLUMNS]
    y_train = _encode(features.loc[train_mask, "FTR"])
    X_test = features.loc[test_mask, FEATURE_COLUMNS]
    y_test = _encode(features.loc[test_mask, "FTR"])

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_test)
    dummy_log_loss = log_loss(y_test, dummy_probs, labels=[0, 1, 2])

    form_probs = _form_baseline_probs(features.loc[test_mask])
    form_log_loss = log_loss(y_test, form_probs, labels=[0, 1, 2])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    ml_probs = pipeline.predict_proba(X_test)
    ml_preds = pipeline.predict(X_test)
    ml_log_loss = log_loss(y_test, ml_probs, labels=[0, 1, 2])
    ml_accuracy = accuracy_score(y_test, ml_preds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "model.joblib")

    snapshots, last_date = latest_team_snapshots(matches)
    joblib.dump(snapshots, MODELS_DIR / "team_snapshots.joblib")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "train_seasons": sorted(features.loc[train_mask, "Season"].unique().tolist()),
        "test_seasons": sorted(TEST_SEASONS),
        "last_match_date": last_date.strftime("%Y-%m-%d"),
        "metrics": {
            "dummy_log_loss": round(dummy_log_loss, 4),
            "form_log_loss": round(form_log_loss, 4),
            "model_log_loss": round(ml_log_loss, 4),
            "model_accuracy": round(ml_accuracy, 4),
        },
        "confusion_matrix": confusion_matrix(y_test, ml_preds).tolist(),
        "classification_report": classification_report(
            y_test, ml_preds, target_names=LABELS, output_dict=True
        ),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    features.to_csv(ROOT / "data" / "processed_features.csv", index=False)

    return metadata


if __name__ == "__main__":
    result = train_and_evaluate()
    print("Training complete.")
    print(json.dumps(result["metrics"], indent=2))
