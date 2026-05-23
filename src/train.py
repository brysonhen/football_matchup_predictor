from __future__ import annotations

import json
import sys
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
from src.leagues import LEAGUES, League
from src.load_data import load_matches

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
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


def train_and_evaluate(league: League | str = "pl") -> dict:
    if isinstance(league, str):
        league = LEAGUES[league]

    league_models_dir = MODELS_DIR / league.id
    league_models_dir.mkdir(parents=True, exist_ok=True)

    matches = load_matches(league)
    features = build_feature_matrix(matches)

    if features.empty:
        raise ValueError(f"[{league.id}] No features built — not enough match history.")

    # Hold out the most recent season as the test set.
    test_seasons = {str(features["Season"].max())}
    train_mask = ~features["Season"].isin(test_seasons)
    test_mask = features["Season"].isin(test_seasons)

    X_train = features.loc[train_mask, FEATURE_COLUMNS]
    y_train = _encode(features.loc[train_mask, "FTR"])
    X_test = features.loc[test_mask, FEATURE_COLUMNS]
    y_test = _encode(features.loc[test_mask, "FTR"])

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_log_loss = log_loss(y_test, dummy.predict_proba(X_test), labels=[0, 1, 2])

    form_log_loss = log_loss(
        y_test, _form_baseline_probs(features.loc[test_mask]), labels=[0, 1, 2]
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    ml_probs = pipeline.predict_proba(X_test)
    ml_preds = pipeline.predict(X_test)

    joblib.dump(pipeline, league_models_dir / "model.joblib")

    snapshots, last_date = latest_team_snapshots(matches)
    joblib.dump(snapshots, league_models_dir / "team_snapshots.joblib")

    metadata = {
        "league": league.id,
        "league_name": league.name,
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "train_seasons": sorted(features.loc[train_mask, "Season"].unique().tolist()),
        "test_seasons": sorted(test_seasons),
        "last_match_date": last_date.strftime("%Y-%m-%d"),
        "metrics": {
            "dummy_log_loss": round(dummy_log_loss, 4),
            "form_log_loss": round(form_log_loss, 4),
            "model_log_loss": round(log_loss(y_test, ml_probs, labels=[0, 1, 2]), 4),
            "model_accuracy": round(accuracy_score(y_test, ml_preds), 4),
        },
        "confusion_matrix": confusion_matrix(y_test, ml_preds).tolist(),
        "classification_report": classification_report(
            y_test, ml_preds, target_names=LABELS, output_dict=True
        ),
    }
    with open(league_models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    features.to_csv(ROOT / "data" / f"processed_features_{league.id}.csv", index=False)

    return metadata


if __name__ == "__main__":
    # Usage:
    #   python -m src.train          — train all leagues
    #   python -m src.train pl       — train PL only
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(LEAGUES.keys())

    for lid in targets:
        if lid not in LEAGUES:
            print(f"Unknown league: {lid}")
            continue
        print(f"Training {LEAGUES[lid].name}…")
        try:
            result = train_and_evaluate(lid)
            m = result["metrics"]
            print(
                f"  accuracy={m['model_accuracy']:.1%}  "
                f"log-loss={m['model_log_loss']:.4f}  "
                f"(test: {', '.join(result['test_seasons'])})"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
