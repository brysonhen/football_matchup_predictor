import json
from pathlib import Path

import joblib
import numpy as np

from src.features import FEATURE_COLUMNS, matchup_features

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

OUTCOME_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}


def load_artifacts():
    pipeline = joblib.load(MODELS_DIR / "model.joblib")
    snapshots = joblib.load(MODELS_DIR / "team_snapshots.joblib")
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return pipeline, snapshots, metadata


def predict_matchup(home: str, away: str) -> dict:
    pipeline, snapshots, metadata = load_artifacts()

    features = matchup_features(home, away, snapshots)
    if features is None:
        missing = [t for t in (home, away) if t not in snapshots]
        raise ValueError(f"Not enough history for: {', '.join(missing)}")

    probs = pipeline.predict_proba(features.reshape(1, -1))[0]
    pred_idx = int(np.argmax(probs))
    pred_label = metadata["labels"][pred_idx]

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    scaled = scaler.transform(features.reshape(1, -1))[0]
    coefs = model.coef_[pred_idx]
    contributions = list(zip(FEATURE_COLUMNS, (coefs * scaled).tolist()))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "home": home,
        "away": away,
        "probabilities": {
            OUTCOME_NAMES[label]: float(prob)
            for label, prob in zip(metadata["labels"], probs)
        },
        "predicted": OUTCOME_NAMES[pred_label],
        "predicted_label": pred_label,
        "top_drivers": [
            {"feature": name, "impact": round(impact, 4)}
            for name, impact in contributions[:5]
        ],
        "form_through": metadata["last_match_date"],
    }
