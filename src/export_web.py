"""
Export trained model + data as JSON for the Next.js website.

Produces:
  api/_data/model.json      logistic regression coefficients + scaler params
  api/_data/snapshots.json  current form per team
  api/_data/matches.json    all matches (for head-to-head lookups)
  public/data/teams.json    team metadata: stadium coords, logo id, city
  public/data/meta.json     metrics, seasons, monthly accuracy (for charts)

Run after `python -m src.train`. Pass the target web project path as arg 1.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, build_feature_matrix
from src.load_data import load_matches

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# Stadium coordinates + Premier League badge ids for the map and logos.
TEAM_META = {
    "Arsenal":          {"lat": 51.5549, "lng": -0.1084, "city": "London",        "stadium": "Emirates Stadium",            "logoId": 3},
    "Aston Villa":      {"lat": 52.5092, "lng": -1.8849, "city": "Birmingham",    "stadium": "Villa Park",                  "logoId": 7},
    "Bournemouth":      {"lat": 50.7352, "lng": -1.8384, "city": "Bournemouth",   "stadium": "Vitality Stadium",            "logoId": 91},
    "Brentford":        {"lat": 51.4907, "lng": -0.2887, "city": "London",        "stadium": "Gtech Community Stadium",      "logoId": 94},
    "Brighton":         {"lat": 50.8616, "lng": -0.0838, "city": "Brighton",      "stadium": "Amex Stadium",                "logoId": 36},
    "Burnley":          {"lat": 53.7890, "lng": -2.2301, "city": "Burnley",       "stadium": "Turf Moor",                   "logoId": 90},
    "Chelsea":          {"lat": 51.4817, "lng": -0.1910, "city": "London",        "stadium": "Stamford Bridge",             "logoId": 8},
    "Crystal Palace":   {"lat": 51.3983, "lng": -0.0855, "city": "London",        "stadium": "Selhurst Park",               "logoId": 31},
    "Everton":          {"lat": 53.4388, "lng": -2.9663, "city": "Liverpool",     "stadium": "Goodison Park",               "logoId": 11},
    "Fulham":           {"lat": 51.4749, "lng": -0.2217, "city": "London",        "stadium": "Craven Cottage",              "logoId": 54},
    "Ipswich":          {"lat": 52.0550, "lng":  1.1450, "city": "Ipswich",       "stadium": "Portman Road",                "logoId": 40},
    "Leeds":            {"lat": 53.7778, "lng": -1.5722, "city": "Leeds",         "stadium": "Elland Road",                 "logoId": 2},
    "Leicester":        {"lat": 52.6204, "lng": -1.1422, "city": "Leicester",     "stadium": "King Power Stadium",          "logoId": 13},
    "Liverpool":        {"lat": 53.4308, "lng": -2.9608, "city": "Liverpool",     "stadium": "Anfield",                     "logoId": 14},
    "Luton":            {"lat": 51.8842, "lng": -0.4316, "city": "Luton",         "stadium": "Kenilworth Road",             "logoId": 102},
    "Man City":         {"lat": 53.4831, "lng": -2.2004, "city": "Manchester",    "stadium": "Etihad Stadium",              "logoId": 43},
    "Man United":       {"lat": 53.4631, "lng": -2.2913, "city": "Manchester",    "stadium": "Old Trafford",                "logoId": 1},
    "Newcastle":        {"lat": 54.9756, "lng": -1.6217, "city": "Newcastle",     "stadium": "St James' Park",              "logoId": 4},
    "Norwich":          {"lat": 52.6220, "lng":  1.3092, "city": "Norwich",       "stadium": "Carrow Road",                 "logoId": 45},
    "Nott'm Forest":    {"lat": 52.9400, "lng": -1.1328, "city": "Nottingham",    "stadium": "City Ground",                 "logoId": 17},
    "Sheffield United": {"lat": 53.3703, "lng": -1.4709, "city": "Sheffield",     "stadium": "Bramall Lane",                "logoId": 49},
    "Southampton":      {"lat": 50.9058, "lng": -1.3911, "city": "Southampton",   "stadium": "St Mary's Stadium",           "logoId": 20},
    "Sunderland":       {"lat": 54.9145, "lng": -1.3882, "city": "Sunderland",    "stadium": "Stadium of Light",            "logoId": 56},
    "Tottenham":        {"lat": 51.6043, "lng": -0.0664, "city": "London",        "stadium": "Tottenham Hotspur Stadium",   "logoId": 6},
    "Watford":          {"lat": 51.6499, "lng": -0.4015, "city": "Watford",       "stadium": "Vicarage Road",               "logoId": 57},
    "West Brom":        {"lat": 52.5093, "lng": -1.9639, "city": "West Bromwich", "stadium": "The Hawthorns",               "logoId": 35},
    "West Ham":         {"lat": 51.5386, "lng": -0.0166, "city": "London",        "stadium": "London Stadium",              "logoId": 21},
    "Wolves":           {"lat": 52.5903, "lng": -2.1303, "city": "Wolverhampton", "stadium": "Molineux Stadium",            "logoId": 39},
}

LABEL_MAP = {0: "H", 1: "D", 2: "A"}


def main(web_root: Path) -> None:
    api_data = web_root / "api" / "_data"
    public_data = web_root / "data"
    api_data.mkdir(parents=True, exist_ok=True)
    public_data.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(MODELS / "model.joblib")
    snapshots = joblib.load(MODELS / "team_snapshots.joblib")
    with open(MODELS / "metadata.json") as f:
        metadata = json.load(f)

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    # ── model.json ────────────────────────────────────────────────────────────
    model_json = {
        "features": FEATURE_COLUMNS,
        "labels": metadata["labels"],
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    (api_data / "model.json").write_text(json.dumps(model_json, indent=2))

    # ── snapshots.json ────────────────────────────────────────────────────────
    snapshots_json = {
        team: {k: v for k, v in snap.items()}
        for team, snap in snapshots.items()
    }
    (api_data / "snapshots.json").write_text(json.dumps(snapshots_json, indent=2))

    # ── matches.json ──────────────────────────────────────────────────────────
    matches = load_matches()
    matches_json = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "home": h,
            "away": a,
            "fthg": int(hg),
            "ftag": int(ag),
            "ftr": r,
            "season": s,
        }
        for d, h, a, hg, ag, r, s in zip(
            matches["Date"], matches["HomeTeam"], matches["AwayTeam"],
            matches["FTHG"], matches["FTAG"], matches["FTR"], matches["Season"],
        )
    ]
    (api_data / "matches.json").write_text(json.dumps(matches_json))

    # ── monthly accuracy on the test season ──────────────────────────────────
    features = build_feature_matrix(matches)
    test = features[features["Season"].isin(metadata["test_seasons"])].copy()
    preds = pipeline.predict(test[FEATURE_COLUMNS])
    test["pred"] = [LABEL_MAP[p] for p in preds]
    test["correct"] = test["pred"] == test["FTR"]
    test["month"] = pd.to_datetime(test["Date"]).dt.strftime("%Y-%m")
    monthly = (
        test.groupby("month")
        .agg(accuracy=("correct", "mean"), matches=("correct", "size"))
        .reset_index()
    )
    monthly_json = [
        {"month": m, "accuracy": round(float(a), 3), "matches": int(n)}
        for m, a, n in zip(monthly["month"], monthly["accuracy"], monthly["matches"])
    ]

    # ── teams.json ────────────────────────────────────────────────────────────
    teams_json = {
        team: {**TEAM_META.get(team, {}), "name": team}
        for team in sorted(snapshots.keys())
    }
    (public_data / "teams.json").write_text(json.dumps(teams_json, indent=2))

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta_json = {
        "lastMatchDate": metadata["last_match_date"],
        "trainSeasons": metadata["train_seasons"],
        "testSeasons": metadata["test_seasons"],
        "metrics": metadata["metrics"],
        "confusionMatrix": metadata["confusion_matrix"],
        "monthlyAccuracy": monthly_json,
        "recentResults": list(reversed(matches_json[-8:])),
        "totalMatches": len(matches_json),
        "teamCount": len(snapshots),
    }
    (public_data / "meta.json").write_text(json.dumps(meta_json, indent=2))

    print(f"Exported to {web_root}:")
    print(f"  api/_data/model.json      ({len(FEATURE_COLUMNS)} features)")
    print(f"  api/_data/snapshots.json  ({len(snapshots)} teams)")
    print(f"  api/_data/matches.json    ({len(matches_json)} matches)")
    print(f"  data/teams.json           ({len(teams_json)} teams)")
    print(f"  data/meta.json            ({len(monthly_json)} months of accuracy)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.export_web <web_project_path>")
        sys.exit(1)
    main(Path(sys.argv[1]).resolve())
