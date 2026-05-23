"""
Export trained model + data as JSON for the Next.js website.

Per-league output (inside the web project root):
  api/_data/{league}/model.json       LR coefficients + scaler params
  api/_data/{league}/snapshots.json   current form per team
  api/_data/{league}/matches.json     all matches (head-to-head lookups)
  data/{league}/teams.json            team metadata: coords, logo, city
  data/{league}/meta.json             metrics, seasons, monthly accuracy

Also writes:
  data/leagues.json                   list of available leagues for the UI

Run after `python -m src.train`. Pass the target web project path as arg 1.

Usage:
    python -m src.export_web <web_path>           # all leagues
    python -m src.export_web <web_path> pl        # one league only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, build_feature_matrix
from src.leagues import LEAGUES, League
from src.load_data import load_matches

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"
LABEL_MAP = {0: "H", 1: "D", 2: "A"}

# ── PL-specific team metadata ─────────────────────────────────────────────────
# Premier League uses the official PL badge CDN for logos (higher quality).
# Other leagues use logoUrl from config/{league}.json (thesportsdb).
PL_TEAM_META: dict[str, dict] = {
    "Arsenal":          {"lat": 51.5549, "lng": -0.1084, "city": "London",        "stadium": "Emirates Stadium",           "logoId": 3},
    "Aston Villa":      {"lat": 52.5092, "lng": -1.8849, "city": "Birmingham",    "stadium": "Villa Park",                 "logoId": 7},
    "Bournemouth":      {"lat": 50.7352, "lng": -1.8384, "city": "Bournemouth",   "stadium": "Vitality Stadium",           "logoId": 91},
    "Brentford":        {"lat": 51.4907, "lng": -0.2887, "city": "London",        "stadium": "Gtech Community Stadium",    "logoId": 94},
    "Brighton":         {"lat": 50.8616, "lng": -0.0838, "city": "Brighton",      "stadium": "Amex Stadium",               "logoId": 36},
    "Burnley":          {"lat": 53.7890, "lng": -2.2301, "city": "Burnley",       "stadium": "Turf Moor",                  "logoId": 90},
    "Chelsea":          {"lat": 51.4817, "lng": -0.1910, "city": "London",        "stadium": "Stamford Bridge",            "logoId": 8},
    "Crystal Palace":   {"lat": 51.3983, "lng": -0.0855, "city": "London",        "stadium": "Selhurst Park",              "logoId": 31},
    "Everton":          {"lat": 53.4388, "lng": -2.9663, "city": "Liverpool",     "stadium": "Goodison Park",              "logoId": 11},
    "Fulham":           {"lat": 51.4749, "lng": -0.2217, "city": "London",        "stadium": "Craven Cottage",             "logoId": 54},
    "Ipswich":          {"lat": 52.0550, "lng":  1.1450, "city": "Ipswich",       "stadium": "Portman Road",               "logoId": 40},
    "Leeds":            {"lat": 53.7778, "lng": -1.5722, "city": "Leeds",         "stadium": "Elland Road",                "logoId": 2},
    "Leicester":        {"lat": 52.6204, "lng": -1.1422, "city": "Leicester",     "stadium": "King Power Stadium",         "logoId": 13},
    "Liverpool":        {"lat": 53.4308, "lng": -2.9608, "city": "Liverpool",     "stadium": "Anfield",                    "logoId": 14},
    "Luton":            {"lat": 51.8842, "lng": -0.4316, "city": "Luton",         "stadium": "Kenilworth Road",            "logoId": 102},
    "Man City":         {"lat": 53.4831, "lng": -2.2004, "city": "Manchester",    "stadium": "Etihad Stadium",             "logoId": 43},
    "Man United":       {"lat": 53.4631, "lng": -2.2913, "city": "Manchester",    "stadium": "Old Trafford",               "logoId": 1},
    "Newcastle":        {"lat": 54.9756, "lng": -1.6217, "city": "Newcastle",     "stadium": "St James' Park",             "logoId": 4},
    "Norwich":          {"lat": 52.6220, "lng":  1.3092, "city": "Norwich",       "stadium": "Carrow Road",                "logoId": 45,  "noRetina": True},
    "Nott'm Forest":    {"lat": 52.9400, "lng": -1.1328, "city": "Nottingham",    "stadium": "City Ground",                "logoId": 17},
    "Sheffield United": {"lat": 53.3703, "lng": -1.4709, "city": "Sheffield",     "stadium": "Bramall Lane",               "logoId": 49},
    "Southampton":      {"lat": 50.9058, "lng": -1.3911, "city": "Southampton",   "stadium": "St Mary's Stadium",          "logoId": 20},
    "Sunderland":       {"lat": 54.9145, "lng": -1.3882, "city": "Sunderland",    "stadium": "Stadium of Light",           "logoId": 56,  "noRetina": True},
    "Tottenham":        {"lat": 51.6043, "lng": -0.0664, "city": "London",        "stadium": "Tottenham Hotspur Stadium",  "logoId": 6},
    "Watford":          {"lat": 51.6499, "lng": -0.4015, "city": "Watford",       "stadium": "Vicarage Road",              "logoId": 57,  "noRetina": True},
    "West Brom":        {"lat": 52.5093, "lng": -1.9639, "city": "West Bromwich", "stadium": "The Hawthorns",              "logoId": 35,  "noRetina": True},
    "West Ham":         {"lat": 51.5386, "lng": -0.0166, "city": "London",        "stadium": "London Stadium",             "logoId": 21},
    "Wolves":           {"lat": 52.5903, "lng": -2.1303, "city": "Wolverhampton", "stadium": "Molineux Stadium",           "logoId": 39},
}


def _load_team_meta_config(league: League) -> dict:
    """Load cached team metadata from config/{league.id}.json, if it exists."""
    path = CONFIG_DIR / f"{league.id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _build_teams_json(league: League, snapshots: dict) -> dict:
    """Build the teams.json dict for the given league."""
    if league.id == "pl":
        return {
            team: {"name": team, **PL_TEAM_META.get(team, {})}
            for team in sorted(snapshots.keys())
        }

    config = _load_team_meta_config(league)
    result = {}
    for team in sorted(snapshots.keys()):
        meta = dict(config.get(team, {}))
        meta.pop("_tsdb_name", None)  # internal key, don't ship to web
        result[team] = {"name": team, **meta}
    return result


def export_league(league: League | str, web_root: Path) -> None:
    if isinstance(league, str):
        league = LEAGUES[league]

    league_models = MODELS_DIR / league.id
    if not league_models.exists():
        print(f"  [{league.id}] No trained model found — skipping")
        return

    api_data = web_root / "api" / "_data" / league.id
    public_data = web_root / "data" / league.id
    api_data.mkdir(parents=True, exist_ok=True)
    public_data.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(league_models / "model.joblib")
    snapshots = joblib.load(league_models / "team_snapshots.joblib")
    with open(league_models / "metadata.json") as f:
        metadata = json.load(f)

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    # ── model.json ────────────────────────────────────────────────────────────
    (api_data / "model.json").write_text(
        json.dumps(
            {
                "features": FEATURE_COLUMNS,
                "labels": metadata["labels"],
                "scaler": {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                },
                "coef": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
            },
            indent=2,
        )
    )

    # ── snapshots.json ────────────────────────────────────────────────────────
    (api_data / "snapshots.json").write_text(
        json.dumps({team: dict(snap) for team, snap in snapshots.items()}, indent=2)
    )

    # ── matches.json ──────────────────────────────────────────────────────────
    matches = load_matches(league)
    matches_list = [
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
            matches["Date"],
            matches["HomeTeam"],
            matches["AwayTeam"],
            matches["FTHG"],
            matches["FTAG"],
            matches["FTR"],
            matches["Season"],
        )
    ]
    (api_data / "matches.json").write_text(json.dumps(matches_list))

    # ── monthly accuracy on test season ──────────────────────────────────────
    features = build_feature_matrix(matches)
    test = features[features["Season"].isin(metadata["test_seasons"])].copy()
    preds = pipeline.predict(test[FEATURE_COLUMNS])
    test = test.copy()
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
    teams_json = _build_teams_json(league, snapshots)
    (public_data / "teams.json").write_text(json.dumps(teams_json, indent=2))

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta_json = {
        "league": league.id,
        "leagueName": league.name,
        "country": league.country,
        "lastMatchDate": metadata["last_match_date"],
        "trainSeasons": metadata["train_seasons"],
        "testSeasons": metadata["test_seasons"],
        "metrics": metadata["metrics"],
        "confusionMatrix": metadata["confusion_matrix"],
        "monthlyAccuracy": monthly_json,
        "recentResults": list(reversed(matches_list[-8:])),
        "totalMatches": len(matches_list),
        "teamCount": len(snapshots),
    }
    (public_data / "meta.json").write_text(json.dumps(meta_json, indent=2))

    print(
        f"  [{league.id}] {len(snapshots)} teams | "
        f"{len(matches_list)} matches | "
        f"accuracy={metadata['metrics']['model_accuracy']:.1%}"
    )


def export_leagues_index(web_root: Path, exported: list[str]) -> None:
    """Write data/leagues.json so the UI knows which leagues are available."""
    leagues_list = [
        {
            "id": LEAGUES[lid].id,
            "name": LEAGUES[lid].name,
            "country": LEAGUES[lid].country,
        }
        for lid in exported
        if lid in LEAGUES
    ]
    out = web_root / "data" / "leagues.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(leagues_list, indent=2))
    print(f"  Wrote data/leagues.json ({len(leagues_list)} leagues)")


def main(web_root: Path, league_ids: list[str] | None = None) -> None:
    targets = league_ids or list(LEAGUES.keys())
    print(f"Exporting to {web_root}:")
    exported = []
    for lid in targets:
        if lid not in LEAGUES:
            print(f"  Unknown league: {lid} — skipping")
            continue
        export_league(LEAGUES[lid], web_root)
        exported.append(lid)
    export_leagues_index(web_root, exported)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.export_web <web_project_path> [league ...]")
        sys.exit(1)
    web_path = Path(sys.argv[1]).resolve()
    leagues = sys.argv[2:] if len(sys.argv) > 2 else None
    main(web_path, leagues)
