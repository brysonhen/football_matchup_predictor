from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "home_ppg_l5",
    "away_ppg_l5",
    "home_ppg_l10",
    "away_ppg_l10",
    "home_gf_l5",
    "home_ga_l5",
    "away_gf_l5",
    "away_ga_l5",
    "home_rest_days",
    "away_rest_days",
    "ppg_diff_l5",
    "gf_diff_l5",
]

WINDOWS = (5, 10)


def _points(result: str) -> int:
    return {"H": 3, "D": 1, "A": 0}[result]


def _away_points(result: str) -> int:
    return {"H": 0, "D": 1, "A": 3}[result]


@dataclass
class TeamHistory:
    dates: list[pd.Timestamp] = field(default_factory=list)
    goals_for: list[int] = field(default_factory=list)
    goals_against: list[int] = field(default_factory=list)
    points: list[int] = field(default_factory=list)

    def snapshot(self, as_of: pd.Timestamp) -> dict[str, float]:
        mask = [d < as_of for d in self.dates]
        if not any(mask):
            return {}

        idx = [i for i, m in enumerate(mask) if m]
        pts = [self.points[i] for i in idx]
        gf = [self.goals_for[i] for i in idx]
        ga = [self.goals_against[i] for i in idx]
        dates = [self.dates[i] for i in idx]

        out: dict[str, float] = {}
        for window in WINDOWS:
            w_pts = pts[-window:]
            w_gf = gf[-window:]
            w_ga = ga[-window:]
            if len(w_pts) < window:
                continue
            out[f"ppg_l{window}"] = sum(w_pts) / window
            if window == 5:
                out["gf_l5"] = sum(w_gf) / window
                out["ga_l5"] = sum(w_ga) / window

        out["rest_days"] = (as_of - dates[-1]).days

        # Last 5 results as W/D/L strings (most recent last)
        recent_pts = pts[-5:]
        out["recent_form"] = ["W" if p == 3 else ("D" if p == 1 else "L") for p in recent_pts]

        return out

    def add_match(
        self,
        match_date: pd.Timestamp,
        goals_for: int,
        goals_against: int,
        points: int,
    ) -> None:
        self.dates.append(match_date)
        self.goals_for.append(goals_for)
        self.goals_against.append(goals_against)
        self.points.append(points)


def build_feature_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    histories: dict[str, TeamHistory] = {}
    rows: list[dict] = []

    for row in matches.itertuples(index=False):
        date = row.Date
        home, away = row.HomeTeam, row.AwayTeam

        if home not in histories:
            histories[home] = TeamHistory()
        if away not in histories:
            histories[away] = TeamHistory()

        home_snap = histories[home].snapshot(date)
        away_snap = histories[away].snapshot(date)

        if (
            "ppg_l5" in home_snap
            and "ppg_l5" in away_snap
            and "ppg_l10" in home_snap
            and "ppg_l10" in away_snap
        ):
            feature_row = {
                "Date": date,
                "Season": row.Season,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTR": row.FTR,
                "home_ppg_l5": home_snap["ppg_l5"],
                "away_ppg_l5": away_snap["ppg_l5"],
                "home_ppg_l10": home_snap["ppg_l10"],
                "away_ppg_l10": away_snap["ppg_l10"],
                "home_gf_l5": home_snap["gf_l5"],
                "home_ga_l5": home_snap["ga_l5"],
                "away_gf_l5": away_snap["gf_l5"],
                "away_ga_l5": away_snap["ga_l5"],
                "home_rest_days": home_snap["rest_days"],
                "away_rest_days": away_snap["rest_days"],
                "ppg_diff_l5": home_snap["ppg_l5"] - away_snap["ppg_l5"],
                "gf_diff_l5": home_snap["gf_l5"] - away_snap["gf_l5"],
            }
            rows.append(feature_row)

        histories[home].add_match(date, row.FTHG, row.FTAG, _points(row.FTR))
        histories[away].add_match(date, row.FTAG, row.FTHG, _away_points(row.FTR))

    return pd.DataFrame(rows)


def latest_team_snapshots(matches: pd.DataFrame) -> tuple[dict[str, dict], pd.Timestamp]:
    histories: dict[str, TeamHistory] = {}

    for row in matches.itertuples(index=False):
        date = row.Date
        home, away = row.HomeTeam, row.AwayTeam
        for team in (home, away):
            if team not in histories:
                histories[team] = TeamHistory()

        histories[home].add_match(date, row.FTHG, row.FTAG, _points(row.FTR))
        histories[away].add_match(date, row.FTAG, row.FTHG, _away_points(row.FTR))

    last_date = matches["Date"].max()
    snapshots = {}
    for team, history in histories.items():
        snap = history.snapshot(last_date + pd.Timedelta(days=1))
        if "ppg_l5" in snap and "ppg_l10" in snap:
            snapshots[team] = snap

    return snapshots, last_date


def matchup_features(home: str, away: str, snapshots: dict[str, dict]) -> np.ndarray | None:
    if home not in snapshots or away not in snapshots:
        return None

    h, a = snapshots[home], snapshots[away]
    return np.array(
        [
            h["ppg_l5"],
            a["ppg_l5"],
            h["ppg_l10"],
            a["ppg_l10"],
            h["gf_l5"],
            h["ga_l5"],
            a["gf_l5"],
            a["ga_l5"],
            h["rest_days"],
            a["rest_days"],
            h["ppg_l5"] - a["ppg_l5"],
            h["gf_l5"] - a["gf_l5"],
        ],
        dtype=float,
    )
