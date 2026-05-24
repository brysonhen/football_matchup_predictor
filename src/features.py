from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    # Short-window form
    "home_ppg_l5",
    "away_ppg_l5",
    "home_ppg_l10",
    "away_ppg_l10",
    # Goal output (short window)
    "home_gf_l5",
    "home_ga_l5",
    "away_gf_l5",
    "away_ga_l5",
    # Goal output (long window)
    "home_gf_l10",
    "home_ga_l10",
    "away_gf_l10",
    "away_ga_l10",
    # Goal difference per game
    "home_gd_l5",
    "away_gd_l5",
    "home_gd_l10",
    "away_gd_l10",
    # Rest / fatigue
    "home_rest_days",
    "away_rest_days",
    # Composite diffs (help logistic regression find the comparison signal)
    "ppg_diff_l5",
    "ppg_diff_l10",
    "gf_diff_l5",
    "gd_diff_l5",
    "ga_diff_l5",          # home_ga_l5 minus away_gf_l5 (home defense vs away attack)
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
            out[f"ppg_l{window}"]  = sum(w_pts) / window
            out[f"gf_l{window}"]   = sum(w_gf) / window
            out[f"ga_l{window}"]   = sum(w_ga) / window
            out[f"gd_l{window}"]   = (sum(w_gf) - sum(w_ga)) / window

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

        # Require both teams to have at least 10 games of history
        if (
            "ppg_l5" in home_snap
            and "ppg_l5" in away_snap
            and "ppg_l10" in home_snap
            and "ppg_l10" in away_snap
        ):
            h, a = home_snap, away_snap
            feature_row = {
                "Date": date,
                "Season": row.Season,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTR": row.FTR,
                # Short form
                "home_ppg_l5":  h["ppg_l5"],
                "away_ppg_l5":  a["ppg_l5"],
                "home_ppg_l10": h["ppg_l10"],
                "away_ppg_l10": a["ppg_l10"],
                # Goals (short)
                "home_gf_l5":   h["gf_l5"],
                "home_ga_l5":   h["ga_l5"],
                "away_gf_l5":   a["gf_l5"],
                "away_ga_l5":   a["ga_l5"],
                # Goals (long)
                "home_gf_l10":  h["gf_l10"],
                "home_ga_l10":  h["ga_l10"],
                "away_gf_l10":  a["gf_l10"],
                "away_ga_l10":  a["ga_l10"],
                # Goal difference
                "home_gd_l5":   h["gd_l5"],
                "away_gd_l5":   a["gd_l5"],
                "home_gd_l10":  h["gd_l10"],
                "away_gd_l10":  a["gd_l10"],
                # Rest
                "home_rest_days": h["rest_days"],
                "away_rest_days": a["rest_days"],
                # Composite diffs
                "ppg_diff_l5":  h["ppg_l5"]  - a["ppg_l5"],
                "ppg_diff_l10": h["ppg_l10"] - a["ppg_l10"],
                "gf_diff_l5":   h["gf_l5"]  - a["gf_l5"],
                "gd_diff_l5":   h["gd_l5"]  - a["gd_l5"],
                "ga_diff_l5":   h["ga_l5"]  - a["gf_l5"],  # home defense vs away attack
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

    def g(snap: dict, key: str, default: float = 0.0) -> float:
        return float(snap.get(key, default))

    return np.array([
        g(h, "ppg_l5"),   g(a, "ppg_l5"),
        g(h, "ppg_l10"),  g(a, "ppg_l10"),
        g(h, "gf_l5"),    g(h, "ga_l5"),
        g(a, "gf_l5"),    g(a, "ga_l5"),
        g(h, "gf_l10"),   g(h, "ga_l10"),
        g(a, "gf_l10"),   g(a, "ga_l10"),
        g(h, "gd_l5"),    g(a, "gd_l5"),
        g(h, "gd_l10"),   g(a, "gd_l10"),
        g(h, "rest_days"), g(a, "rest_days"),
        g(h, "ppg_l5")  - g(a, "ppg_l5"),
        g(h, "ppg_l10") - g(a, "ppg_l10"),
        g(h, "gf_l5")   - g(a, "gf_l5"),
        g(h, "gd_l5")   - g(a, "gd_l5"),
        g(h, "ga_l5")   - g(a, "gf_l5"),
    ], dtype=float)
