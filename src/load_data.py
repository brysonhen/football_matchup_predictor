from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.leagues import League, LEAGUES

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def load_matches(league: League | str = "pl", raw_dir: Path | None = None) -> pd.DataFrame:
    if isinstance(league, str):
        league = LEAGUES[league]

    league_dir = (raw_dir or RAW_DIR) / league.id
    frames: list[pd.DataFrame] = []

    for path in sorted(league_dir.glob(f"{league.fd_code}_*.csv")):
        season = path.stem.replace(f"{league.fd_code}_", "")
        try:
            df = pd.read_csv(path, encoding="latin-1", usecols=lambda c: True)
            available = [c for c in CORE_COLUMNS if c in df.columns]
            if len(available) < len(CORE_COLUMNS):
                continue
            frames.append(df[CORE_COLUMNS].assign(Season=season))
        except Exception:
            continue

    if not frames:
        raise FileNotFoundError(
            f"No valid {league.fd_code}_*.csv files found in {league_dir}"
        )

    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    matches = matches[matches["FTR"].isin(["H", "D", "A"])].copy()
    matches = matches.sort_values("Date").drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"], keep="first"
    )
    return matches[CORE_COLUMNS + ["Season"]].reset_index(drop=True)


if __name__ == "__main__":
    import sys
    league_id = sys.argv[1] if len(sys.argv) > 1 else "pl"
    data = load_matches(league_id)
    print(f"[{league_id}] Loaded {len(data)} matches from {data['Season'].nunique()} seasons")
    print(data["FTR"].value_counts(normalize=True).round(3))
