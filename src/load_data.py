from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def load_matches(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or RAW_DIR
    frames: list[pd.DataFrame] = []

    for path in sorted(raw_dir.glob("E0_*.csv")):
        season = path.stem.replace("E0_", "")
        df = pd.read_csv(path, encoding="latin-1", usecols=lambda c: True)
        frames.append(df[CORE_COLUMNS].assign(Season=season))

    if not frames:
        raise FileNotFoundError(f"No E0_*.csv files found in {raw_dir}")

    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    matches = matches[matches["FTR"].isin(["H", "D", "A"])].copy()
    matches = matches.sort_values("Date").drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"], keep="first"
    )
    return matches[CORE_COLUMNS + ["Season"]].reset_index(drop=True)


if __name__ == "__main__":
    data = load_matches()
    print(f"Loaded {len(data)} matches from {data['Season'].nunique()} seasons")
    print(data["FTR"].value_counts(normalize=True).round(3))
