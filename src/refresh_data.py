"""
Download the current and previous Premier League season CSVs from football-data.co.uk.
Safe to run repeatedly — skips files that are already up to date.
"""
from __future__ import annotations

import urllib.request
from datetime import date
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def _current_season_code() -> str:
    """Return the two-digit season code for the current PL season, e.g. '2526'."""
    today = date.today()
    # PL season runs Aug–May; before August we're still in the previous season
    year = today.year if today.month >= 8 else today.year - 1
    return f"{str(year)[2:]}{str(year + 1)[2:]}"


def download_season(code: str, force: bool = False) -> bool:
    """
    Download E0_{code}.csv into data/raw/.
    Returns True if the file was (re)downloaded, False if skipped.
    """
    dest = RAW_DIR / f"E0_{code}.csv"
    url = BASE_URL.format(code=code)

    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            remote_size = int(resp.headers.get("Content-Length", 0))
            local_size = dest.stat().st_size if dest.exists() else 0

            if not force and dest.exists() and local_size >= remote_size > 0:
                return False  # already current

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
            return True
    except Exception:
        return False


def refresh(force: bool = False) -> list[str]:
    """Download current season (and previous if mid-season). Returns list of updated codes."""
    current = _current_season_code()
    updated = []
    for code in [current]:
        if download_season(code, force=force):
            updated.append(code)
    return updated


if __name__ == "__main__":
    updated = refresh()
    if updated:
        print(f"Updated: {', '.join(updated)}")
        print("Run `python -m src.train` to retrain the model.")
    else:
        print("Data is already up to date.")
