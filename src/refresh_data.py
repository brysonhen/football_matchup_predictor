"""
Download season CSVs from football-data.co.uk for any supported league.
Safe to run repeatedly — skips files that are already up to date.
"""
from __future__ import annotations

import ssl
import sys
import urllib.request
from datetime import date
from pathlib import Path

from src.leagues import BOOTSTRAP_SEASONS, League, LEAGUES

# macOS Python ships without system CA certs; create a context that works on
# both macOS and Linux without disabling verification entirely.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def _current_season_code() -> str:
    """Two-digit season code for the running season, e.g. '2526'."""
    today = date.today()
    year = today.year if today.month >= 8 else today.year - 1
    return f"{str(year)[2:]}{str(year + 1)[2:]}"


def download_season(league: League, season: str, force: bool = False) -> bool:
    """
    Download {code}_{season}.csv into data/raw/{league.id}/.
    Returns True if the file was (re)downloaded, False if skipped.
    """
    league_dir = RAW_DIR / league.id
    dest = league_dir / f"{league.fd_code}_{season}.csv"
    url = BASE_URL.format(season=season, code=league.fd_code)

    try:
        with urllib.request.urlopen(url, timeout=10, context=_SSL_CTX) as resp:
            remote_size = int(resp.headers.get("Content-Length", 0))
            local_size = dest.stat().st_size if dest.exists() else 0

            if not force and dest.exists() and local_size >= remote_size > 0:
                return False

            league_dir.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
            return True
    except Exception:
        return False


def refresh(league: League | str = "pl", force: bool = False) -> list[str]:
    """Download the current season for one league. Returns list of updated codes."""
    if isinstance(league, str):
        league = LEAGUES[league]
    current = _current_season_code()
    updated = []
    if download_season(league, current, force=force):
        updated.append(current)
    return updated


def bootstrap(league: League | str = "pl", force: bool = False) -> list[str]:
    """Download all bootstrap seasons for one league."""
    if isinstance(league, str):
        league = LEAGUES[league]
    updated = []
    for season in BOOTSTRAP_SEASONS:
        if download_season(league, season, force=force):
            updated.append(season)
            print(f"  [{league.id}] downloaded {season}")
    return updated


def refresh_all(force: bool = False) -> dict[str, list[str]]:
    """Refresh the current season for every league."""
    return {lid: refresh(league, force=force) for lid, league in LEAGUES.items()}


def bootstrap_all(force: bool = False) -> dict[str, list[str]]:
    """Download all bootstrap seasons for every league."""
    return {lid: bootstrap(league, force=force) for lid, league in LEAGUES.items()}


if __name__ == "__main__":
    # Usage:
    #   python -m src.refresh_data              — refresh current season, all leagues
    #   python -m src.refresh_data pl           — refresh current season, PL only
    #   python -m src.refresh_data --bootstrap  — download all history, all leagues
    args = sys.argv[1:]
    do_bootstrap = "--bootstrap" in args
    league_filter = [a for a in args if not a.startswith("--")]

    targets = (
        {lid: LEAGUES[lid] for lid in league_filter if lid in LEAGUES}
        if league_filter
        else LEAGUES
    )

    for lid, league in targets.items():
        if do_bootstrap:
            print(f"Bootstrapping {league.name}…")
            updated = bootstrap(league, force=False)
            print(f"  {len(updated)} seasons downloaded")
        else:
            updated = refresh(league)
            if updated:
                print(f"[{lid}] Updated: {', '.join(updated)}")
            else:
                print(f"[{lid}] Already up to date")
