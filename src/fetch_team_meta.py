"""
One-time script to fetch team metadata (logo URLs, stadium coords, city) from
TheSportsDB and cache it to config/{league_id}.json.

Run after bootstrapping CSV data so we know the exact team name strings used
by football-data.co.uk, then we fuzzy-match them to thesportsdb results.

Usage:
    python -m src.fetch_team_meta              # all leagues
    python -m src.fetch_team_meta laliga       # one league
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from src.leagues import LEAGUES, League
from src.load_data import load_matches

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

try:
    import certifi
    import ssl
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    import ssl
    _SSL_CTX = ssl.create_default_context()

# TheSportsDB league names (free API, no key needed for search)
TSDB_LEAGUE_NAMES: dict[str, str] = {
    "pl":         "English Premier League",
    "laliga":     "Spanish La Liga",
    "bundesliga": "German Bundesliga",
    "seriea":     "Italian Serie A",
    "ligue1":     "French Ligue 1",
}

TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode())


def _tsdb_teams(league_name: str) -> list[dict]:
    encoded = urllib.request.quote(league_name)
    data = _fetch_json(f"{TSDB_BASE}/search_all_teams.php?l={encoded}")
    return data.get("teams") or []


def _tsdb_search_team(name: str) -> dict | None:
    """Search thesportsdb for a single team by name."""
    encoded = urllib.request.quote(name)
    try:
        data = _fetch_json(f"{TSDB_BASE}/searchteams.php?t={encoded}")
        teams = data.get("teams") or []
        return teams[0] if teams else None
    except Exception:
        return None


def _normalize(name: str) -> str:
    return name.lower().replace(".", "").replace("-", " ").replace("'", "").strip()


def _best_match(fd_name: str, tsdb_teams: list[dict]) -> dict | None:
    """Find the closest thesportsdb team entry for a football-data.co.uk team name."""
    norm_fd = _normalize(fd_name)

    # Exact match first
    for t in tsdb_teams:
        if _normalize(t.get("strTeam", "")) == norm_fd:
            return t

    # Substring match (handles "Man City" → "Manchester City")
    for t in tsdb_teams:
        tsdb_norm = _normalize(t.get("strTeam", ""))
        if norm_fd in tsdb_norm or tsdb_norm in norm_fd:
            return t

    # Word overlap ≥ 1
    fd_words = set(norm_fd.split())
    best, best_score = None, 0
    for t in tsdb_teams:
        tsdb_words = set(_normalize(t.get("strTeam", "")).split())
        score = len(fd_words & tsdb_words)
        if score > best_score:
            best, best_score = t, score

    return best if best_score >= 1 else None


def _parse_coord(val: str | None) -> float | None:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None


def fetch_and_cache(league: League | str) -> dict:
    if isinstance(league, str):
        league = LEAGUES[league]

    CONFIG_DIR.mkdir(exist_ok=True)
    out_path = CONFIG_DIR / f"{league.id}.json"

    # Load existing cache so we don't overwrite manual edits
    existing: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    # Get team names from actual CSV data
    try:
        matches = load_matches(league)
    except FileNotFoundError:
        print(f"  [{league.id}] No CSV data found — run bootstrap first")
        return existing

    fd_teams = sorted(
        set(matches["HomeTeam"].tolist() + matches["AwayTeam"].tolist())
    )
    print(f"  [{league.id}] {len(fd_teams)} teams in CSV data")

    # Fetch from thesportsdb
    tsdb_league = TSDB_LEAGUE_NAMES.get(league.id, "")
    tsdb_teams: list[dict] = []
    if tsdb_league:
        try:
            tsdb_teams = _tsdb_teams(tsdb_league)
            print(f"  [{league.id}] {len(tsdb_teams)} teams from thesportsdb")
            time.sleep(0.5)  # be polite
        except Exception as e:
            print(f"  [{league.id}] thesportsdb fetch failed: {e}")

    # Common football-data.co.uk abbreviations → full names for better matching
    FD_EXPANSIONS: dict[str, list[str]] = {
        "Ath Bilbao": ["Athletic Club", "Athletic Bilbao"],
        "Ath Madrid": ["Atletico Madrid", "Atlético Madrid"],
        "Sociedad": ["Real Sociedad"],
        "Vallecano": ["Rayo Vallecano"],
        "Espanol": ["Espanyol", "RCD Espanyol"],
        "Celta": ["Celta Vigo", "RC Celta"],
        "Betis": ["Real Betis"],
        "Nott'm Forest": ["Nottingham Forest"],
        "Man City": ["Manchester City"],
        "Man United": ["Manchester United"],
        "West Brom": ["West Bromwich Albion"],
        "Sheffield United": ["Sheffield United"],
        "Hertha": ["Hertha Berlin", "Hertha BSC"],
        "Leverkusen": ["Bayer Leverkusen"],
        "Dortmund": ["Borussia Dortmund"],
        "M'gladbach": ["Borussia Mönchengladbach", "Borussia Monchengladbach"],
        "Ein Frankfurt": ["Eintracht Frankfurt"],
        "Greuther Furth": ["SpVgg Greuther Fürth"],
        "FC Koln": ["FC Köln", "Cologne"],
        "Bayern Munich": ["Bayern München", "FC Bayern Munich"],
        "Werder": ["Werder Bremen"],
        "St Pauli": ["FC St. Pauli"],
        "Inter": ["Inter Milan", "FC Internazionale"],
        "AC Milan": ["AC Milan", "Milan"],
        "Lazio": ["SS Lazio"],
        "Roma": ["AS Roma"],
        "Juventus": ["Juventus"],
        "Hellas Verona": ["Hellas Verona", "Verona"],
        "Spal": ["SPAL"],
        "Paris SG": ["Paris Saint-Germain", "PSG"],
        "St Etienne": ["Saint-Étienne", "AS Saint-Etienne"],
        "Marseille": ["Olympique de Marseille"],
        "Lyon": ["Olympique Lyonnais"],
        "Nantes": ["FC Nantes"],
        "Lens": ["RC Lens"],
        "Rennes": ["Stade Rennais"],
        "Montpellier": ["Montpellier HSC"],
        "Strasbourg": ["RC Strasbourg"],
        "Bordeaux": ["FC Girondins de Bordeaux"],
        "Metz": ["FC Metz"],
        "Brest": ["Stade Brestois 29", "Stade Brest"],
        "Lorient": ["FC Lorient"],
        "Clermont": ["Clermont Foot"],
        "Auxerre": ["AJ Auxerre"],
        "Troyes": ["ESTAC Troyes"],
        "Angers": ["SCO Angers"],
        "Le Havre": ["Le Havre AC"],
        "Reims": ["Stade de Reims"],
    }

    result: dict = {}
    for fd_name in fd_teams:
        entry = dict(existing.get(fd_name, {}))

        # Skip if we already have logo from a previous run
        if entry.get("logoUrl"):
            result[fd_name] = entry
            continue

        # Try league-level results first
        match = _best_match(fd_name, tsdb_teams) if tsdb_teams else None

        # Fall back to individual searches using expansions or the raw name
        if not match:
            search_names = FD_EXPANSIONS.get(fd_name, [fd_name])
            for sname in search_names:
                m = _tsdb_search_team(sname)
                time.sleep(2.5)
                if m:
                    match = m
                    break

        if match:
            # Logo URL — prefer the badge, fall back to jersey
            logo = match.get("strTeamBadge") or match.get("strTeamJersey") or ""
            if logo and not entry.get("logoUrl"):
                entry["logoUrl"] = logo

            # Stadium name
            if not entry.get("stadium"):
                entry["stadium"] = match.get("strStadium") or ""

            # City
            if not entry.get("city"):
                entry["city"] = match.get("strCity") or match.get("strLocation") or ""

            # Coordinates — thesportsdb sometimes has them
            if not entry.get("lat"):
                lat = _parse_coord(match.get("strStadiumLocation", "").split(",")[0] if "," in (match.get("strStadiumLocation") or "") else None)
                if lat:
                    entry["lat"] = lat
            if not entry.get("lng"):
                parts = (match.get("strStadiumLocation") or "").split(",")
                if len(parts) == 2:
                    lng = _parse_coord(parts[1])
                    if lng:
                        entry["lng"] = lng

            entry["_tsdb_name"] = match.get("strTeam", "")

        result[fd_name] = entry

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  [{league.id}] Saved to {out_path}")
    return result


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(LEAGUES.keys())
    for lid in targets:
        if lid not in LEAGUES:
            print(f"Unknown league: {lid}")
            continue
        print(f"Fetching team metadata for {LEAGUES[lid].name}…")
        fetch_and_cache(lid)
