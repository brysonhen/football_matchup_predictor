"""
Fetch club logo URLs from ESPN's public API (no auth required).
Merges results into config/{league}.json logoUrl fields,
only filling in teams that don't already have a logo.
"""
import json, time
from pathlib import Path
import requests

CONFIG_DIR = Path(__file__).parent.parent / "config"

# ESPN league slug -> our league id
ESPN_LEAGUES = {
    "esp.1": "laliga",
    "ger.1": "bundesliga",
    "ita.1": "seriea",
    "fra.1": "ligue1",
}

# football-data abbreviated name -> ESPN displayName variants to try
# Only needed where fuzzy matching might fail
ALIASES: dict[str, list[str]] = {
    "Ath Bilbao":       ["Athletic Club", "Athletic Bilbao"],
    "Ath Madrid":       ["Atlético Madrid", "Atletico Madrid"],
    "Betis":            ["Real Betis Balompié", "Real Betis"],
    "Celta":            ["Celta de Vigo", "RC Celta de Vigo", "Celta Vigo"],
    "Espanol":          ["Espanyol", "RCD Espanyol"],
    "Sociedad":         ["Real Sociedad"],
    "Vallecano":        ["Rayo Vallecano"],
    "Ein Frankfurt":    ["Eintracht Frankfurt", "Frankfurt"],
    "FC Koln":          ["FC Köln", "Köln", "Koln"],
    "Fortuna Dusseldorf": ["Fortuna Düsseldorf", "Düsseldorf"],
    "Greuther Furth":   ["SpVgg Greuther Fürth", "Greuther Fürth"],
    "M'gladbach":       ["Borussia Mönchengladbach", "Mönchengladbach", "Mgladbach"],
    "RB Leipzig":       ["RB Leipzig", "Leipzig"],
    "St Pauli":         ["FC St. Pauli", "St. Pauli"],
    "Paris SG":         ["Paris Saint-Germain", "PSG"],
    "St Etienne":       ["AS Saint-Étienne", "Saint-Etienne", "St-Etienne"],
    "Lyon":             ["Olympique Lyonnais", "Olympique de Lyon"],
    "Marseille":        ["Olympique de Marseille"],
    "Monaco":           ["AS Monaco"],
    "Nimes":            ["Nîmes Olympique", "Nimes Olympique"],
    "Bordeaux":         ["Girondins de Bordeaux", "FC Girondins de Bordeaux"],
    "Brest":            ["Stade Brestois 29", "Stade Brestois"],
    "Clermont":         ["Clermont Foot", "Clermont Foot 63"],
    "Montpellier":      ["Montpellier HSC"],
    "Le Havre":         ["Le Havre AC"],
    "Lens":             ["RC Lens"],
    "Lille":            ["LOSC Lille"],
    "Lorient":          ["FC Lorient"],
    "Reims":            ["Stade de Reims"],
    "Rennes":           ["Stade Rennais FC", "Stade Rennais", "Stade rennais"],
    "Strasbourg":       ["RC Strasbourg", "RC Strasbourg Alsace"],
    "Troyes":           ["ES Troyes AC", "Troyes AC"],
    "Auxerre":          ["AJ Auxerre"],
}


def normalize(name: str) -> str:
    return name.lower().replace("fc ", "").replace(" fc", "").replace("sc ", "").replace(" sc", "").strip()


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 SoccerPredictor/1.0"})


def fetch_espn_logos(slug: str) -> dict[str, str]:
    """Returns {display_name_lower: logo_url} for all teams in ESPN league."""
    r = _SESSION.get(
        f"http://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams",
        timeout=15,
    )
    r.raise_for_status()
    teams = r.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    result: dict[str, str] = {}
    for t in teams:
        team = t.get("team", {})
        name = team.get("displayName", "")
        logos = team.get("logos", [])
        if name and logos:
            url = logos[0].get("href", "")
            if url:
                result[name.lower()] = url
                result[normalize(name)] = url
    return result


def match_team(fd_name: str, espn_map: dict[str, str]) -> str | None:
    # Try exact (case-insensitive)
    key = fd_name.lower()
    if key in espn_map:
        return espn_map[key]
    # Try normalised
    nkey = normalize(fd_name)
    if nkey in espn_map:
        return espn_map[nkey]
    # Try aliases
    for alias in ALIASES.get(fd_name, []):
        akey = alias.lower()
        if akey in espn_map:
            return espn_map[akey]
        nk = normalize(alias)
        if nk in espn_map:
            return espn_map[nk]
    # Substring match
    for espn_key, url in espn_map.items():
        if nkey in espn_key or espn_key in nkey:
            return url
    return None


def main():
    for espn_slug, league_id in ESPN_LEAGUES.items():
        path = CONFIG_DIR / f"{league_id}.json"
        config = json.loads(path.read_text())

        print(f"\n[{league_id}] Fetching ESPN logos ({espn_slug})...")
        espn_map = fetch_espn_logos(espn_slug)
        print(f"  ESPN returned {len(espn_map)//2} teams")

        updated = skipped = missing = 0
        for fd_name, entry in config.items():
            url = match_team(fd_name, espn_map)
            if url:
                entry["logoUrl"] = url
                updated += 1
                print(f"  + {fd_name}")
            elif entry.get("logoUrl"):
                skipped += 1  # keep existing
            else:
                missing += 1
                print(f"  - {fd_name} [no match]")

        path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
        total = updated + skipped
        print(f"  [{league_id}] {total}/{len(config)} logos total ({updated} from ESPN, {skipped} kept, {missing} missing)")

        time.sleep(0.5)


if __name__ == "__main__":
    main()
