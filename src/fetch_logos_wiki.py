"""
Fetch club crest URLs from Wikipedia pageimages API.
Updates config/{league}.json with logoUrl fields.
"""
import json, time, os
from pathlib import Path
import requests

CONFIG_DIR = Path(__file__).parent.parent / "config"

# football-data.co.uk name -> Wikipedia article title
WIKI_TITLES: dict[str, str] = {
    # La Liga
    "Alaves":       "Deportivo Alavés",
    "Almeria":      "UD Almería",
    "Ath Bilbao":   "Athletic Club",
    "Ath Madrid":   "Atlético Madrid",
    "Barcelona":    "FC Barcelona",
    "Betis":        "Real Betis",
    "Cadiz":        "Cádiz CF",
    "Celta":        "RC Celta de Vigo",
    "Eibar":        "SD Eibar",
    "Elche":        "Elche CF",
    "Espanol":      "RCD Espanyol",
    "Getafe":       "Getafe CF",
    "Girona":       "Girona FC",
    "Granada":      "Granada CF",
    "Huesca":       "SD Huesca",
    "Las Palmas":   "UD Las Palmas",
    "Leganes":      "CD Leganés",
    "Levante":      "Levante UD",
    "Mallorca":     "RCD Mallorca",
    "Osasuna":      "CA Osasuna",
    "Oviedo":       "Real Oviedo",
    "Real Madrid":  "Real Madrid CF",
    "Sevilla":      "Sevilla FC",
    "Sociedad":     "Real Sociedad",
    "Valencia":     "Valencia CF",
    "Valladolid":   "Real Valladolid",
    "Vallecano":    "Rayo Vallecano",
    "Villarreal":   "Villarreal CF",
    # Bundesliga
    "Augsburg":          "FC Augsburg",
    "Bayern Munich":     "FC Bayern Munich",
    "Bielefeld":         "Arminia Bielefeld",
    "Bochum":            "VfL Bochum",
    "Darmstadt":         "SV Darmstadt 98",
    "Dortmund":          "Borussia Dortmund",
    "Ein Frankfurt":     "Eintracht Frankfurt",
    "FC Koln":           "1. FC Köln",
    "Fortuna Dusseldorf":"Fortuna Düsseldorf",
    "Freiburg":          "Sport-Club Freiburg",
    "Greuther Furth":    "SpVgg Greuther Fürth",
    "Hamburg":           "Hamburger SV",
    "Heidenheim":        "1. FC Heidenheim 1846",
    "Hertha":            "Hertha BSC",
    "Hoffenheim":        "TSG 1899 Hoffenheim",
    "Holstein Kiel":     "Holstein Kiel",
    "Leverkusen":        "Bayer 04 Leverkusen",
    "M'gladbach":        "Borussia Mönchengladbach",
    "Mainz":             "1. FSV Mainz 05",
    "Paderborn":         "SC Paderborn 07",
    "RB Leipzig":        "RB Leipzig",
    "Schalke 04":        "FC Schalke 04",
    "St Pauli":          "FC St. Pauli",
    "Stuttgart":         "VfB Stuttgart",
    "Union Berlin":      "1. FC Union Berlin",
    "Werder Bremen":     "SV Werder Bremen",
    "Wolfsburg":         "VfL Wolfsburg",
    # Serie A
    "Atalanta":     "Atalanta BC",
    "Benevento":    "Benevento Calcio",
    "Bologna":      "Bologna FC 1909",
    "Brescia":      "Brescia Calcio",
    "Cagliari":     "Cagliari Calcio",
    "Como":         "Como 1907",
    "Cremonese":    "US Cremonese",
    "Crotone":      "FC Crotone",
    "Empoli":       "Empoli FC",
    "Fiorentina":   "ACF Fiorentina",
    "Frosinone":    "Frosinone Calcio",
    "Genoa":        "Genoa CFC",
    "Inter":        "Inter Milan",
    "Juventus":     "Juventus FC",
    "Lazio":        "SS Lazio",
    "Lecce":        "US Lecce",
    "Milan":        "AC Milan",
    "Monza":        "AC Monza",
    "Napoli":       "SSC Napoli",
    "Parma":        "Parma Calcio 1913",
    "Pisa":         "AC Pisa 1909",
    "Roma":         "AS Roma",
    "Salernitana":  "US Salernitana 1919",
    "Sampdoria":    "UC Sampdoria",
    "Sassuolo":     "US Sassuolo Calcio",
    "Spal":         "SPAL (football club)",
    "Spezia":       "Spezia Calcio",
    "Torino":       "Torino FC",
    "Udinese":      "Udinese Calcio",
    "Venezia":      "Venezia FC",
    "Verona":       "Hellas Verona FC",
    # Ligue 1
    "Ajaccio":      "AC Ajaccio",
    "Amiens":       "Amiens SC",
    "Angers":       "Angers SCO",
    "Auxerre":      "AJ Auxerre",
    "Bordeaux":     "FC Girondins de Bordeaux",
    "Brest":        "Stade Brestois 29",
    "Clermont":     "Clermont Foot 63",
    "Dijon":        "Dijon FCO",
    "Le Havre":     "Le Havre AC",
    "Lens":         "RC Lens",
    "Lille":        "LOSC Lille",
    "Lorient":      "FC Lorient",
    "Lyon":         "Olympique Lyonnais",
    "Marseille":    "Olympique de Marseille",
    "Metz":         "FC Metz",
    "Monaco":       "AS Monaco FC",
    "Montpellier":  "Montpellier HSC",
    "Nantes":       "FC Nantes",
    "Nice":         "OGC Nice",
    "Nimes":        "Nîmes Olympique",
    "Paris FC":     "Paris FC",
    "Paris SG":     "Paris Saint-Germain FC",
    "Reims":        "Stade de Reims",
    "Rennes":       "Stade Rennais FC",
    "St Etienne":   "AS Saint-Étienne",
    "Strasbourg":   "RC Strasbourg Alsace",
    "Toulouse":     "Toulouse FC",
    "Troyes":       "ES Troyes AC",
}

LEAGUES = ["laliga", "bundesliga", "seriea", "ligue1"]


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "SoccerPredictor/1.0 (educational project)"})


def fetch_wiki_thumbnails_batch(titles: list[str]) -> dict[str, str]:
    """Query up to 50 titles at once. Returns {normalized_title: url}."""
    result: dict[str, str] = {}
    try:
        r = _SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": "|".join(titles),
                "prop": "pageimages",
                "pithumbsize": 80,
                "format": "json",
                "redirects": 1,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        # Build a map of normalized/redirected title → original request title
        norm_map: dict[str, str] = {}
        for n in data.get("query", {}).get("normalized", []):
            norm_map[n["to"]] = n["from"]
        for rd in data.get("query", {}).get("redirects", []):
            norm_map[rd["to"]] = norm_map.get(rd["from"], rd["from"])
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            src = page.get("thumbnail", {}).get("source")
            if src:
                src = src.replace("/80px-", "/200px-")
                page_title = page.get("title", "")
                # Find which original title maps to this page
                orig = norm_map.get(page_title, page_title)
                result[orig] = src
                result[page_title] = src  # also index by resolved title
    except Exception as e:
        print(f"  BATCH ERROR: {e}")
    return result


def main():
    results: dict[str, str] = {}

    # Collect unique teams needed
    needed: list[str] = []
    for league in LEAGUES:
        path = CONFIG_DIR / f"{league}.json"
        data = json.loads(path.read_text())
        for fd_name in data:
            if fd_name not in needed:
                if fd_name in WIKI_TITLES:
                    needed.append(fd_name)
                else:
                    print(f"  [WARN] No wiki title for: {fd_name}")

    wiki_titles = [WIKI_TITLES[n] for n in needed]
    print(f"Fetching logos for {len(needed)} teams in batches from Wikipedia...")

    # Batch into groups of 50 (API limit)
    BATCH = 50
    for batch_start in range(0, len(wiki_titles), BATCH):
        batch_titles = wiki_titles[batch_start: batch_start + BATCH]
        batch_fd = needed[batch_start: batch_start + BATCH]
        print(f"  Batch {batch_start // BATCH + 1}: {len(batch_titles)} titles")
        batch_results = fetch_wiki_thumbnails_batch(batch_titles)
        for fd_name, wiki_title in zip(batch_fd, batch_titles):
            url = batch_results.get(wiki_title) or batch_results.get(fd_name)
            if url:
                results[fd_name] = url
                print(f"    OK  {fd_name}")
            else:
                print(f"    --  {fd_name}")
        if batch_start + BATCH < len(wiki_titles):
            time.sleep(2)

    ok = len(results)
    print(f"\n{ok}/{len(needed)} logos found")

    for league in LEAGUES:
        path = CONFIG_DIR / f"{league}.json"
        data = json.loads(path.read_text())
        updated = 0
        for fd_name, entry in data.items():
            logo = results.get(fd_name)
            if logo:
                entry["logoUrl"] = logo
                updated += 1
            # Leave existing logoUrl if no new result (don't delete)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  [{league}] {updated}/{len(data)} logos written")


if __name__ == "__main__":
    main()
