"""
One-time script: seed stadium lat/lng into config/{league}.json for non-PL leagues.
Run after fetch_team_meta — this only fills in coordinates, never overwrites logos.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

COORDS: dict[str, dict[str, tuple[float, float]]] = {
    "laliga": {
        "Alaves":       (42.8509, -2.6777),
        "Almeria":      (36.8498, -2.4007),
        "Ath Bilbao":   (43.2641, -2.9499),
        "Ath Madrid":   (40.4360, -3.5994),
        "Barcelona":    (41.3809,  2.1228),
        "Betis":        (37.3562, -5.9818),
        "Cadiz":        (36.5150, -6.2706),
        "Celta":        (42.2118, -8.7378),
        "Eibar":        (43.1791, -2.4731),
        "Elche":        (38.2620, -0.6972),
        "Espanol":      (41.3480,  2.0745),
        "Getafe":       (40.3254, -3.7178),
        "Girona":       (41.9843,  2.8179),
        "Granada":      (37.1662, -3.5956),
        "Huesca":       (42.1342, -0.4095),
        "Las Palmas":   (28.0997,-15.4468),
        "Leganes":      (40.3389, -3.7671),
        "Levante":      (39.4965, -0.3449),
        "Mallorca":     (39.5951,  2.6302),
        "Osasuna":      (42.7994, -1.6367),
        "Oviedo":       (43.3573, -5.8419),
        "Real Madrid":  (40.4531, -3.6883),
        "Sevilla":      (37.3840, -5.9706),
        "Sociedad":     (43.3018, -1.9733),
        "Valencia":     (39.4745, -0.3585),
        "Valladolid":   (41.6399, -4.7336),
        "Vallecano":    (40.3913, -3.6557),
        "Villarreal":   (39.9441, -0.1030),
    },
    "bundesliga": {
        "Augsburg":           (48.3233, 10.8864),
        "Bayern Munich":      (48.2188, 11.6248),
        "Bielefeld":          (51.9931,  8.5325),
        "Bochum":             (51.4897,  7.2361),
        "Darmstadt":          (49.8533,  8.6555),
        "Dortmund":           (51.4926,  7.4519),
        "Ein Frankfurt":      (50.0686,  8.6453),
        "FC Koln":            (50.9333,  6.8750),
        "Fortuna Dusseldorf": (51.2600,  6.7934),
        "Freiburg":           (47.9976,  7.8905),
        "Greuther Furth":     (49.4819, 10.9723),
        "Hamburg":            (53.5876, 10.0565),
        "Heidenheim":         (48.6750, 10.1556),
        "Hertha":             (52.5147, 13.2395),
        "Hoffenheim":         (49.2385,  8.8898),
        "Kiel":               (54.3739, 10.1327),
        "Leverkusen":         (51.0383,  7.0024),
        "M'gladbach":         (51.1744,  6.3853),
        "Mainz":              (49.9841,  8.2242),
        "Paderborn":          (51.7106,  8.7490),
        "RB Leipzig":         (51.3457, 12.3484),
        "St Pauli":           (53.5549,  9.9678),
        "Stuttgart":          (48.7924,  9.2319),
        "Union Berlin":       (52.4571, 13.5680),
        "Werder":             (53.0663,  8.8376),
        "Wolfsburg":          (52.4323, 10.8043),
        "Schalke":            (51.5543,  7.0674),
    },
    "seriea": {
        "Atalanta":      (45.7052,  9.6800),
        "Benevento":     (41.1341, 14.7854),
        "Bologna":       (44.4929, 11.3097),
        "Brescia":       (45.5348, 10.2397),
        "Cagliari":      (39.2077,  9.1247),
        "Como":          (45.8114,  9.0895),
        "Cremonese":     (45.1347, 10.0256),
        "Crotone":       (39.0797, 17.1342),
        "Empoli":        (43.7219, 10.9530),
        "Fiorentina":    (43.7798, 11.2820),
        "Frosinone":     (41.6371, 13.3528),
        "Genoa":         (44.4169,  8.9531),
        "Hellas Verona": (45.4341, 10.9659),
        "Inter":         (45.4780,  9.1238),
        "Juventus":      (45.1097,  7.6413),
        "Lazio":         (41.9339, 12.4544),
        "Lecce":         (40.3516, 18.1799),
        "Milan":         (45.4780,  9.1238),
        "Monza":         (45.6163,  9.2762),
        "Napoli":        (40.8279, 14.1932),
        "Parma":         (44.7973, 10.3333),
        "Roma":          (41.9339, 12.4544),
        "Salernitana":   (40.6732, 14.7681),
        "Sampdoria":     (44.4169,  8.9531),
        "Sassuolo":      (44.6877, 10.9147),
        "Spezia":        (44.1153,  9.8290),
        "Torino":        (45.0407,  7.6500),
        "Udinese":       (46.0849, 13.2208),
        "Venezia":       (45.4654, 12.2875),
        "Verona":        (45.4341, 10.9659),
        "Cremonese":     (45.1347, 10.0256),
    },
    "ligue1": {
        "Ajaccio":      (41.9190,  8.7371),
        "Amiens":       (49.8965,  2.2894),
        "Angers":       (47.4699, -0.5457),
        "Auxerre":      (47.8017,  3.5564),
        "Bordeaux":     (44.8283, -0.5714),
        "Brest":        (48.4103, -4.5250),
        "Clermont":     (45.7876,  3.0873),
        "Dijon":        (47.3265,  5.0583),
        "Le Havre":     (49.5059,  0.1290),
        "Lens":         (50.4323,  2.8278),
        "Lille":        (50.6122,  3.1306),
        "Lorient":      (47.7490, -3.3698),
        "Lyon":         (45.7652,  4.9822),
        "Marseille":    (43.2696,  5.3961),
        "Metz":         (49.1114,  6.2153),
        "Monaco":       (43.7275,  7.4158),
        "Montpellier":  (43.6226,  3.8133),
        "Nantes":       (47.2566, -1.5260),
        "Nice":         (43.7054,  7.1926),
        "Paris SG":     (48.8414,  2.2530),
        "Reims":        (49.2468,  4.0256),
        "Rennes":       (48.1072, -1.7133),
        "St Etienne":   (45.4601,  4.3898),
        "Strasbourg":   (48.5580,  7.7531),
        "Toulouse":     (43.5836,  1.4342),
        "Troyes":       (48.2908,  4.0744),
        "Valenciennes": (50.3528,  3.5244),
        "Caen":         (49.1695, -0.3773),
    },
}


def seed():
    for league_id, team_coords in COORDS.items():
        path = CONFIG_DIR / f"{league_id}.json"
        if not path.exists():
            print(f"  [{league_id}] config file not found — skipping")
            continue
        with open(path) as f:
            data: dict = json.load(f)

        updated = 0
        for team, (lat, lng) in team_coords.items():
            if team in data:
                if not data[team].get("lat"):
                    data[team]["lat"] = lat
                    data[team]["lng"] = lng
                    updated += 1
            else:
                # Team may have been relegated; add minimal entry so coords exist
                data[team] = {"lat": lat, "lng": lng}
                updated += 1

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [{league_id}] seeded coordinates for {updated} teams")


if __name__ == "__main__":
    seed()
