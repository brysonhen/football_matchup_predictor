"""
League configuration for all supported competitions.

Each entry maps a short slug (used as directory names and API params) to its
football-data.co.uk CSV code and display metadata.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    id: str        # internal slug — used for dirs and API params
    name: str      # display name
    country: str
    fd_code: str   # football-data.co.uk file code, e.g. "E0"


LEAGUES: dict[str, League] = {
    "pl": League(
        id="pl",
        name="Premier League",
        country="England",
        fd_code="E0",
    ),
    "laliga": League(
        id="laliga",
        name="La Liga",
        country="Spain",
        fd_code="SP1",
    ),
    "bundesliga": League(
        id="bundesliga",
        name="Bundesliga",
        country="Germany",
        fd_code="D1",
    ),
    "seriea": League(
        id="seriea",
        name="Serie A",
        country="Italy",
        fd_code="I1",
    ),
    "ligue1": League(
        id="ligue1",
        name="Ligue 1",
        country="France",
        fd_code="F1",
    ),
}

# Seasons to download for a fresh bootstrap (oldest → newest).
# Matches the 6-season window we use for the PL model.
BOOTSTRAP_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
