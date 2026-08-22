"""Fetching qualifying laps from the TracingInsights mirror.

Not FastF1: F1's timing CDN returns 403 to datacenter IPs, so `session.load()` degrades silently
there, keeping the results table while weather, lap timing and race control all fail. The mirror
serves the same data over raw.githubusercontent.com.

Everything here returns raw frames. Cleaning is `cleaning.py`.
"""

import io
import zipfile

import pandas as pd
import requests

MIRROR = "https://raw.githubusercontent.com/TracingInsights"
CONTENTS_API = "https://api.github.com/repos/TracingInsights"
RACE_ARCHIVE = f"{MIRROR}/RaceData/main/data.zip"
SEASONS = range(2018, 2026)
TIMEOUT_SECONDS = 30
ARCHIVE_TIMEOUT_SECONDS = 60

# Sprint points live in their own table. A championship total that leaves them out is not a
# championship total, so the baseline needs both.
ARCHIVE_TABLES = ["races", "results", "sprint_results", "drivers", "constructors"]


def session_json(year, grand_prix, session, filename):
    url = f"{MIRROR}/{year}/main/{requests.utils.quote(grand_prix)}/{requests.utils.quote(session)}/{filename}"
    return requests.get(url, timeout=TIMEOUT_SECONDS).json()


def grand_prix_names(year):
    response = requests.get(f"{CONTENTS_API}/{year}/contents", timeout=TIMEOUT_SECONDS)
    return [
        entry["name"]
        for entry in response.json()
        if entry["type"] == "dir" and "Grand Prix" in entry["name"]
    ]


def qualifying_laps(seasons=SEASONS, on_skip=None):
    """Every qualifying lap across `seasons`, one row per lap, with `season` and `gp` attached.

    A session that fails to load is skipped rather than raising: 14 of 173 are missing the fields
    this project needs, and one unreachable file should not cost the other 172.
    """
    frames = []
    for year in seasons:
        for grand_prix in grand_prix_names(year):
            try:
                laps = session_json(year, grand_prix, "Qualifying", "session_laptimes.json")
            except (requests.RequestException, ValueError) as failure:
                # The two things that actually go wrong: the request fails, or a 404 page comes
                # back and does not parse as JSON. Anything else is a bug and should surface.
                if on_skip:
                    on_skip(year, grand_prix, failure)
                continue
            frames.append(pd.DataFrame(laps).assign(season=year, gp=grand_prix))
    return pd.concat(frames, ignore_index=True)


def race_archive(tables=ARCHIVE_TABLES):
    """Race results, grid and standings from the F1DB/Ergast mirror, for the baseline model."""
    response = requests.get(RACE_ARCHIVE, timeout=ARCHIVE_TIMEOUT_SECONDS)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return {name: pd.read_csv(archive.open(f"{name}.csv")) for name in tables}
