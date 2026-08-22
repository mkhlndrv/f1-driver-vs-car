"""Fetch behaviour, without touching the network.

The one thing worth pinning here is the skip: a session that fails to load is skipped rather than
raising, because 14 of 173 are missing the fields this project needs and one unreachable file
should not cost the other 172. The catch is deliberately narrow, so a bug still surfaces.
"""

import io
import zipfile

import pandas as pd
import pytest
import requests

from f1_skill import data


def test_a_grand_prix_name_with_a_space_is_url_encoded():
    calls = {}

    def fake_get(url, timeout=None):
        calls["url"] = url

        class Response:
            @staticmethod
            def json():
                return []

        return Response()

    original = requests.get
    requests.get = fake_get
    try:
        data.session_json(2023, "Miami Grand Prix", "Qualifying", "session_laptimes.json")
    finally:
        requests.get = original
    assert "Miami%20Grand%20Prix" in calls["url"]
    assert "/2023/main/" in calls["url"]


def test_only_grand_prix_directories_are_treated_as_sessions(monkeypatch):
    listing = [
        {"name": "Miami Grand Prix", "type": "dir"},
        {"name": "README.md", "type": "file"},
        {"name": "scripts", "type": "dir"},
    ]

    class Response:
        @staticmethod
        def json():
            return listing

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: Response())
    assert data.grand_prix_names(2023) == ["Miami Grand Prix"]


def test_an_unreachable_session_is_skipped_and_reported(monkeypatch):
    monkeypatch.setattr(data, "grand_prix_names", lambda year: ["Good GP", "Broken GP"])

    def fake_session(year, grand_prix, session, filename):
        if grand_prix == "Broken GP":
            raise requests.RequestException("404")
        return [{"time": 90.0, "drv": "VER"}]

    monkeypatch.setattr(data, "session_json", fake_session)
    skipped = []
    laps = data.qualifying_laps(seasons=[2023], on_skip=lambda *args: skipped.append(args[1]))

    assert skipped == ["Broken GP"]
    assert len(laps) == 1
    assert laps["gp"].tolist() == ["Good GP"]


def test_a_response_that_is_not_json_is_skipped_too(monkeypatch):
    """A 404 comes back as an HTML page, which raises on .json() rather than on the request."""
    monkeypatch.setattr(data, "grand_prix_names", lambda year: ["Broken GP"])

    def fake_session(*args):
        raise ValueError("Expecting value: line 1 column 1")

    monkeypatch.setattr(data, "session_json", fake_session)
    skipped = []
    with pytest.raises(ValueError):
        # every session skipped means nothing to concatenate, which is a real failure, not a skip
        data.qualifying_laps(seasons=[2023], on_skip=lambda *args: skipped.append(args[1]))
    assert skipped == ["Broken GP"]


def test_a_bug_in_the_fetch_is_not_swallowed_as_a_skip(monkeypatch):
    """The narrow catch is the point: an AttributeError here is a defect, not a missing session."""
    monkeypatch.setattr(data, "grand_prix_names", lambda year: ["Any GP"])

    def fake_session(*args):
        raise AttributeError("typo in the caller")

    monkeypatch.setattr(data, "session_json", fake_session)
    with pytest.raises(AttributeError):
        data.qualifying_laps(seasons=[2023])


def test_the_season_and_grand_prix_travel_with_every_lap(monkeypatch):
    monkeypatch.setattr(data, "grand_prix_names", lambda year: ["Miami Grand Prix"])
    monkeypatch.setattr(data, "session_json", lambda *args: [{"time": 90.0}, {"time": 91.0}])
    laps = data.qualifying_laps(seasons=[2023])
    assert laps["season"].tolist() == [2023, 2023]
    assert laps["gp"].tolist() == ["Miami Grand Prix"] * 2
    assert isinstance(laps, pd.DataFrame)


def test_the_race_archive_reads_the_named_tables_out_of_the_zip(monkeypatch):
    """The archive arrives as one zip of CSVs; only the tables the baseline needs are read."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("races.csv", "raceId,year\n1,2023\n")
        archive.writestr("results.csv", "raceId,driverId\n1,44\n")
        archive.writestr("qualifying.csv", "raceId\n1\n")

    class Response:
        content = buffer.getvalue()

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: Response())
    tables = data.race_archive(["races", "results"])

    assert sorted(tables) == ["races", "results"]
    assert tables["races"]["year"].tolist() == [2023]
    assert tables["results"]["driverId"].tolist() == [44]
