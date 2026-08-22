"""The dashboard, run for real against the committed artifacts.

Streamlit only executes a page when a client connects, so a page that crashes still starts a server
happily. `AppTest` runs the script and collects what it produced, which is the only cheap way to
know the thing opens at all.

It reads `models/`, which is committed, so this passes on a fresh clone. It would fail if an
artifact's schema drifted from what the page expects, which is the failure worth catching: the app
holds no numbers of its own, so a rename in `effects_in_seconds` breaks it silently otherwise.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
RESULTS = ROOT / "models" / "decomposition_results.json"


@pytest.fixture(scope="module")
def page():
    if not RESULTS.exists():
        pytest.skip("no committed results to render; run `make train`")
    return AppTest.from_file(str(APP), default_timeout=90).run()


def test_the_page_renders_without_raising(page):
    assert not page.exception, [str(e.value) for e in page.exception]


def test_every_section_is_present(page):
    assert [tab.label for tab in page.tabs] == [
        "Drivers",
        "Cars",
        "Where the variation comes from",
        "Does it hold up?",
    ]


def test_the_headline_is_read_from_the_artifact_rather_than_written_into_the_page(page):
    """The whole design of the app: it holds no numbers of its own, so it cannot contradict the
    results it is displaying."""
    ratio = json.loads(RESULTS.read_text())["car_to_driver_ratio"]
    shown = [metric.value for metric in page.metric]
    assert f"{ratio:.2f}x" in shown


def caption_containing(app, phrase):
    return next(c.value for c in app.caption if phrase in c.value)


def widget(app, kind, label):
    """By label, not by index: adding one control elsewhere on the page renumbers every index."""
    return next(w for w in getattr(app, kind) if w.label == label)


def test_raising_the_session_floor_drops_the_thin_records(page):
    """The filters are the reason this is a page rather than a table in the README, so they are
    worth a test. An untested widget is the same trap as an untested page."""
    assert caption_containing(page, "drivers shown").startswith("43 of 43 drivers")

    after = widget(page, "slider", "Minimum qualifying sessions").set_value(100).run()
    shown = int(caption_containing(after, "drivers shown").split(" of ")[0])
    assert 0 < shown < 43


def test_picking_drivers_compares_only_those(page):
    after = widget(page, "multiselect", "Compare specific drivers").set_value(["VER", "HAM"]).run()
    assert caption_containing(after, "drivers shown").startswith("2 of 43 drivers")


def test_filtering_the_cars_to_one_team_leaves_only_its_seasons(page):
    after = widget(page, "multiselect", "Teams").set_value(["Ferrari"]).run()
    assert caption_containing(after, "team-seasons shown").startswith("8 of 81 team-seasons")
