"""The variant runner. The MCMC refits are checked for what they pass to Bambi, not by fitting:
a real fit is half a minute and the thing worth pinning is the specification, not the sampler.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f1_skill.models import sensitivity


def laps(*rows):
    base = {"iacc": True, "del": False, "qs": "Q1", "wR": False, "season": 2023, "gp": "Miami"}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_a_segment_table_measures_gaps_within_that_segment():
    frame = laps(
        {"team": "Red Bull", "drv": "VER", "time": 90.0, "qs": "Q3"},
        {"team": "Ferrari", "drv": "LEC", "time": 91.8, "qs": "Q3"},
        {"team": "Red Bull", "drv": "VER", "time": 80.0, "qs": "Q1"},
    )
    q3 = sensitivity.segment_table(frame, "Q3")
    assert len(q3) == 2
    assert q3.loc[q3["drv"] == "VER", "gap_to_fastest"].item() == 0.0
    assert q3.loc[q3["drv"] == "LEC", "gap_to_fastest"].item() == pytest.approx(2.0)


def test_a_segment_table_carries_the_columns_the_formula_needs():
    frame = laps({"team": "Red Bull", "drv": "VER", "time": 90.0})
    table = sensitivity.segment_table(frame, "Q1")
    for column in ["team_season", "sqrt_gap", "within_107", "rain_fraction"]:
        assert column in table.columns


def test_matched_rows_keeps_only_driver_sessions_present_in_every_segment():
    """Comparing Q1 with Q3 as they come compares different fields as well as different segments."""
    frame = laps(
        {"team": "Red Bull", "drv": "VER", "time": 90.0, "qs": "Q1"},
        {"team": "Williams", "drv": "ALB", "time": 95.0, "qs": "Q1"},
        {"team": "Red Bull", "drv": "VER", "time": 89.0, "qs": "Q3"},
    )
    tables = {segment: sensitivity.segment_table(frame, segment) for segment in ["Q1", "Q3"]}
    matched = sensitivity.matched_rows(tables)
    assert len(matched["Q1"]) == len(matched["Q3"]) == 1
    assert matched["Q1"]["drv"].tolist() == ["VER"]


def test_matched_rows_returns_the_same_rows_in_the_same_order_for_both_segments():
    frame = laps(
        {"team": "Red Bull", "drv": "VER", "time": 90.0, "qs": "Q1"},
        {"team": "Ferrari", "drv": "LEC", "time": 91.0, "qs": "Q1"},
        {"team": "Red Bull", "drv": "VER", "time": 89.0, "qs": "Q3"},
        {"team": "Ferrari", "drv": "LEC", "time": 90.5, "qs": "Q3"},
    )
    tables = {segment: sensitivity.segment_table(frame, segment) for segment in ["Q1", "Q3"]}
    matched = sensitivity.matched_rows(tables)
    assert matched["Q1"]["drv"].tolist() == matched["Q3"]["drv"].tolist()


def test_effect_sizes_reports_the_ratio_and_its_interval():
    draws = 400
    posterior = xr.Dataset(
        {
            "1|drv_sigma": (("chain", "draw"), np.full((1, draws), 0.1)),
            "1|team_season_sigma": (("chain", "draw"), np.full((1, draws), 0.2)),
            "sigma": (("chain", "draw"), np.full((1, draws), 0.3)),
        }
    )

    class FakeIData:
        pass

    idata = FakeIData()
    idata.posterior = posterior
    sizes = sensitivity.effect_sizes(idata)
    assert sizes["car_over_driver"] == pytest.approx(2.0)
    assert sizes["driver_sd"] == pytest.approx(0.1)
    assert sizes["residual_sd"] == pytest.approx(0.3)


def test_the_log_refit_drops_the_session_fastest_lap_and_says_how_many(monkeypatch):
    """log is undefined at zero, so each session's reference lap falls out. That is a different
    row in Q1 than in Q3, and not even the same number of them."""
    captured = {}

    def fake_fit(table, formula=None, priors=None):
        captured["rows"] = len(table)
        captured["formula"] = formula
        captured["log_gap_finite"] = bool(np.isfinite(table["log_gap"]).all())
        return "idata"

    monkeypatch.setattr(sensitivity, "fit_variant", fake_fit)
    table = pd.DataFrame({"gap_to_fastest": [0.0, 1.0, 2.0, 4.0]})
    _, kept = sensitivity.log_scale_refit(table)

    assert kept == 3
    assert captured["rows"] == 3
    assert captured["log_gap_finite"]
    assert captured["formula"] == sensitivity.LOG_FORMULA


def test_the_variant_fit_uses_the_shipped_specification(monkeypatch):
    """A variant that quietly changed the formula or priors would not be a variant of anything."""
    from f1_skill.models import hierarchical

    seen = {}

    class FakeModel:
        def __init__(self, formula, data, priors):
            seen["formula"] = formula
            seen["priors"] = priors

        def fit(self, **kwargs):
            seen["fit"] = kwargs
            return "idata"

    import bambi

    monkeypatch.setattr(bambi, "Model", FakeModel)
    sensitivity.fit_variant(pd.DataFrame({"sqrt_gap": [1.0]}))

    assert seen["formula"] == hierarchical.FORMULA
    assert seen["priors"] == hierarchical.PRIORS
    assert seen["fit"] == sensitivity.VARIANT_FIT


def test_compare_puts_one_variant_per_row():
    draws = 100

    class FakeIData:
        def __init__(self, car):
            self.posterior = xr.Dataset(
                {
                    "1|drv_sigma": (("chain", "draw"), np.full((1, draws), 0.1)),
                    "1|team_season_sigma": (("chain", "draw"), np.full((1, draws), car)),
                    "sigma": (("chain", "draw"), np.full((1, draws), 0.3)),
                }
            )

    table = sensitivity.compare({"Q1": FakeIData(0.2), "Q3": FakeIData(0.3)})
    assert table.index.tolist() == ["Q1", "Q3"]
    assert table.loc["Q3", "car_over_driver"] > table.loc["Q1", "car_over_driver"]
