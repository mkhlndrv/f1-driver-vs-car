"""The decomposition is arithmetic on a posterior, so it can be checked against a known one."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f1_skill.models.decomposition import (
    car_to_driver_ratio,
    driver_ranking,
    effects_in_seconds,
    explained_shares,
    observation_weighted,
    spread_in_seconds,
    summary,
    variance_shares,
)

DRAWS = 500


def posterior(car_sd, driver_sd, season_sd, residual_sd):
    """An InferenceData-alike whose group spreads are fixed, so every share is known in advance."""
    constant = {
        "1|team_season_sigma": car_sd,
        "1|drv_sigma": driver_sd,
        "1|season_sigma": season_sd,
        "sigma": residual_sd,
    }
    return xr.Dataset(
        {name: (("chain", "draw"), np.full((1, DRAWS), value)) for name, value in constant.items()}
    )


class FakeIData:
    def __init__(self, posterior):
        self.posterior = posterior


def test_shares_of_all_variation_sum_to_one():
    shares = variance_shares(FakeIData(posterior(0.2, 0.1, 0.1, 0.3)))
    assert shares.sum(axis=1).round(12).eq(1.0).all()


def test_explained_shares_drop_the_residual_and_renormalise():
    shares = variance_shares(FakeIData(posterior(0.2, 0.1, 0.1, 0.3)))
    explained = explained_shares(shares)
    assert list(explained.columns) == ["car", "driver", "season"]
    assert explained.sum(axis=1).round(12).eq(1.0).all()


def test_a_component_twice_the_spread_of_another_is_four_times_the_variance():
    """Shares are variances, so doubling a standard deviation quadruples the share."""
    shares = variance_shares(FakeIData(posterior(0.2, 0.1, 0.0, 0.0)))
    assert car_to_driver_ratio(shares).median() == pytest.approx(4.0)


def test_the_ratio_is_car_over_driver_not_the_other_way_round():
    """The label on this was printed inverted in the notebook for months."""
    shares = variance_shares(FakeIData(posterior(0.3, 0.1, 0.1, 0.1)))
    assert car_to_driver_ratio(shares).median() > 1


def test_the_explained_denominator_is_larger_than_the_total_denominator():
    """67% of explained and a smaller share of everything are the same result, twice."""
    idata = FakeIData(posterior(0.23, 0.138, 0.078, 0.296))
    shares = variance_shares(idata)
    explained = explained_shares(shares)
    assert explained["car"].median() > shares["car"].median()


def test_equal_spreads_split_evenly():
    shares = variance_shares(FakeIData(posterior(0.2, 0.2, 0.2, 0.2)))
    assert shares.median().round(6).eq(0.25).all()
    assert pd.Series(car_to_driver_ratio(shares)).median() == pytest.approx(1.0)


def test_the_driver_ranking_uses_the_median_not_the_mean():
    """The convention matters: on the real posterior, mean instead of median moves the rank
    correlation against the Elo ranking by about 0.004."""
    skewed = np.concatenate([np.full(90, 0.0), np.full(10, 10.0)])  # median 0, mean 1
    steady = np.full(100, 0.5)
    posterior = xr.Dataset(
        {
            "1|drv": (
                ("chain", "draw", "drv__factor_dim"),
                np.stack([skewed, steady], axis=-1)[None, :, :],
            )
        },
        coords={"drv__factor_dim": ["SKEWED", "STEADY"]},
    )
    ranking = driver_ranking(FakeIData(posterior))
    # by median SKEWED is quicker (0.0 < 0.5); by mean it would be slower (1.0 > 0.5)
    assert ranking["SKEWED"] < ranking["STEADY"]


def effects_posterior():
    """Two drivers and two cars with known effects, so seconds and covariance are predictable."""
    draws = 200
    return xr.Dataset(
        {
            "Intercept": (("chain", "draw"), np.full((1, draws), 1.0)),
            "1|drv": (
                ("chain", "draw", "drv__factor_dim"),
                np.tile([-0.1, 0.1], (1, draws, 1)),
            ),
            "1|team_season": (
                ("chain", "draw", "team_season__factor_dim"),
                np.tile([-0.2, 0.2], (1, draws, 1)),
            ),
        },
        coords={
            "drv__factor_dim": ["FAST", "SLOW"],
            "team_season__factor_dim": ["GOOD 2023", "BAD 2023"],
        },
    )


def paired_table():
    return pd.DataFrame(
        {
            "drv": ["FAST", "FAST", "SLOW", "SLOW"],
            "team_season": ["GOOD 2023", "GOOD 2023", "BAD 2023", "BAD 2023"],
            "best_lap": [90.0, 90.0, 90.0, 90.0],
        }
    )


def test_spread_in_seconds_converts_at_a_real_operating_point():
    """The model works on a square root, so an effect is worth more or less depending on where it
    starts. (1.1^2 - 0.9^2)/100 * 90 = 0.36s."""
    seconds = spread_in_seconds(
        FakeIData(effects_posterior()), paired_table(), "1|drv", "drv__factor_dim"
    )
    assert float(np.median(seconds)) == pytest.approx(0.36, abs=0.01)


def test_the_car_spread_is_wider_than_the_driver_spread_in_seconds_too():
    idata, table = FakeIData(effects_posterior()), paired_table()
    driver = np.median(spread_in_seconds(idata, table, "1|drv", "drv__factor_dim"))
    car = np.median(spread_in_seconds(idata, table, "1|team_season", "team_season__factor_dim"))
    assert car > driver


def test_observation_weighted_finds_the_overlap_the_four_way_split_assumes_away():
    """Fast drivers are in good cars here, so driver and car effects correlate perfectly."""
    observed = observation_weighted(FakeIData(effects_posterior()), paired_table())
    assert observed["correlation"].median() == pytest.approx(1.0)
    assert observed["covariance"].median() > 0


def test_effects_in_seconds_measures_each_level_against_an_average_one():
    """FAST sits 0.1 below the intercept: (1.0^2 - 0.9^2)/100 * 90 = 0.171s quicker than average,
    and SLOW is not the mirror image, because squaring is not a straight line."""
    ratings = effects_in_seconds(
        FakeIData(effects_posterior()), paired_table(), "1|drv", "drv__factor_dim"
    )
    assert ratings.index.tolist() == ["FAST", "SLOW"]
    assert ratings.loc["FAST", "seconds"] == pytest.approx(-0.171, abs=0.001)
    assert ratings.loc["SLOW", "seconds"] == pytest.approx(0.189, abs=0.001)
    assert abs(ratings.loc["SLOW", "seconds"]) > abs(ratings.loc["FAST", "seconds"])


def test_the_per_level_effects_and_the_best_to_worst_spread_agree():
    """Two views of the same conversion. They are not the same statistic (one differences the
    medians, the other takes the median of the differences), so this only pins the scale."""
    idata, table = FakeIData(effects_posterior()), paired_table()
    ratings = effects_in_seconds(idata, table, "1|drv", "drv__factor_dim")
    spread = float(np.median(spread_in_seconds(idata, table, "1|drv", "drv__factor_dim")))
    assert ratings["seconds"].max() - ratings["seconds"].min() == pytest.approx(spread, abs=1e-9)


def test_effects_in_seconds_carries_an_interval_around_every_level():
    ratings = effects_in_seconds(
        FakeIData(effects_posterior()), paired_table(), "1|team_season", "team_season__factor_dim"
    )
    assert (ratings["lower_89"] <= ratings["seconds"]).all()
    assert (ratings["seconds"] <= ratings["upper_89"]).all()


def test_summary_reports_both_denominators_and_the_ratio():
    result = summary(FakeIData(posterior(0.2, 0.1, 0.1, 0.3)))
    assert set(result) == {"shares", "explained", "ratio_median", "ratio_89"}
    assert result["ratio_median"] == pytest.approx(4.0)
    assert len(result["ratio_89"]) == 2
