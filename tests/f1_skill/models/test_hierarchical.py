"""The known-answer checks for the hand-written diagnostics.

These replace `az.summary`, which returns NaN for every parameter of these fits in the environment
the analysis ran in. A hand-written replacement for a library function has to earn its place, and
this is where it does: three cases whose answer is known before the code runs.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f1_skill.models import hierarchical
from f1_skill.models.hierarchical import (
    CONVERGENCE_THRESHOLD,
    convergence,
    effective_size,
    fit,
    model_table,
    split_rhat,
)

CHAINS, LENGTH = 4, 2000


def chains(seed=0):
    return np.random.default_rng(seed).standard_normal((CHAINS, LENGTH))


def test_independent_draws_return_rhat_one():
    assert split_rhat(chains()) == pytest.approx(1.0, abs=0.005)


def test_independent_draws_recover_almost_the_full_sample():
    assert effective_size(chains()) > 0.9 * CHAINS * LENGTH


def test_chains_parked_apart_are_caught():
    """Four chains at different means is the failure R-hat exists to detect."""
    offsets = np.linspace(-2, 2, CHAINS).reshape(CHAINS, 1)
    parked = chains() + offsets
    assert split_rhat(parked) > 1.5
    assert split_rhat(parked) > CONVERGENCE_THRESHOLD


def test_correlated_draws_report_the_sample_size_theory_predicts():
    """For an AR(1), the effective size is n(1-rho)/(1+rho). Nothing here is fitted to that."""
    rho = 0.8
    noise = chains(seed=1)
    series = np.zeros((CHAINS, LENGTH))
    for step in range(1, LENGTH):
        series[:, step] = rho * series[:, step - 1] + noise[:, step] * np.sqrt(1 - rho**2)

    theoretical = CHAINS * LENGTH * (1 - rho) / (1 + rho)
    assert effective_size(series) == pytest.approx(theoretical, rel=0.1)


def test_correlated_draws_report_far_fewer_than_the_raw_count():
    rho = 0.8
    noise = chains(seed=1)
    series = np.zeros((CHAINS, LENGTH))
    for step in range(1, LENGTH):
        series[:, step] = rho * series[:, step - 1] + noise[:, step] * np.sqrt(1 - rho**2)
    assert effective_size(series) < 0.2 * CHAINS * LENGTH


def test_the_model_table_applies_the_107_rule_and_builds_both_derived_columns():
    pace = pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "gp": ["Miami Grand Prix"] * 3,
            "team": ["Red Bull", "Red Bull", "Williams"],
            "drv": ["VER", "PER", "ALB"],
            "gap_to_fastest": [0.0, 1.0, 9.0],
            "within_107": [True, True, False],
        }
    )
    table = model_table(pace)
    assert len(table) == 2
    assert table["team_season"].tolist() == ["Red Bull 2023", "Red Bull 2023"]
    assert table["sqrt_gap"].tolist() == [0.0, 1.0]


def test_the_square_root_cannot_produce_an_impossible_gap():
    """The reason this transform was chosen: squaring back is never negative."""
    pace = pd.DataFrame(
        {
            "season": [2023, 2023],
            "gp": ["Miami Grand Prix"] * 2,
            "team": ["Red Bull"] * 2,
            "drv": ["VER", "PER"],
            "gap_to_fastest": [0.0, 4.0],
            "within_107": [True, True],
        }
    )
    assert (model_table(pace)["sqrt_gap"] ** 2 >= 0).all()


def test_convergence_flags_a_fit_that_has_not_converged():
    offsets = np.linspace(-2, 2, CHAINS).reshape(CHAINS, 1)
    posterior = xr.Dataset(
        {
            "sigma": (("chain", "draw"), chains()),
            "1|drv_sigma": (("chain", "draw"), chains(seed=2) + offsets),
        }
    )
    report = convergence(FakeIData(posterior, diverging=0))
    assert report["worst_rhat_parameter"] == "1|drv_sigma"
    assert not report["converged"]


def test_convergence_passes_a_healthy_fit_and_counts_divergences():
    posterior = xr.Dataset(
        {"sigma": (("chain", "draw"), chains()), "Intercept": (("chain", "draw"), chains(seed=3))}
    )
    report = convergence(FakeIData(posterior, diverging=7))
    assert report["converged"]
    assert report["divergences"] == 7
    assert report["lowest_ess"] > 0.9 * CHAINS * LENGTH


def test_the_fit_asks_for_the_log_likelihood_at_fit_time(monkeypatch):
    """Bambi's non-centred parameterisation cannot recover it afterwards, and without it the
    predictive checks in notebook 07 have nothing to run on. Sampling itself is not exercised
    here: it takes half a minute and is the sampler's business, not this module's."""
    seen = {}

    class FakeModel:
        def __init__(self, formula, data, priors):
            seen["formula"] = formula
            seen["priors"] = priors

        def fit(self, **kwargs):
            seen["fit"] = kwargs
            return "idata"

    monkeypatch.setattr(hierarchical.bmb, "Model", FakeModel)
    table = pd.DataFrame({"sqrt_gap": [1.0]})

    model, idata = fit(table)
    assert idata == "idata"
    assert isinstance(model, FakeModel)
    assert seen["formula"] == hierarchical.FORMULA
    assert seen["priors"] is hierarchical.PRIORS
    assert seen["fit"]["idata_kwargs"] == {"log_likelihood": True}
    assert {key: seen["fit"][key] for key in hierarchical.FIT} == hierarchical.FIT

    fit(table, log_likelihood=False)
    assert seen["fit"]["idata_kwargs"] == {"log_likelihood": False}


class FakeIData:
    def __init__(self, posterior, diverging):
        self.posterior = posterior
        self.sample_stats = xr.Dataset(
            {"diverging": (("draw",), np.full(diverging or 1, diverging > 0))}
        )
