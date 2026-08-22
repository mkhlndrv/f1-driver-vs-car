"""The primary model: a driver term and a car term estimated together, plus its diagnostics.

Fitted on the square root of the percentage gap. The untransformed gap is floored at zero and
trails right, which a symmetric bell curve cannot describe: it puts 8.6% of its simulated laps
below zero, where a gap cannot go. The square root is the only candidate that produces no
impossible values, and it does so by arithmetic rather than by tuning.

The car term is per season (`team_season`), not per team. Forcing Mercedes 2019 and Mercedes 2022
into one number leaves the car's year-to-year swings nowhere to go except the driver term, which
credited drivers with performance that belonged to their cars.

R-hat and effective sample size are computed here rather than read from `az.summary`, which returns
NaN for every parameter of these fits in the Kaggle image the analysis ran on. Both are checked
against cases whose answer is known in advance, in the tests.
"""

import bambi as bmb
import numpy as np

FORMULA = "sqrt_gap ~ 1 + (1|drv) + (1|team_season) + (1|season)"

PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=1, sigma=0.5),
    "1|drv": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.3)),
    "1|team_season": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.5)),
    "1|season": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.3)),
    "sigma": bmb.Prior("HalfNormal", sigma=0.5),
}

# target_accept is 0.99 because 0.95 left 13 divergences on this geometry.
FIT = {"draws": 2000, "tune": 1000, "chains": 4, "random_seed": 0, "target_accept": 0.99}

CONVERGENCE_THRESHOLD = 1.01


def model_table(pace):
    table = pace[pace["within_107"]].copy()
    table["team_season"] = table["team"] + " " + table["season"].astype(str)
    table["sqrt_gap"] = np.sqrt(table["gap_to_fastest"])
    return table


def fit(table, log_likelihood=True):
    """Bambi's non-centred parameterisation cannot recover the log-likelihood after the fact,
    so it has to be asked for at fit time if the predictive checks are going to run."""
    model = bmb.Model(FORMULA, data=table, priors=PRIORS)
    return model, model.fit(**FIT, idata_kwargs={"log_likelihood": log_likelihood})


def split_rhat(draws):
    """Split R-hat: each chain halved, then within-chain variance against between-chain."""
    _, length = draws.shape
    half = length // 2
    pieces = np.concatenate([draws[:, :half], draws[:, half : 2 * half]])
    within = pieces.var(axis=1, ddof=1).mean()
    between = half * pieces.mean(axis=1).var(ddof=1)
    return np.sqrt(((half - 1) / half * within + between / half) / within)


def effective_size(draws):
    """ESS from the autocorrelation, truncated by Geyer's initial positive sequence.

    This normalises by the variance within each chain rather than by the pooled variance that
    includes disagreement between chains, so where R-hat sits above 1.01 it reads a little
    optimistically. On the shipped fit it returns 1084 for `1|season_sigma` where arviz's
    rank-normalised `ess_bulk` returns 908. Both are defensible; they are not the same estimator,
    and the difference is largest exactly where convergence is weakest.
    """
    chains, length = draws.shape
    centred = draws - draws.mean(axis=1, keepdims=True)
    padded = 2 ** int(np.ceil(np.log2(2 * length)))
    spectrum = np.fft.rfft(centred, n=padded, axis=1)
    autocovariance = (
        np.fft.irfft(spectrum * np.conjugate(spectrum), n=padded, axis=1)[:, :length].real / length
    )
    correlation = autocovariance.mean(axis=0)
    correlation = correlation / correlation[0]
    total = 0.0
    for lag in range(1, length - 1, 2):
        pair = correlation[lag] + correlation[lag + 1]
        if pair < 0:
            break
        total += pair
    return chains * length / (1 + 2 * total)


def convergence(idata):
    """Worst R-hat and lowest ESS across the scalar parameters.

    Scalars only: the intercept, the residual spread and the three group spreads. The ~130
    individual driver and car effects are not included, so "worst" means worst among those.
    """
    scalars = [
        name
        for name in idata.posterior.data_vars
        if idata.posterior[name].dims == ("chain", "draw")
    ]
    rhats = {name: split_rhat(idata.posterior[name].values) for name in scalars}
    sizes = {name: effective_size(idata.posterior[name].values) for name in scalars}
    worst = max(rhats, key=rhats.get)
    return {
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "worst_rhat": rhats[worst],
        "worst_rhat_parameter": worst,
        "lowest_ess": min(sizes.values()),
        "converged": rhats[worst] <= CONVERGENCE_THRESHOLD,
    }
