"""Turning the fit into shares: how much of qualifying is the driver, and how much the car.

Variance rather than standard deviation, because variances of separate causes add and can
therefore be read as shares of a total. That is why the headline is 2.8 and the group spreads are
0.230 and 0.138: same result, squared.

Two things this deliberately reports side by side. The four-way split assumes the driver and car
terms are independent, and they are not: good drivers are hired by good teams. The
observation-weighted version below measures that overlap instead of assuming it away, and the two
ratios agreeing is what says the headline survives the assumption even though the split is
incomplete.
"""

import numpy as np
import pandas as pd

INTERVAL = [0.055, 0.5, 0.945]
INTERVAL_NAMES = ["lower_89", "median", "upper_89"]

# np.cov across 3,008 rows for every draw is not free and adds nothing: every sixteenth is plenty
# to summarise a posterior.
COVARIANCE_STRIDE = 16

COMPONENTS = {
    "car": "1|team_season_sigma",
    "driver": "1|drv_sigma",
    "season": "1|season_sigma",
    "unexplained": "sigma",
}


def _summarise(frame):
    return frame.quantile(INTERVAL).T.set_axis(INTERVAL_NAMES, axis=1)


def variance_shares(idata):
    """Each component's share of all the variation, including what the model cannot explain."""
    posterior = idata.posterior
    variances = {
        name: posterior[variable].values.flatten() ** 2 for name, variable in COMPONENTS.items()
    }
    total = sum(variances.values())
    return pd.DataFrame({name: values / total for name, values in variances.items()})


def explained_shares(shares):
    """The same split over what the model does account for, dropping the residual.

    Both denominators are worth reporting and they are not the same number: the car is 67% of the
    variation the model explains and a smaller share of all variation.
    """
    explained = shares[["car", "driver", "season"]]
    return explained.div(explained.sum(axis=1), axis=0)


def car_to_driver_ratio(shares):
    return shares["car"] / shares["driver"]


def summary(idata):
    shares = variance_shares(idata)
    ratio = car_to_driver_ratio(shares)
    return {
        "shares": _summarise(shares),
        "explained": _summarise(explained_shares(shares)),
        "ratio_median": float(ratio.median()),
        "ratio_89": [float(ratio.quantile(0.055)), float(ratio.quantile(0.945))],
    }


def _levels(posterior, term, dimension):
    draws = posterior[term].values.reshape(-1, posterior[term].shape[-1])
    return draws, posterior[term].coords[dimension].values


def spread_in_seconds(idata, table, term, dimension):
    """Best level against worst, in lap time, at a typical session with everything else average.

    The model works on the square root of the gap, and squaring is not a straight line, so an
    effect is worth more or less depending on where it starts. Rather than pick one conversion
    factor, evaluate at a real operating point and convert there.
    """
    posterior = idata.posterior
    intercept = posterior["Intercept"].values.flatten()
    draws, _ = _levels(posterior, term, dimension)
    lap_seconds = table["best_lap"].median()

    best = np.clip(intercept + draws.min(axis=1), 0, None) ** 2
    worst = np.clip(intercept + draws.max(axis=1), 0, None) ** 2
    return (worst - best) / 100 * lap_seconds


def effects_in_seconds(idata, table, term, dimension):
    """Each level against an average one, in seconds on a typical lap, with 89% intervals.

    Same conversion as `spread_in_seconds`, kept per level rather than collapsed to best-minus-
    worst, because the interval widths are the point: a driver with 17 sessions has a range
    several times wider than one with 154.
    """
    posterior = idata.posterior
    intercept = posterior["Intercept"].values.flatten()[:, None]
    draws, names = _levels(posterior, term, dimension)

    average = intercept**2
    shifted = np.clip(intercept + draws, 0, None) ** 2
    seconds = (shifted - average) / 100 * table["best_lap"].median()

    return pd.DataFrame(
        {
            "seconds": np.median(seconds, axis=0),
            "lower_89": np.percentile(seconds, 5.5, axis=0),
            "upper_89": np.percentile(seconds, 94.5, axis=0),
        },
        index=names,
    )


def observation_weighted(idata, table):
    """The split as the drivers and cars actually paired up, one row per driver-session.

    The four-way split counts every driver once, so a driver with one session weighs the same as
    one with 154. This weights by the laps actually driven, and lets the overlap between driver
    quality and car quality appear as a term of its own instead of being forced to zero.
    """
    posterior = idata.posterior
    driver_draws, driver_names = _levels(posterior, "1|drv", "drv__factor_dim")
    car_draws, car_names = _levels(posterior, "1|team_season", "team_season__factor_dim")

    driver_index = pd.Categorical(table["drv"], categories=driver_names).codes
    car_index = pd.Categorical(table["team_season"], categories=car_names).codes

    rows = []
    for draw in range(0, driver_draws.shape[0], COVARIANCE_STRIDE):
        per_row_driver = driver_draws[draw][driver_index]
        per_row_car = car_draws[draw][car_index]
        rows.append(
            {
                "var_driver": per_row_driver.var(ddof=1),
                "var_car": per_row_car.var(ddof=1),
                "covariance": np.cov(per_row_driver, per_row_car, ddof=1)[0, 1],
                "correlation": np.corrcoef(per_row_driver, per_row_car)[0, 1],
            }
        )

    observed = pd.DataFrame(rows)
    observed["twice_cov"] = 2 * observed["covariance"]
    observed["total"] = observed["var_driver"] + observed["var_car"] + observed["twice_cov"]
    return observed


def driver_ranking(idata):
    """Drivers ordered fastest first, from the posterior median of each driver effect.

    The median, not the mean. On this posterior the choice moves the rank correlation against the
    Elo ranking by about 0.004, which is small but enough to make two runs disagree if the
    convention is re-derived rather than shared.
    """
    posterior = idata.posterior
    draws, names = _levels(posterior, "1|drv", "drv__factor_dim")
    # lower sqrt-gap effect means quicker, so rank 1 is the fastest driver
    return pd.Series(np.median(draws, axis=0), index=names).rank()
