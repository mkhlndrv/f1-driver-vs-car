"""Variant runs: does the answer survive the choices that were made along the way?

Every function here refits the primary model on a different slice or a different scale. That is
expensive, roughly half a minute per fit, so nothing in this module runs as part of the default
pipeline. It exists so the checks can be reproduced, not so they run every time.

The prediction was written down before any of it ran: the gaps between cars should widen from Q1 to
Q3, and the gaps between drivers should stay about where they are. Only the first half held.
"""

import numpy as np
import pandas as pd

from f1_skill.models.hierarchical import FORMULA, PRIORS

# Fewer draws than the shipped fit. These are comparisons between variants, not the headline.
VARIANT_FIT = {"draws": 1500, "tune": 1000, "chains": 4, "random_seed": 0, "target_accept": 0.99}

# On a log scale a multiplicative level shift lands entirely in the intercept, so it cannot inflate
# a variance component. Wider than the sqrt priors because log gaps are wider.
LOG_FORMULA = "log_gap ~ 1 + (1|drv) + (1|team_season) + (1|season)"


def _log_priors():
    import bambi as bmb

    return {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=2),
        "1|drv": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
        "1|team_season": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
        "1|season": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
        "sigma": bmb.Prior("HalfNormal", sigma=2),
    }


def segment_table(laps, segment):
    """The same pace table as the main pipeline, but measured within any chosen segment."""
    from f1_skill.cleaning import rain_fraction, valid_segment_laps
    from f1_skill.features import PERCENT_OVER_FASTEST

    valid = valid_segment_laps(laps, segment)
    table = (
        valid.groupby(["season", "gp", "team", "drv"])["time"]
        .agg(best_lap="min", n_valid_laps="size")
        .reset_index()
    )
    fastest = table.groupby(["season", "gp"])["best_lap"].transform("min")
    table["gap_to_fastest"] = (table["best_lap"] - fastest) / fastest * 100
    table["team_season"] = table["team"] + " " + table["season"].astype(str)
    table["sqrt_gap"] = np.sqrt(table["gap_to_fastest"])
    table = table.join(rain_fraction(laps).rename("rain_fraction"), on=["season", "gp"])
    table["within_107"] = table["gap_to_fastest"] <= PERCENT_OVER_FASTEST
    return table


def fit_variant(table, formula=FORMULA, priors=None):
    import bambi as bmb

    model = bmb.Model(formula, data=table, priors=priors or PRIORS)
    return model.fit(**VARIANT_FIT)


def effect_sizes(idata):
    posterior = idata.posterior
    driver = posterior["1|drv_sigma"].values.flatten()
    car = posterior["1|team_season_sigma"].values.flatten()
    ratio = car / driver
    return {
        "driver_sd": float(np.median(driver)),
        "car_sd": float(np.median(car)),
        "residual_sd": float(np.median(posterior["sigma"].values.flatten())),
        "car_over_driver": float(np.median(ratio)),
        "ratio_lower": float(np.percentile(ratio, 5.5)),
        "ratio_upper": float(np.percentile(ratio, 94.5)),
    }


def matched_rows(tables, segments=("Q1", "Q3")):
    """The driver-sessions that set a valid lap in every named segment.

    Comparing Q1 against Q3 as they come compares different fields as well as different segments:
    Q3 holds ten cars rather than twenty. Holding the rows constant is what isolates the segment.
    """
    keys = ["season", "gp", "team", "drv"]
    shared = None
    for segment in segments:
        rows = tables[segment][tables[segment]["within_107"]][keys]
        shared = rows if shared is None else shared.merge(rows, on=keys)
    return {
        segment: tables[segment].merge(shared, on=keys).sort_values(keys).reset_index(drop=True)
        for segment in segments
    }


def log_scale_refit(table):
    """The same comparison with the square root replaced by a log.

    `log` is undefined at zero, so this silently drops each session's fastest lap. That is a
    different row in Q1 than in Q3, and not even the same number of them, so it breaks the matching
    the comparison exists to preserve. Read it as evidence that the widening is not an artefact of
    the square root, which is what it was run for, and not as a second estimate of the effect.
    """
    positive = table[table["gap_to_fastest"] > 0].copy()
    positive["log_gap"] = np.log(positive["gap_to_fastest"])
    return fit_variant(positive, formula=LOG_FORMULA, priors=_log_priors()), len(positive)


def compare(fits):
    return pd.DataFrame({name: effect_sizes(idata) for name, idata in fits.items()}).T
