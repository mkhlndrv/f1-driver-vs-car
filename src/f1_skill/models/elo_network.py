"""The robustness check: rank drivers a second way and see whether the order survives.

Teammates share a car, so a teammate comparison is the cleanest driver-versus-driver evidence
available. It is also why this cannot be the project's main method: differencing two drivers in the
same car cancels the car out, and constructor strength becomes unestimable.

Three rankings, deliberately: a raw win rate that ignores the opponent, Elo, and Bradley-Terry.
Elo is a single ordered pass with a fixed step size, so it never converges and where a driver ends
up depends partly on the order results arrived. Bradley-Terry is the converged version of the same
idea, which is what says whether Elo's agreement with the main model is the method working or the
walk stopping somewhere flattering.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

INITIAL_RATING = 1500
K_FACTOR = 32  # chess's default; tested against 4 to 64, and the curve is flat from 16 to 48
ELO_SCALE = 400
RIDGE = 1e-3


def head_to_head(table):
    """One row per teammate pair per session. Same car, so the quicker lap wins.

    Three of the 1,475 duels are exact ties, where neither driver set a quicker lap: the sort has
    no tie-break, so those go to whichever driver happens to sort first. Flipping them moves six
    drivers by one place. Small, but decided by row order rather than by pace.
    """
    pairs = []
    for (season, rnd, team), entry in table.groupby(["season", "round", "team"]):
        if len(entry) != 2:
            continue
        faster, slower = entry.sort_values("gap_to_fastest").itertuples(index=False)
        pairs.append(
            {
                "season": season,
                "round": rnd,
                "team": team,
                "winner": faster.drv,
                "loser": slower.drv,
                "margin": slower.gap_to_fastest - faster.gap_to_fastest,
            }
        )
    columns = ["season", "round", "team", "winner", "loser", "margin"]
    if not pairs:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(pairs).sort_values(["season", "round"]).reset_index(drop=True)


def run_elo(pairs, k=K_FACTOR, evaluate_from=None):
    """Walk the results in calendar order. Held-out predictions are made before updating."""
    ratings = {}
    held_out = []

    for row in pairs.itertuples(index=False):
        winner_rating = ratings.get(row.winner, INITIAL_RATING)
        loser_rating = ratings.get(row.loser, INITIAL_RATING)
        expected = 1 / (1 + 10 ** ((loser_rating - winner_rating) / ELO_SCALE))

        if evaluate_from is not None and row.season >= evaluate_from:
            held_out.append(expected)

        surprise = 1 - expected
        ratings[row.winner] = winner_rating + k * surprise
        ratings[row.loser] = loser_rating - k * surprise

    return pd.Series(ratings), np.array(held_out)


def bradley_terry(pairs, ridge=RIDGE):
    """The model Elo approximates, fitted directly rather than walked toward.

    Order-independent and converged, so it separates what the duels say from what a sequential
    update happens to leave behind.
    """
    names = sorted(set(pairs["winner"]) | set(pairs["loser"]))
    position = {name: index for index, name in enumerate(names)}
    winner = pairs["winner"].map(position).to_numpy()
    loser = pairs["loser"].map(position).to_numpy()

    def negative_log_likelihood(strength):
        return np.sum(np.logaddexp(0.0, -(strength[winner] - strength[loser]))) + ridge * np.sum(
            strength**2
        )

    fitted = minimize(negative_log_likelihood, np.zeros(len(names)), method="L-BFGS-B")
    return pd.Series(fitted.x, index=names)


def win_rate(pairs):
    """No model at all: count wins, ignore who they came against. The floor to beat."""
    wins = pairs["winner"].value_counts()
    appearances = pd.concat([pairs["winner"], pairs["loser"]]).value_counts()
    return (wins.reindex(appearances.index).fillna(0) / appearances).rename("win_rate")


def rank_agreement(rating, reference_rank):
    """Spearman between a rating (higher is faster) and a ranking (1 is faster)."""
    shared = reference_rank.index.intersection(rating.index)
    return float(spearmanr(reference_rank.loc[shared], (-rating.loc[shared]).rank()).statistic)
