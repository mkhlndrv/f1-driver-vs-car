import numpy as np
import pandas as pd
import pytest

from f1_skill.models.elo_network import (
    INITIAL_RATING,
    K_FACTOR,
    bradley_terry,
    head_to_head,
    rank_agreement,
    run_elo,
    win_rate,
)


def sessions(*rows):
    """rows: (season, round, team, driver, gap)."""
    return pd.DataFrame(
        [
            {"season": s, "round": r, "team": t, "drv": d, "gap_to_fastest": g}
            for s, r, t, d, g in rows
        ]
    )


def duels(*winners_losers):
    return pd.DataFrame(
        [
            {
                "season": 2023,
                "round": index + 1,
                "team": "T",
                "winner": w,
                "loser": l,
                "margin": 0.1,
            }
            for index, (w, l) in enumerate(winners_losers)
        ]
    )


def test_the_quicker_teammate_wins_the_duel():
    pairs = head_to_head(
        sessions((2023, 1, "Red Bull", "VER", 0.0), (2023, 1, "Red Bull", "PER", 0.4))
    )
    assert pairs["winner"].tolist() == ["VER"]
    assert pairs["loser"].tolist() == ["PER"]
    assert pairs["margin"].tolist() == [pytest.approx(0.4)]


def test_a_driver_without_a_teammate_that_session_produces_no_duel():
    pairs = head_to_head(sessions((2023, 1, "Red Bull", "VER", 0.0)))
    assert pairs.empty


def test_a_tied_session_is_decided_by_row_order_not_by_pace():
    """Three real duels are exact ties. Recorded so the behaviour cannot change unnoticed."""
    tied = sessions((2023, 1, "Mercedes", "HAM", 1.0), (2023, 1, "Mercedes", "RUS", 1.0))
    assert head_to_head(tied)["winner"].item() == "HAM"
    assert head_to_head(tied.iloc[::-1])["winner"].item() == "RUS"


def test_elo_moves_both_drivers_by_the_same_amount():
    ratings, _ = run_elo(duels(("A", "B")))
    assert ratings["A"] - INITIAL_RATING == pytest.approx(INITIAL_RATING - ratings["B"])


def test_beating_an_equal_opponent_moves_half_the_k_factor():
    ratings, _ = run_elo(duels(("A", "B")))
    assert ratings["A"] == pytest.approx(INITIAL_RATING + K_FACTOR / 2)


def test_elo_depends_on_the_order_results_arrived():
    """Not a defect: a single ordered pass with a fixed step never converges, and saying so is
    the reason Bradley-Terry is fitted alongside it."""
    forwards, _ = run_elo(duels(("A", "B"), ("B", "C"), ("A", "C")))
    backwards, _ = run_elo(duels(("A", "C"), ("B", "C"), ("A", "B")))
    assert forwards["B"] != pytest.approx(backwards["B"])


def test_bradley_terry_does_not_depend_on_order():
    forward = duels(("A", "B"), ("B", "C"), ("A", "C"))
    reversed_order = forward.iloc[::-1].reset_index(drop=True)
    first, second = bradley_terry(forward), bradley_terry(reversed_order)
    # The ranking is what this is used for, and it does not move at all. The strengths agree to
    # the optimiser's tolerance rather than to the last bit, which is a property of L-BFGS-B and
    # not of the model.
    assert first.rank().equals(second.rank())
    assert np.allclose(first, second, atol=1e-4)


def test_bradley_terry_ranks_an_undefeated_driver_top():
    strengths = bradley_terry(duels(("A", "B"), ("A", "C"), ("B", "C")))
    assert strengths.idxmax() == "A"
    assert strengths.idxmin() == "C"


def test_win_rate_ignores_who_the_opponent_was():
    rates = win_rate(duels(("A", "B"), ("A", "B"), ("B", "A")))
    assert rates["A"] == pytest.approx(2 / 3)
    assert rates["B"] == pytest.approx(1 / 3)


def test_held_out_predictions_are_made_before_the_result_updates_the_rating():
    pairs = pd.DataFrame(
        [
            {"season": 2023, "round": 1, "team": "T", "winner": "A", "loser": "B", "margin": 0.1},
            {"season": 2024, "round": 1, "team": "T", "winner": "A", "loser": "B", "margin": 0.1},
        ]
    )
    _, held_out = run_elo(pairs, evaluate_from=2024)
    assert len(held_out) == 1
    # by 2024 A leads B, so the prediction is above a coin flip but was made before the update
    assert 0.5 < held_out[0] < 1.0
    assert not np.isclose(held_out[0], 0.5)


def ranking(*drivers):
    """A hierarchical ranking: 1 is the fastest."""
    return pd.Series(range(1, len(drivers) + 1), index=list(drivers))


def test_rank_agreement_reads_a_rating_the_right_way_up():
    """The two orderings run in opposite directions, so getting the sign wrong here would report
    perfect agreement as perfect disagreement."""
    ratings = pd.Series({"A": 1600.0, "B": 1500.0, "C": 1400.0})
    assert rank_agreement(ratings, ranking("A", "B", "C")) == pytest.approx(1.0)
    assert rank_agreement(ratings, ranking("C", "B", "A")) == pytest.approx(-1.0)


def test_rank_agreement_compares_only_the_drivers_both_sides_have():
    """Elo knows only drivers with a teammate duel; the model ranks every driver in the data."""
    ratings = pd.Series({"A": 1600.0, "B": 1500.0})
    assert rank_agreement(ratings, ranking("A", "B", "NEVER_RACED_A_TEAMMATE")) == pytest.approx(
        1.0
    )
