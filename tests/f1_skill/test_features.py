import pandas as pd

from f1_skill.features import STAGE_A_COLUMNS, q1_pace


def session(*driver_times, season=2023, gp="Miami Grand Prix", wet=False):
    rows = []
    for driver, times in driver_times:
        for lap_time in times:
            rows.append(
                {
                    "season": season,
                    "gp": gp,
                    "team": f"team-{driver}",
                    "drv": driver,
                    "time": lap_time,
                    "iacc": True,
                    "del": False,
                    "qs": "Q1",
                    "wR": wet,
                    "sesT": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_the_session_fastest_driver_sits_at_zero_by_construction():
    table = q1_pace(session(("VER", [90.0, 89.0]), ("HAM", [91.0])))
    assert table.loc[table["drv"] == "VER", "gap_to_fastest"].item() == 0.0


def test_the_gap_is_a_percentage_of_the_fastest_lap_not_a_time_difference():
    table = q1_pace(session(("VER", [100.0]), ("HAM", [101.0])))
    assert table.loc[table["drv"] == "HAM", "gap_to_fastest"].item() == 1.0


def test_only_the_best_valid_lap_counts_and_the_rest_are_counted():
    table = q1_pace(session(("VER", [90.0, 89.0, 95.0])))
    row = table.iloc[0]
    assert row["best_lap"] == 89.0
    assert row["n_valid_laps"] == 3


def test_the_107_percent_flag_follows_the_sport_s_rule():
    table = q1_pace(session(("VER", [100.0]), ("INSIDE", [106.9]), ("OUTSIDE", [107.5])))
    flags = dict(zip(table["drv"], table["within_107"]))
    assert flags["INSIDE"]
    assert not flags["OUTSIDE"]


def test_a_lap_at_exactly_107_percent_falls_outside_which_is_floating_point():
    """The rule would include it; the arithmetic does not.

    (107.0 - 100.0) / 100.0 * 100 is 7.000000000000001, so `<= 7` rejects it. No driver-session in
    the dataset lands there (the nearest is 6.906%), so nothing the project reports depends on this.
    Recorded rather than patched: adding a tolerance would be inventing a threshold the notebook
    never had.
    """
    table = q1_pace(session(("VER", [100.0]), ("BOUNDARY", [107.0])))
    flags = dict(zip(table["drv"], table["within_107"]))
    assert not flags["BOUNDARY"]


def test_the_table_carries_the_columns_the_models_expect():
    table = q1_pace(session(("VER", [90.0])))
    assert list(table.columns) == STAGE_A_COLUMNS


def test_gaps_are_computed_within_a_session_not_across_the_dataset():
    slow = session(("BOT", [100.0]), ("ZHO", [101.0]), gp="Monaco Grand Prix")
    quick = session(("VER", [80.0]), ("HAM", [80.8]), gp="Miami Grand Prix")
    table = q1_pace(pd.concat([slow, quick], ignore_index=True))
    gaps = dict(zip(table["drv"], table["gap_to_fastest"]))
    assert gaps["BOT"] == 0.0
    assert gaps["VER"] == 0.0
    assert round(gaps["HAM"], 4) == 1.0
