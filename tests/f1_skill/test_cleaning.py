import pandas as pd

from f1_skill.cleaning import is_valid, numeric_times, rain_fraction, valid_segment_laps


def laps(**overrides):
    base = {"iacc": True, "del": False, "qs": "Q1", "wR": False, "time": 90.0}
    return pd.DataFrame([{**base, **row} for row in overrides["rows"]])


def test_both_halves_of_the_validity_rule_are_load_bearing():
    frame = laps(
        rows=[
            {"time": 88.0},
            {"time": 80.0, "del": True},  # a deleted lap is quicker than the real best
            {"time": 79.0, "iacc": False},  # so is an inaccurate one
        ]
    )
    assert is_valid(frame).tolist() == [True, False, False]
    assert frame[is_valid(frame)]["time"].min() == 88.0


def test_segment_filter_keeps_only_the_named_segment():
    frame = laps(rows=[{"qs": "Q1"}, {"qs": "Q2"}, {"qs": "Q3"}, {"qs": None}])
    assert len(valid_segment_laps(frame, "Q1")) == 1
    assert len(valid_segment_laps(frame, "Q3")) == 1


def test_rain_fraction_is_the_share_of_wet_laps_in_a_session():
    frame = laps(rows=[{"wR": True}, {"wR": True}, {"wR": False}, {"wR": False}])
    frame[["season", "gp"]] = [2023, "Miami Grand Prix"]
    assert rain_fraction(frame).loc[2023, "Miami Grand Prix"] == 0.5


def test_a_dry_session_scores_zero_rather_than_missing():
    frame = laps(rows=[{"wR": False}, {"wR": False}])
    frame[["season", "gp"]] = [2023, "Miami Grand Prix"]
    assert rain_fraction(frame).loc[2023, "Miami Grand Prix"] == 0.0


def test_a_lap_with_no_time_arrives_as_the_string_None_and_becomes_a_null():
    """Object dtype here silently poisons every derived column downstream."""
    frame = laps(rows=[{"time": 90.0}, {"time": "None"}])
    converted = numeric_times(frame)
    assert converted["time"].dtype.kind == "f"
    assert converted["time"].isna().sum() == 1


def test_valid_laps_come_back_with_a_numeric_time_column():
    frame = laps(rows=[{"time": "None"}, {"time": 88.0}])
    assert valid_segment_laps(frame, "Q1")["time"].dtype.kind == "f"
