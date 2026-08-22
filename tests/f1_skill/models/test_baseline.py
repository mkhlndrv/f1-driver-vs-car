"""The leakage test, required by CLAUDE1.md section 10 as a test rather than a note.

A running total that fails to exclude its own race leaks the result it is meant to predict, and the
scores come out excellent and meaningless. Rows earlier than the tampered race cannot change no
matter what, because a cumulative sum and a shifted rolling mean have no way to propagate
backwards, so a test that only looks earlier cannot fail. What discriminates is whether the
tampered race's OWN features move.
"""

import pandas as pd

from f1_skill.models import baseline
from f1_skill.models.baseline import build_features

ENGINEERED = ["points_before", "recent_form", "recent_form_finishers", "circuit_history"]


def season(*finishes, code="VER", year=2023):
    """One driver, one race per round, finishing where told."""
    return pd.DataFrame(
        [
            {
                "year": year,
                "round": index + 1,
                "code": code,
                "name": "Miami Grand Prix",
                "points": points,
                "sprint": 0.0,
                "positionOrder": position,
                "positionText": str(position),
            }
            for index, (position, points) in enumerate(finishes)
        ]
    )


def test_a_races_own_features_ignore_its_own_result():
    honest = season((1, 25.0), (2, 18.0), (3, 15.0))
    tampered = honest.copy()
    last = tampered["round"] == 3
    tampered.loc[last, ["points", "positionOrder", "positionText"]] = [99.0, 1, "1"]

    before = build_features(honest)
    after = build_features(tampered)

    changed = before.loc[before["round"] == 3, ENGINEERED].compare(
        after.loc[after["round"] == 3, ENGINEERED]
    )
    assert changed.empty, f"the tampered race's own features moved: {changed.columns.tolist()}"


def test_earlier_races_are_untouched_by_a_later_result():
    honest = season((1, 25.0), (2, 18.0), (3, 15.0))
    tampered = honest.copy()
    tampered.loc[tampered["round"] == 3, ["points", "positionOrder", "positionText"]] = [
        99.0,
        1,
        "1",
    ]

    before = build_features(honest)
    after = build_features(tampered)
    earlier = before["round"] < 3
    assert before.loc[earlier, ENGINEERED].compare(after.loc[earlier, ENGINEERED]).empty


def test_points_before_accumulates_earlier_races_only():
    frame = build_features(season((1, 25.0), (2, 18.0), (3, 15.0)))
    assert frame["points_before"].tolist() == [0.0, 25.0, 43.0]


def test_points_before_counts_sprint_points():
    frame = season((1, 25.0), (2, 18.0))
    frame.loc[frame["round"] == 1, "sprint"] = 8.0
    assert build_features(frame)["points_before"].tolist() == [0.0, 33.0]


def test_recent_form_excludes_the_current_race():
    frame = build_features(season((10, 1.0), (2, 18.0), (4, 12.0)))
    assert pd.isna(frame["recent_form"].iloc[0])
    assert frame["recent_form"].iloc[1] == 10.0
    assert frame["recent_form"].iloc[2] == 6.0


def test_a_retirement_drops_out_of_the_finishers_average_rather_than_counting_as_last():
    frame = season((2, 18.0), (20, 0.0), (4, 12.0))
    frame.loc[frame["round"] == 2, "positionText"] = "R"
    built = build_features(frame)
    assert built["recent_form"].iloc[2] == 11.0
    assert built["recent_form_finishers"].iloc[2] == 2.0


def archive_tables():
    """The five tables the race archive supplies, cut down to two races."""
    return {
        "races": pd.DataFrame(
            {
                "raceId": [1, 2],
                "year": [2023, 2023],
                "round": [1, 2],
                "name": ["Miami Grand Prix", "Monaco Grand Prix"],
                "date": ["2023-05-07", "2023-05-28"],
            }
        ),
        "results": pd.DataFrame(
            {
                "raceId": [1, 1, 2, 2],
                "driverId": [1, 2, 1, 2],
                "constructorId": [1, 2, 1, 2],
                "grid": ["1", "0", "2", r"\N"],
                "positionText": ["1", "5", "2", "R"],
                "positionOrder": [1, 5, 2, 20],
                "points": [25.0, 10.0, 18.0, 0.0],
            }
        ),
        "sprint_results": pd.DataFrame({"raceId": [1], "driverId": [1], "points": [8.0]}),
        "drivers": pd.DataFrame({"driverId": [1, 2], "code": ["VER", "HAM"]}),
        "constructors": pd.DataFrame({"constructorId": [1, 2], "name": ["Red Bull", "Mercedes"]}),
    }


def test_a_pit_lane_start_is_encoded_behind_the_last_real_grid_slot():
    """Recorded as 0, which would otherwise read as the best slot on the grid."""
    results = baseline.race_results(archive_tables())
    pit_lane = results[results["grid"] == "0"]
    assert pit_lane["grid_position"].item() > results["grid_position"].max(skipna=True) - 1


def test_a_missing_grid_slot_stays_null_and_is_dropped_only_after_features_are_built():
    tables = archive_tables()
    results = baseline.race_results(tables)
    assert results["grid_position"].isna().sum() == 1
    # the row still contributed its points to the running totals before being dropped
    assert len(baseline.model_frame(results)) == len(results) - 1


def test_the_podium_target_counts_the_first_three_classified_places():
    results = baseline.race_results(archive_tables())
    assert results["podium"].tolist() == [True, False, True, False]


def test_grid_only_scores_without_fitting_anything():
    """The reference the models have to beat, and do not."""
    frame = pd.DataFrame(
        {"podium": [True, True, False, False], "grid_position": [1.0, 2.0, 15.0, 20.0]}
    )
    assert baseline.grid_only_pr_auc(frame) == 1.0


def field(year, drivers=6):
    """One race, `drivers` cars, podium going to the front three of the grid."""
    return pd.DataFrame(
        {
            "year": year,
            "round": 1,
            "grid_position": [float(slot) for slot in range(1, drivers + 1)],
            "code": [f"D{slot}" for slot in range(1, drivers + 1)],
            "constructor": ["Red Bull", "Mercedes"] * (drivers // 2),
            "circuit": "Miami Grand Prix",
            "points_before": 0.0,
            "recent_form": 5.0,
            "podium": [slot <= 3 for slot in range(1, drivers + 1)],
        }
    )


def seasons(*years):
    """What `model_frame` hands `evaluate`: several seasons, categoricals as `category` dtype.
    LightGBM refuses an object column outright, so the dtype is part of the contract."""
    frame = pd.concat([field(year) for year in years], ignore_index=True)
    return frame.astype({column: "category" for column in baseline.CATEGORICAL})


def test_score_reports_the_four_metrics_under_the_fold_it_was_given():
    frame = field(2024)
    model = baseline.logistic_pipeline(["grid_position", "round", "points_before", "recent_form"])
    model.fit(frame[baseline.FEATURES], frame["podium"])
    row = baseline.score(model, frame, baseline.FEATURES, "in sample")
    assert row["fold"] == "in sample"
    assert sorted(row) == ["fold", "log_loss", "pr_auc", "precision", "recall"]
    assert 0.0 <= row["pr_auc"] <= 1.0


def test_evaluate_scores_the_test_seasons_and_the_grid_beside_them():
    metrics = baseline.evaluate(seasons(2022, 2023, 2024, 2025))
    assert metrics.index.tolist() == [
        "LightGBM",
        "logistic regression",
        "grid slot only, no model",
    ]
    # the grid row is not a fit, so it has no likelihood to report
    assert pd.isna(metrics.loc["grid slot only, no model", "log_loss"])


def test_evaluate_never_trains_on_a_test_season(monkeypatch):
    """The whole design of the baseline rests on the test seasons being touched once, at scoring.
    This checks the rows themselves rather than a metric, so it cannot pass by luck."""
    frame = seasons(2022, 2023, 2024, 2025)
    trained_on = []

    class Recording(baseline.LGBMClassifier):
        def fit(self, X, y, **kwargs):
            trained_on.append(X.index)
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(baseline, "LGBMClassifier", Recording)
    baseline.evaluate(frame)

    years = frame.loc[trained_on[0], "year"]
    assert not years.isin(baseline.TEST_SEASONS).any()
    assert sorted(years.unique()) == [2022, 2023]


def test_the_front_three_rule_needs_no_fitting_and_is_scored_on_the_same_rows():
    metrics = baseline.evaluate(seasons(2022, 2023, 2024, 2025))
    # the grid is perfectly ordered here, so the two-line rule is exactly right
    assert metrics.loc["grid slot only, no model", "precision"] == 1.0
    assert metrics.loc["grid slot only, no model", "recall"] == 1.0


def test_the_logistic_pipeline_handles_a_category_it_never_saw():
    """A driver who debuts in a test season must not crash the fit."""
    pipeline = baseline.logistic_pipeline(["grid_position"])
    train = pd.DataFrame(
        {
            "grid_position": [1.0, 20.0],
            "code": ["VER", "HAM"],
            "constructor": ["Red Bull", "Mercedes"],
            "circuit": ["Miami", "Monaco"],
        }
    )
    pipeline.fit(train, [True, False])
    unseen = train.assign(code=["NEW", "NEW"])
    assert len(pipeline.predict_proba(unseen)) == 2
