"""Supporting model: predicting a podium from pre-race information.

This exists as a contrast piece, not as the project's answer. Its finding is negative and that is
the useful part: on seasons kept out of every decision, a conventional pipeline does not beat
ranking the field by starting slot, because the grid slot already contains the driver and the car
fused into one number. A model can predict a result accurately and still be unable to say how much
of that result was the driver, which is what the hierarchical model is for.

Every choice here was made on 2021 to 2023 and the 2024-25 seasons are scored once.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# LightGBM is multithreaded, so an identical seed still drifts across machines unless the thread
# count is pinned too. With these three, every fit reproduces exactly.
BOOSTING = {
    "random_state": 0,
    "verbose": -1,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
}

FEATURES = [
    "grid_position",
    "round",
    "code",
    "constructor",
    "circuit",
    "points_before",
    "recent_form",
]
CATEGORICAL = ["code", "constructor", "circuit"]
ENGINEERED = ["points_before", "recent_form", "recent_form_finishers", "circuit_history"]
VALIDATION_SEASONS = [2021, 2022, 2023]
TEST_SEASONS = [2024, 2025]
SEASONS = range(2018, 2026)
PODIUM_PLACES = ["1", "2", "3"]
FORM_RACES = 3


def race_results(tables):
    races = tables["races"][tables["races"]["year"].between(min(SEASONS), max(SEASONS))]
    results = (
        tables["results"]
        .merge(races[["raceId", "year", "round", "name", "date"]], on="raceId")
        .merge(tables["drivers"][["driverId", "code"]], on="driverId")
        .merge(
            tables["constructors"][["constructorId", "name"]].rename(
                columns={"name": "constructor"}
            ),
            on="constructorId",
        )
    )
    sprint_points = (
        tables["sprint_results"].groupby(["raceId", "driverId"])["points"].sum().rename("sprint")
    )
    results = results.merge(sprint_points, on=["raceId", "driverId"], how="left")
    results["sprint"] = results["sprint"].fillna(0)

    grid = pd.to_numeric(results["grid"], errors="coerce")
    # A pit-lane start is recorded as 0, which would read as the best slot on the grid. Encode it
    # as one place behind the last real one instead.
    last_slot = grid[grid > 0].max()
    results["grid_position"] = grid.where(grid > 0, last_slot + 1)
    results.loc[results["grid"].astype(str) == r"\N", "grid_position"] = np.nan

    results["podium"] = results["positionText"].isin(PODIUM_PLACES)
    return results


def build_features(results):
    """Pre-race features only. Every cumulative column excludes the race it sits on."""
    out = results.sort_values(["year", "round"]).copy()
    out["circuit"] = out["name"]

    out["season_points"] = out["points"] + out["sprint"]
    out["points_before"] = (
        out.groupby(["year", "code"])["season_points"].cumsum() - out["season_points"]
    )

    out["recent_form"] = out.groupby("code")["positionOrder"].transform(
        lambda finishes: finishes.shift(1).rolling(FORM_RACES, min_periods=1).mean()
    )

    # positionText is a number only when a driver was classified, so retirements become NaN and
    # drop out of the average rather than counting as a last place.
    classified = pd.to_numeric(out["positionText"], errors="coerce")
    out["recent_form_finishers"] = classified.groupby(out["code"]).transform(
        lambda finishes: finishes.shift(1).rolling(FORM_RACES, min_periods=1).mean()
    )

    out["circuit_history"] = out.groupby(["code", "circuit"])["positionOrder"].transform(
        lambda finishes: finishes.shift(1).expanding().mean()
    )
    return out


def model_frame(results):
    """Features, then drop rows with no grid slot: they still count towards running totals."""
    frame = build_features(results)
    frame = frame[frame["grid_position"].notna()].copy()
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("category")
    return frame


def logistic_pipeline(numeric_columns):
    return Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric_columns,
                        ),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=2000, random_state=0)),
        ]
    )


def score(model, frame, columns, label):
    probability = model.predict_proba(frame[columns])[:, 1]
    prediction = model.predict(frame[columns])
    return {
        "fold": label,
        "pr_auc": average_precision_score(frame["podium"], probability),
        "log_loss": log_loss(frame["podium"], probability),
        "precision": precision_score(frame["podium"], prediction, zero_division=0),
        "recall": recall_score(frame["podium"], prediction, zero_division=0),
    }


def grid_only_pr_auc(frame):
    """No model at all: rank by starting slot, nearer the front means likelier podium."""
    return average_precision_score(frame["podium"], -frame["grid_position"])


def evaluate(frame, columns=FEATURES):
    """Fit on everything up to 2023, score the untouched seasons once, against the grid."""
    train = frame[frame["year"] <= max(VALIDATION_SEASONS)]
    test = frame[frame["year"].isin(TEST_SEASONS)]

    boosting = LGBMClassifier(**BOOSTING).fit(train[columns], train["podium"])
    linear = logistic_pipeline([c for c in columns if c not in CATEGORICAL]).fit(
        train[columns], train["podium"]
    )
    front_three = test["grid_position"] <= len(PODIUM_PLACES)

    return pd.DataFrame(
        [
            score(boosting, test, columns, "LightGBM"),
            score(linear, test, columns, "logistic regression"),
            {
                "fold": "grid slot only, no model",
                "pr_auc": grid_only_pr_auc(test),
                "log_loss": float("nan"),
                "precision": precision_score(test["podium"], front_three, zero_division=0),
                "recall": recall_score(test["podium"], front_three, zero_division=0),
            },
        ]
    ).set_index("fold")
