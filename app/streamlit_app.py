"""Dashboard over the committed results. Reads artifacts, never fits anything.

Every number on the page comes out of `models/`, so the app cannot disagree with the README or the
notebooks: if a figure here looks wrong, the artifact is wrong and `make train` regenerates it.
Nothing is hardcoded except the labels.

The charts are drawn here rather than loaded from `reports/figures/`. The committed PNGs are report
figures at a fixed size; on a page where the tables are sortable and filterable, a static image
beside them cannot be interrogated, which is the whole reason to put this in a browser.

    uv run streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

COMPONENTS = ["car", "driver", "season", "unexplained"]
MOVER, STAYER, NEUTRAL = "#2980b9", "#e67e22", "#7f8c8d"


@st.cache_data
def _read(path, modified):
    """`modified` is unused by the body and load-bearing in the signature: it is what makes the
    cache key change when an artifact is regenerated."""
    del modified
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    return json.loads(path.read_text())


def load(name):
    path = MODELS / name
    return _read(path, path.stat().st_mtime) if path.exists() else None


ROW_HEIGHT = 21
MOVED, STAYED = "changed team", "one team only"


def effect_chart(frame, label_column, what, colour=None):
    """Estimate and 89% interval per level, quickest at the top.

    The interval is a rule rather than an error bar on a column, because the estimate is not a
    magnitude measured from zero: zero means an average driver, and a bar would imply otherwise.
    """
    order = frame.sort_values("seconds")[label_column].tolist()
    # labelOverlap defaults to thinning an axis that does not fit, which silently drops every
    # other name and leaves points nothing identifies them by.
    axis = alt.Y(f"{label_column}:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False))
    tooltip = [
        alt.Tooltip(f"{label_column}:N", title=label_column.replace("_", " ")),
        alt.Tooltip("seconds:Q", title="seconds vs average", format=".3f"),
        alt.Tooltip("lower_89:Q", title="lower 89%", format=".3f"),
        alt.Tooltip("upper_89:Q", title="upper 89%", format=".3f"),
        alt.Tooltip("sessions:Q", title="sessions"),
    ]

    base = alt.Chart(frame)
    rules = base.mark_rule(color="#bdc3c7", strokeWidth=2).encode(
        y=axis, x=alt.X("lower_89:Q", title=f"seconds vs an average {what}"), x2="upper_89:Q"
    )
    marks = base.mark_circle(size=70).encode(y=axis, x="seconds:Q", tooltip=tooltip)
    if colour:
        marks = marks.encode(
            color=alt.Color(
                f"{colour}:N",
                title=None,
                # Domain pinned, because sorting the values alphabetically would hand "changed
                # team" the stayer colour and invert the project's own convention.
                scale=alt.Scale(domain=[MOVED, STAYED], range=[MOVER, STAYER]),
                legend=alt.Legend(orient="top"),
            )
        )
    else:
        marks = marks.encode(color=alt.value(MOVER))

    zero = base.mark_rule(color="black", strokeDash=[4, 4]).encode(x=alt.datum(0))
    return (rules + marks + zero).properties(height=max(260, ROW_HEIGHT * len(frame)))


def ratings_table(frame, label):
    shown = frame.sort_values("seconds").copy()
    shown["89% interval"] = shown.apply(
        lambda row: f"[{row['lower_89']:.3f}, {row['upper_89']:.3f}]", axis=1
    )
    heading = f"seconds vs an average {label}"
    columns = ["seconds", "89% interval"] + [
        c for c in shown.columns if c in ("sessions", "teams", "drivers")
    ]
    st.dataframe(
        shown[columns].rename(columns={"seconds": heading}),
        width="stretch",
        height=420,
        # Without an explicit format a trailing zero is dropped, so -0.540 prints as -0.54 next to
        # -0.321 and the decimal points stop lining up in a column built for comparing magnitudes.
        column_config={heading: st.column_config.NumberColumn(format="%.3f")},
    )


results = load("decomposition_results.json")
if results is None:
    st.error("No results found. Run `make train` first to generate `models/`.")
    st.stop()

st.set_page_config(page_title="F1 driver vs car", layout="wide")
st.title("How much of a Formula 1 result is the driver, and how much is the car?")
st.caption(
    "Bayesian hierarchical decomposition of qualifying pace, 2018 to 2025. "
    "Every number here is read from the committed results; nothing is refitted."
)

ratio = results["car_to_driver_ratio"]
low, high = results["car_to_driver_89"]
shares = results["shares_of_all_variation"]
lap = results["reference_lap_seconds"]

left, middle, right = st.columns(3)
left.metric("car : driver variance", f"{ratio:.2f}x", help=f"89% interval [{low}, {high}]")
middle.metric("car share of all variation", f"{shares['median']['car']}%")
right.metric("unexplained", f"{shares['median']['unexplained']}%")

st.info(
    f"The car accounts for about {ratio:.1f} times as much variation in qualifying pace as the "
    f"driver, 89% interval [{low}, {high}]. The residual is the largest single component, so "
    f'"the car is {results["shares_of_explained_variation"]["median"]["car"]}%" means that share '
    "of what the model explains, not that share of qualifying."
)

with st.expander("Before reading the numbers"):
    st.markdown(
        f"""
- **The model explains under half of qualifying.** The residual is
  {shares["median"]["unexplained"]}% and is the largest single component. Every share below is a
  share of *all* variation unless it says otherwise.
- **Q1 measures the car conservatively.** Strong teams have pace in hand in Q1 and do not always
  use it, which pushes the measured car share down rather than up.
- **The split is not clean.** Driver and car effects correlate at
  {results["driver_car_correlation"]}: good drivers are hired by good teams, and an additive model
  assigns that overlap to neither.
- **Read the intervals, not the order.** Two drivers whose 89% intervals overlap are not ranked by
  this data, however far apart their point estimates look.

The full writeup, including everything that was tried and failed, is the repository's README.
"""
    )

drivers, cars, variation, holds_up = st.tabs(
    ["Drivers", "Cars", "Where the variation comes from", "Does it hold up?"]
)

with drivers:
    frame = load("driver_ratings.csv")
    if frame is None:
        st.warning("`models/driver_ratings.csv` is missing. Run `make train`.")
    else:
        st.markdown(
            f"**How much quicker or slower each driver is than an average one in the same car**, "
            f"in seconds on this dataset's median qualifying lap of {lap}s. Negative is quicker. "
            f"Estimated across {results['driver_sessions']:,} driver-sessions."
        )
        frame = frame.reset_index()
        picked = st.multiselect(
            "Compare specific drivers",
            sorted(frame["drv"]),
            default=[],
            help="Leave empty for the whole field. Overlapping intervals mean the order between "
            "two drivers is not settled by this data.",
        )
        fewest = int(frame["sessions"].min())
        most = int(frame["sessions"].max())
        floor = st.slider(
            "Minimum qualifying sessions",
            fewest,
            most,
            fewest,
            disabled=bool(picked),
            help="Drag it up to drop the thin records and watch the wide intervals disappear.",
        )
        kept = frame[frame["drv"].isin(picked)] if picked else frame[frame["sessions"] >= floor]
        kept = kept.copy()
        kept["record"] = kept["teams"].gt(1).map({True: MOVED, False: STAYED})

        st.altair_chart(effect_chart(kept, "drv", "driver", colour="record"), width="stretch")
        st.caption(
            f"{len(kept)} of {len(frame)} drivers shown. Interval width is the honest part: a "
            "driver with one session and a driver with 154 are not equally well known, and partial "
            "pooling shrinks thin records toward the middle rather than letting one good lap "
            "create a rating. Colour marks who changed team, because those moves are what make "
            "separating driver from car possible at all."
        )
        ratings_table(kept.set_index("drv").drop(columns="record"), "driver")

with cars:
    frame = load("car_ratings.csv")
    if frame is None:
        st.warning("`models/car_ratings.csv` is missing. Run `make train`.")
    else:
        st.markdown(
            f"**One row per team per season**, because the 2023 Red Bull and the 2024 Red Bull are "
            f"different cars. Seconds against an average car on a {lap}s lap."
        )
        frame = frame.reset_index()
        seasons = sorted(frame["season"].unique())
        chosen_seasons = st.multiselect("Seasons", seasons, default=seasons)
        chosen_teams = st.multiselect("Teams", sorted(frame["team"].unique()), default=[])

        kept = frame[frame["season"].isin(chosen_seasons)]
        if chosen_teams:
            kept = kept[kept["team"].isin(chosen_teams)]

        if kept.empty:
            st.warning("Nothing selected.")
        else:
            st.altair_chart(effect_chart(kept, "team_season", "car"), width="stretch")
            st.caption(f"{len(kept)} of {len(frame)} team-seasons shown.")
            ratings_table(kept.set_index("team_season"), "car")

with variation:
    split = pd.DataFrame(
        [
            {
                "component": name,
                "share": shares["median"][name],
                "lower_89": shares["lower_89"][name],
                "upper_89": shares["upper_89"][name],
            }
            for name in COMPONENTS
        ]
    )
    bars = (
        alt.Chart(split)
        .mark_bar()
        .encode(
            y=alt.Y("component:N", sort=COMPONENTS, title=None),
            x=alt.X("share:Q", title="share of all variation in Q1 pace (%)"),
            color=alt.Color(
                "component:N",
                sort=COMPONENTS,
                scale=alt.Scale(domain=COMPONENTS, range=[MOVER, STAYER, NEUTRAL, "#bdc3c7"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("component:N"),
                alt.Tooltip("share:Q", title="share %", format=".1f"),
                alt.Tooltip("lower_89:Q", title="lower 89%", format=".1f"),
                alt.Tooltip("upper_89:Q", title="upper 89%", format=".1f"),
            ],
        )
    )
    intervals = (
        alt.Chart(split)
        .mark_rule(color="black", strokeWidth=2)
        .encode(y=alt.Y("component:N", sort=COMPONENTS), x="lower_89:Q", x2="upper_89:Q")
    )
    st.altair_chart((bars + intervals).properties(height=220), width="stretch")
    st.caption(
        "The residual is kept in view on purpose: dropping it turns "
        f'"{results["shares_of_explained_variation"]["median"]["car"]}% of what the model '
        'explains" into a much stronger claim about qualifying itself.'
    )

    st.subheader("The split is not quite clean")
    st.write(
        f"Driver and car effects correlate at {results['driver_car_correlation']} across the "
        "actual assignment: good drivers are hired by good teams, and an additive model assigns "
        "that overlap to neither. Weighting by the sessions actually driven rather than counting "
        f"each driver once gives {results['observation_weighted_ratio']}x against "
        f"{ratio:.2f}x from the hyperparameters, so the ratio survives the assumption even though "
        "the split does not fully resolve."
    )

with holds_up:
    st.subheader("A second method, built differently, puts drivers in a similar order")
    agreement = pd.DataFrame(
        [{"method": name, "rho": value} for name, value in load("model_agreement.json").items()]
    )
    st.altair_chart(
        alt.Chart(agreement)
        .mark_bar(color=MOVER)
        .encode(
            y=alt.Y("method:N", sort="-x", title=None),
            # The full scale, because a rho of 0.75 on an axis starting at 0.7 looks like
            # agreement it has not earned.
            x=alt.X("rho:Q", scale=alt.Scale(domain=[0, 1]), title="Spearman rho"),
            tooltip=[alt.Tooltip("method:N"), alt.Tooltip("rho:Q", format=".3f")],
        )
        .properties(height=160),
        width="stretch",
    )
    st.caption(
        "Teammate duels within a session, ranked three ways, none of which share machinery with "
        "the hierarchical model."
    )

    st.subheader("Sampler diagnostics")
    health = results["convergence"]
    a, b, c = st.columns(3)
    a.metric("divergences", health["divergences"])
    b.metric("worst R-hat", f"{health['worst_rhat']:.4f}", help=health["worst_rhat_parameter"])
    c.metric("lowest ESS", f"{health['lowest_ess']:,.0f}")
    st.caption(
        "Zero divergences and an R-hat at 1.00 mean the chains agree and the sampler explored the "
        "posterior cleanly. Effective sample size is how many independent draws the correlated "
        "ones are worth."
        if health["converged"]
        else "R-hat is above the threshold: the numbers on this page are not trustworthy."
    )

    variants = load("sensitivity_results.json")
    if variants:
        st.subheader("Refitting on each qualifying segment")
        table = pd.DataFrame(variants)
        st.dataframe(
            table,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns},
        )
        st.caption(
            "Q2 and Q3 hold fewer cars, drawn from the fastest part of the field, so these are "
            "not competing estimates of one quantity. The gap between cars widening when everyone "
            "is trying was a written-down prediction, and it passed."
        )

    st.subheader("The baseline loses to the starting grid")
    st.dataframe(pd.DataFrame(load("baseline_metrics.json")), width="stretch")
    st.caption(
        "On the 2024 and 2025 seasons, held out of every decision and scored once. Ranking the "
        "field by starting slot beats both fitted models, because the grid slot already contains "
        "the driver and the car fused into one number. That is why this project needs a model "
        "built for attribution rather than prediction."
    )
