"""Figures for the committed results.

Agg backend, set before pyplot is imported, so these render in CI and in Docker where there is no
display. Every function takes what it needs and returns a Figure; nothing here reads or writes
files, so the caller decides where a figure goes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MOVER_COLOUR = "#2980b9"
STAYER_COLOUR = "#e67e22"
INTERVAL_COLOUR = "#bdc3c7"


def driver_ratings(ratings, lap_seconds):
    """Every driver against an average one in the same car, with 89% intervals.

    Ordered fastest first. The interval width is the honest part of this chart: a driver with 17
    sessions has a range several times wider than one with 150, and the two should not be read as
    equally well known.
    """
    ranked = ratings.sort_values("seconds")
    moved = ranked["teams"] > 1
    positions = range(len(ranked))

    figure, axes = plt.subplots(figsize=(8, 11))
    axes.hlines(positions, ranked["lower_89"], ranked["upper_89"], color=INTERVAL_COLOUR, lw=2)
    axes.scatter(
        ranked["seconds"],
        positions,
        c=[MOVER_COLOUR if m else STAYER_COLOUR for m in moved],
        s=34,
        zorder=3,
    )
    axes.axvline(0, color="k", lw=1, ls="--")
    axes.set_yticks(list(positions))
    axes.set_yticklabels(ranked.index, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel(f"seconds against an average driver, same car ({lap_seconds:.0f}s lap)")
    axes.set_title(
        "Car-adjusted driver ratings, 89% intervals\n"
        "blue = drove under more than one team name, orange = one name only"
    )
    figure.tight_layout()
    return figure


def variance_split(shares):
    """Where qualifying variation comes from, with the residual kept in view.

    The residual is the largest single piece, and leaving it out of the picture is what turns
    "the car is 67% of what the model explains" into the much stronger claim that it is 67% of
    qualifying.
    """
    order = ["car", "driver", "season", "unexplained"]
    medians = [shares.loc[name, "median"] * 100 for name in order]
    lower = [shares.loc[name, "lower_89"] * 100 for name in order]
    upper = [shares.loc[name, "upper_89"] * 100 for name in order]

    figure, axes = plt.subplots(figsize=(8, 3.5))
    positions = range(len(order))
    axes.barh(positions, medians, color=[MOVER_COLOUR, STAYER_COLOUR, "#7f8c8d", INTERVAL_COLOUR])
    for position, (low, high) in enumerate(zip(lower, upper)):
        axes.hlines(position, low, high, color="k", lw=1.5)
    axes.set_yticks(list(positions))
    axes.set_yticklabels(order)
    axes.invert_yaxis()
    axes.set_xlabel("share of all variation in Q1 pace (%)")
    axes.set_title("What moves a qualifying lap, with 89% intervals")
    figure.tight_layout()
    return figure


def ranking_agreement(agreements):
    """The robustness check: three teammate-based rankings against the hierarchical one."""
    names = list(agreements)
    values = [agreements[name] for name in names]

    figure, axes = plt.subplots(figsize=(7, 3))
    axes.barh(range(len(names)), values, color=MOVER_COLOUR)
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names)
    axes.invert_yaxis()
    axes.set_xlim(0, 1)
    axes.set_xlabel("Spearman rank correlation with the hierarchical ranking")
    axes.set_title("Does a second method put the drivers in the same order?")
    figure.tight_layout()
    return figure
