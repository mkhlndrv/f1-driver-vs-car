"""Figures are checked for the things a reader would be misled by, not for pixels."""

import pandas as pd

from f1_skill import plots


def ratings():
    return pd.DataFrame(
        {
            "seconds": [-0.54, 0.0, 0.51],
            "lower_89": [-0.65, -0.10, 0.20],
            "upper_89": [-0.44, 0.10, 0.87],
            "teams": [1, 2, 1],
        },
        index=["VER", "OCO", "MAZ"],
    )


def test_the_driver_chart_puts_the_fastest_at_the_top():
    axes = plots.driver_ratings(ratings(), 83.9).axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["VER", "OCO", "MAZ"]


def test_the_driver_chart_says_which_lap_length_the_seconds_are_on():
    """A gap in seconds is meaningless without the lap it was converted at."""
    axes = plots.driver_ratings(ratings(), 83.9).axes[0]
    assert "84s" in axes.get_xlabel()


def test_the_driver_chart_marks_movers_and_stayers_differently():
    """Which drivers changed team is what makes the separation possible, so it is on the chart."""
    figure = plots.driver_ratings(ratings(), 83.9)
    colours = {tuple(c) for c in figure.axes[0].collections[-1].get_facecolors()}
    assert len(colours) == 2


def test_the_variance_chart_keeps_the_residual_in_view():
    """Dropping it turns "67% of what the model explains" into a much stronger claim."""
    shares = pd.DataFrame(
        {
            "lower_89": [0.24, 0.07, 0.00, 0.45],
            "median": [0.31, 0.11, 0.03, 0.53],
            "upper_89": [0.40, 0.17, 0.14, 0.59],
        },
        index=["car", "driver", "season", "unexplained"],
    )
    axes = plots.variance_split(shares).axes[0]
    assert "unexplained" in [label.get_text() for label in axes.get_yticklabels()]


def test_the_agreement_chart_is_drawn_on_the_full_correlation_scale():
    """A rho of 0.75 on an axis starting at 0.7 looks like agreement it has not earned."""
    axes = plots.ranking_agreement({"Elo": 0.752, "Bradley-Terry": 0.801}).axes[0]
    assert axes.get_xlim() == (0.0, 1.0)
