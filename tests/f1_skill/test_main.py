"""The stage cache.

The pipeline itself is verified by running it, not from here: `python -m f1_skill` reproduces the
notebooks' figures and that is the check that means something. What a test can add is the caching
rule, because getting it wrong is silent. A stage that rebuilds when it should reuse costs 173
downloads; one that reuses when it should rebuild reports a stale number as a fresh one.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f1_skill.__main__ import _cached


def test_a_missing_stage_is_built_and_written(tmp_path):
    path = tmp_path / "stage.csv"
    frame = _cached(path, lambda: pd.DataFrame({"gap": [0.0, 1.5]}))
    assert path.exists()
    assert frame["gap"].tolist() == [0.0, 1.5]


def test_an_existing_stage_is_read_back_rather_than_rebuilt(tmp_path):
    path = tmp_path / "stage.csv"
    pd.DataFrame({"gap": [0.0, 1.5]}).to_csv(path, index=False)

    def must_not_run():
        raise AssertionError("the cached stage was rebuilt")

    assert _cached(path, must_not_run)["gap"].tolist() == [0.0, 1.5]


def test_the_stage_directory_is_created_if_it_is_not_there(tmp_path):
    """A fresh clone has no data/processed/, so the first run has to make it."""
    path = tmp_path / "processed" / "stage.csv"
    _cached(path, lambda: pd.DataFrame({"gap": [0.0]}))
    assert path.exists()


def test_the_index_is_not_written_as_a_column(tmp_path):
    """Left in, it comes back as `Unnamed: 0` and every downstream column position shifts."""
    path = tmp_path / "stage.csv"
    _cached(path, lambda: pd.DataFrame({"gap": [0.0, 1.5]}))
    assert pd.read_csv(path).columns.tolist() == ["gap"]


def test_a_posterior_survives_a_trip_to_disk_and_back(tmp_path):
    """The fit is cached as netCDF, which needs a backend nothing else in the project imports.

    Without one declared, `uv sync` produces an environment where every other test passes and the
    pipeline dies on its fourth stage. That is exactly what happened: h5netcdf and h5py were present
    locally, absent from the lockfile, and the Docker run was the first thing to notice.
    """
    fit = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"sigma": (("chain", "draw"), np.full((2, 50), 0.23))})}
    )
    path = tmp_path / "fit.nc"
    fit.to_netcdf(str(path))

    reloaded = az.from_netcdf(str(path))
    assert list(reloaded.groups) == ["/", "/posterior"]
    assert float(reloaded.posterior["sigma"].mean()) == pytest.approx(0.23)
