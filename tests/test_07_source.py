"""
Unit tests for source-space HDF5 access in ``lameg.source``.
"""

import h5py
import numpy as np
import pytest

from lameg.source import LaminarSourceData
from tests.source_test_utils import make_source_file


def test_laminar_source_data_metadata(source_file):
    """Test metadata properties and context-manager behavior."""
    fname, expected = source_file

    with LaminarSourceData(fname) as source:
        assert not source.closed
        assert source.shape == (3, 4, 6)
        assert source.dtype == np.dtype("float32")
        assert source.n_layers == 3
        assert source.n_columns == 4
        assert source.n_times == 6
        assert not source.has_trials
        assert source.n_trials == 1
        assert source.axis_order == "layer,column,time"

        np.testing.assert_array_equal(
            source.time,
            expected["time"],
        )
        np.testing.assert_array_equal(
            source.woi,
            expected["woi"],
        )
        np.testing.assert_array_equal(
            source.window_count,
            expected["window_count"],
        )
        np.testing.assert_array_equal(
            source.vertices,
            expected["vertices"],
        )

        assert "shape=3 x 4 x 6" in repr(source)
        assert "open" in repr(source)

    assert source.closed
    assert "closed" in repr(source)

    with pytest.raises(
        RuntimeError,
        match="LaminarSourceData is closed",
    ):
        _ = source.shape


def test_laminar_source_data_layer_indexing(source_file):
    """Test layer, column, time, slice, and negative-index access."""
    fname, expected = source_file
    data = expected["source"]

    with LaminarSourceData(fname) as source:
        np.testing.assert_array_equal(
            source.layer(),
            data,
        )
        np.testing.assert_array_equal(
            source.layer(column=2),
            data[:, 2, :],
        )
        np.testing.assert_array_equal(
            source.layer(layer=1),
            data[1, :, :],
        )
        np.testing.assert_array_equal(
            source.layer(
                layer=1,
                column=2,
            ),
            data[1, 2, :],
        )
        np.testing.assert_array_equal(
            source.layer(
                layer=-1,
                column=-1,
            ),
            data[-1, -1, :],
        )

        # -50 through 100 ms is inclusive: samples 1, 2, 3, and 4.
        np.testing.assert_array_equal(
            source.layer(
                layer=slice(0, 3, 2),
                column=slice(1, 4, 2),
                time=(-50.0, 100.0),
            ),
            data[0:3:2, 1:4:2, 1:5],
        )

        np.testing.assert_array_equal(
            source.time_values(
                (-50.0, 100.0)
            ),
            expected["time"][1:5],
        )


def test_laminar_source_data_trial_indexing(trial_source_file):
    """Test files with an explicit trial dimension."""
    fname, expected = trial_source_file
    data = expected["source"]

    with LaminarSourceData(fname) as source:
        assert source.has_trials
        assert source.n_trials == 2
        assert source.shape == (3, 4, 6, 2)
        assert source.axis_order == "layer,column,time,trial"

        np.testing.assert_array_equal(
            source.layer(),
            data,
        )
        np.testing.assert_array_equal(
            source.layer(trial=1),
            data[:, :, :, 1],
        )
        np.testing.assert_array_equal(
            source.layer(
                layer=2,
                column=1,
                time=(0.0, 50.0),
                trial=0,
            ),
            data[2, 1, 2:4, 0],
        )
        np.testing.assert_array_equal(
            source.layer(
                layer=-1,
                trial=-1,
            ),
            data[-1, :, :, -1],
        )


def test_laminar_source_data_rejects_trial_for_trialless_file(source_file):
    """Trial selection should fail when no trial axis is stored."""
    fname, _ = source_file

    with LaminarSourceData(fname) as source:
        with pytest.raises(
            ValueError,
            match="no trial axis",
        ):
            source.layer(trial=0)


def test_laminar_source_data_selector_errors(source_file):
    """Test invalid layer, column, time, and slice selectors."""
    fname, _ = source_file

    with LaminarSourceData(fname) as source:
        with pytest.raises(
            IndexError,
            match="layer index",
        ):
            source.layer(layer=3)

        with pytest.raises(
            IndexError,
            match="column index",
        ):
            source.layer(column=4)

        with pytest.raises(
            TypeError,
            match="`layer` must be",
        ):
            source.layer(layer=[0, 1])

        with pytest.raises(
            ValueError,
            match="positive step",
        ):
            source.layer(
                layer=slice(None, None, -1)
            )

        with pytest.raises(
            TypeError,
            match="`time` must be",
        ):
            source.layer(time=0.0)

        with pytest.raises(
            ValueError,
            match="exceeds stop",
        ):
            source.layer(
                time=(100.0, -100.0)
            )

        with pytest.raises(
            ValueError,
            match="contains no samples",
        ):
            source.layer(
                time=(500.0, 600.0)
            )


def test_laminar_source_data_optional_metadata(tmp_path):
    """Optional datasets and axis-order metadata may be absent."""
    fname = tmp_path / "minimal_source.h5"
    expected = make_source_file(
        fname,
        include_vertices=False,
        include_metadata=False,
    )

    with LaminarSourceData(fname) as source:
        assert source.vertices is None
        assert source.axis_order == "layer,column,time"
        np.testing.assert_array_equal(
            source.layer(),
            expected["source"],
        )


def test_laminar_source_data_missing_file(tmp_path):
    """Opening a nonexistent source file should fail clearly."""
    fname = tmp_path / "missing.h5"

    with pytest.raises(
        FileNotFoundError,
        match="Source HDF5 file not found",
    ):
        LaminarSourceData(fname)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_source",
        "bad_source_ndim",
        "time_length",
        "n_layers",
        "n_columns",
        "vertices",
        "nonmonotonic_time",
    ],
)
def test_laminar_source_data_schema_validation(
    tmp_path,
    failure,
):
    """Test rejection of internally inconsistent HDF5 source files."""
    fname = tmp_path / f"{failure}.h5"

    if failure == "missing_source":
        with h5py.File(fname, "w") as out_file:
            out_file.create_dataset(
                "time_ms",
                data=np.arange(6),
            )

    elif failure == "bad_source_ndim":
        with h5py.File(fname, "w") as out_file:
            out_file.create_dataset(
                "source_ts",
                data=np.zeros(
                    (3, 4),
                    dtype=np.float32,
                ),
            )
            out_file.create_dataset(
                "time_ms",
                data=np.arange(6),
            )

    else:
        make_source_file(fname)

        with h5py.File(fname, "r+") as out_file:
            if failure == "time_length":
                del out_file["time_ms"]
                out_file.create_dataset(
                    "time_ms",
                    data=np.arange(5),
                )

            elif failure == "n_layers":
                out_file.attrs["n_layers"] = 4

            elif failure == "n_columns":
                out_file.attrs["n_columns"] = 5

            elif failure == "vertices":
                del out_file["source_vertices"]
                out_file.create_dataset(
                    "source_vertices",
                    data=np.zeros(
                        (3, 5, 3),
                        dtype=np.float32,
                    ),
                )

            elif failure == "nonmonotonic_time":
                del out_file["time_ms"]
                out_file.create_dataset(
                    "time_ms",
                    data=np.array(
                        [
                            -100.0,
                            -50.0,
                            0.0,
                            0.0,
                            100.0,
                            150.0,
                        ]
                    ),
                )

    with pytest.raises(ValueError):
        LaminarSourceData(fname)
