"""
Unit tests for source-space HDF5 access in ``lameg.source``.
"""

import h5py
import numpy as np
import pytest

from lameg.source import LaminarSourceData
from tests.source_test_utils import make_source_file


def test_laminar_source_data_metadata(source_file):
    """Test metadata, geometry, and context-manager behavior."""
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

        assert not source.has_bigbrain_mapping

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

        np.testing.assert_allclose(
            source.layer_depth,
            expected["layer_depth"],
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

        # -50 through 100 ms is inclusive:
        # samples 1, 2, 3, and 4.
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


def test_laminar_source_data_trial_indexing(
    trial_source_file,
):
    """Test files with an explicit trial dimension."""
    fname, expected = trial_source_file
    data = expected["source"]

    with LaminarSourceData(fname) as source:
        assert source.has_trials
        assert source.n_trials == 2
        assert source.shape == (3, 4, 6, 2)
        assert (
            source.axis_order
            == "layer,column,time,trial"
        )

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


def test_laminar_source_data_rejects_trial_for_trialless_file(
    source_file,
):
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
                layer=slice(
                    None,
                    None,
                    -1,
                )
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
                time=(
                    100.0,
                    -100.0,
                )
            )

        with pytest.raises(
            ValueError,
            match="contains no samples",
        ):
            source.layer(
                time=(
                    500.0,
                    600.0,
                )
            )


def test_laminar_source_data_optional_vertices(tmp_path):
    """Source vertices are optional."""
    fname = (
        tmp_path
        / "source_without_vertices.h5"
    )

    expected = make_source_file(
        fname,
        include_vertices=False,
    )

    with LaminarSourceData(fname) as source:
        assert source.vertices is None

        np.testing.assert_array_equal(
            source.layer(),
            expected["source"],
        )


def test_laminar_source_data_optional_window_metadata(
    tmp_path,
):
    """Window metadata may be absent."""
    fname = (
        tmp_path
        / "source_without_windows.h5"
    )

    make_source_file(fname)

    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        del source_h5["woi_ms"]
        del source_h5["window_count"]

    with LaminarSourceData(fname) as source:
        assert source.woi is None
        assert source.window_count is None


def test_laminar_source_data_missing_file(tmp_path):
    """Opening a nonexistent source file should fail clearly."""
    fname = (
        tmp_path
        / "missing.h5"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Source HDF5 file not found",
    ):
        LaminarSourceData(fname)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_source",
        "missing_time",
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
    """Reject internally inconsistent source datasets."""
    fname = (
        tmp_path
        / f"{failure}.h5"
    )

    make_source_file(fname)

    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        if failure == "missing_source":
            del source_h5["source_ts"]

        elif failure == "missing_time":
            del source_h5["time_ms"]

        elif failure == "bad_source_ndim":
            del source_h5["source_ts"]

            source_h5.create_dataset(
                "source_ts",
                data=np.zeros(
                    (3, 4),
                    dtype=np.float32,
                ),
            )

        elif failure == "time_length":
            del source_h5["time_ms"]

            source_h5.create_dataset(
                "time_ms",
                data=np.arange(5),
            )

        elif failure == "n_layers":
            source_h5.attrs[
                "n_layers"
            ] = 4

        elif failure == "n_columns":
            source_h5.attrs[
                "n_columns"
            ] = 5

        elif failure == "vertices":
            del source_h5[
                "source_vertices"
            ]

            source_h5.create_dataset(
                "source_vertices",
                data=np.zeros(
                    (3, 5, 3),
                    dtype=np.float32,
                ),
            )

        elif failure == "nonmonotonic_time":
            del source_h5["time_ms"]

            source_h5.create_dataset(
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


def test_laminar_source_data_rejects_missing_schema_version(
    tmp_path,
):
    """Schema version is mandatory."""
    fname = (
        tmp_path
        / "missing_schema_version.h5"
    )

    make_source_file(fname)

    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        del source_h5.attrs[
            "lameg_schema_version"
        ]

    with pytest.raises(
        ValueError,
        match="missing `lameg_schema_version`",
    ):
        LaminarSourceData(fname)


def test_laminar_source_data_rejects_unknown_schema(
    tmp_path,
):
    """Unknown schema versions must not be interpreted silently."""
    fname = (
        tmp_path
        / "future_source.h5"
    )

    make_source_file(fname)

    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        source_h5.attrs[
            "lameg_schema_version"
        ] = "99.0"

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported laMEG "
            "source schema version"
        ),
    ):
        LaminarSourceData(fname)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_layer_depth",
        "bad_layer_depth_shape",
        "bad_layer_depth_range",
        "bad_layer_depth_order",
        "bad_layer_depth_start",
        "bad_layer_depth_stop",
        "missing_depth_convention",
    ],
)
def test_laminar_source_data_layer_geometry_validation(
    tmp_path,
    failure,
):
    """Reject malformed cortical-depth geometry."""
    fname = (
        tmp_path
        / f"geometry_{failure}.h5"
    )

    make_source_file(fname)

    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        if failure == "missing_layer_depth":
            del source_h5[
                "layer_depth"
            ]

        elif failure == "bad_layer_depth_shape":
            del source_h5[
                "layer_depth"
            ]

            layer_ds = (
                source_h5.create_dataset(
                    "layer_depth",
                    data=np.array(
                        [
                            0.0,
                            0.33,
                            0.66,
                            1.0,
                        ]
                    ),
                )
            )

            layer_ds.attrs[
                "depth_convention"
            ] = "0=pial,1=white"

        elif failure == "bad_layer_depth_range":
            source_h5[
                "layer_depth"
            ][...] = np.array(
                [
                    0.0,
                    0.5,
                    1.2,
                ]
            )

        elif failure == "bad_layer_depth_order":
            source_h5[
                "layer_depth"
            ][...] = np.array(
                [
                    0.0,
                    0.8,
                    0.6,
                ]
            )

        elif failure == "bad_layer_depth_start":
            source_h5[
                "layer_depth"
            ][...] = np.array(
                [
                    0.1,
                    0.5,
                    1.0,
                ]
            )

        elif failure == "bad_layer_depth_stop":
            source_h5[
                "layer_depth"
            ][...] = np.array(
                [
                    0.0,
                    0.5,
                    0.9,
                ]
            )

        elif failure == "missing_depth_convention":
            del source_h5[
                "layer_depth"
            ].attrs[
                "depth_convention"
            ]

    with pytest.raises(ValueError):
        LaminarSourceData(fname)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_lameg_version",
        "empty_lameg_version",
        "missing_axis_order",
        "bad_axis_order",
        "missing_source_order",
        "bad_source_order",
        "missing_n_layers",
        "missing_n_columns",
    ],
)
def test_laminar_source_data_required_metadata_validation(
    tmp_path,
    failure,
):
    """Reject missing or inconsistent required schema-v1 metadata."""
    fname = (
        tmp_path
        / f"metadata_{failure}.h5"
    )

    make_source_file(fname)

    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        if failure == "missing_lameg_version":
            del source_h5.attrs[
                "lameg_version"
            ]

        elif failure == "empty_lameg_version":
            source_h5.attrs[
                "lameg_version"
            ] = ""

        elif failure == "missing_axis_order":
            del source_h5.attrs[
                "axis_order"
            ]

        elif failure == "bad_axis_order":
            source_h5.attrs[
                "axis_order"
            ] = "column,layer,time"

        elif failure == "missing_source_order":
            del source_h5.attrs[
                "source_order"
            ]

        elif failure == "bad_source_order":
            source_h5.attrs[
                "source_order"
            ] = "column-major"

        elif failure == "missing_n_layers":
            del source_h5.attrs[
                "n_layers"
            ]

        elif failure == "missing_n_columns":
            del source_h5.attrs[
                "n_columns"
            ]

    with pytest.raises(ValueError):
        LaminarSourceData(fname)
