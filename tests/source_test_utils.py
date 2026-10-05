"""
Shared helpers for source-space HDF5 tests.
"""

import h5py
import numpy as np


def make_source_file(
    fname,
    with_trials=False,
    include_vertices=True,
    include_metadata=True
):
    """Create a small deterministic laMEG source HDF5 file for testing."""
    n_layers = 3
    n_columns = 4
    n_times = 6
    n_trials = 2

    time = np.array(
        [
            -100.0,
            -50.0,
            0.0,
            50.0,
            100.0,
            150.0,
        ],
        dtype=np.float64,
    )

    layer_depth = np.array(
        [
            0.0,
            0.5,
            1.0,
        ],
        dtype=np.float64,
    )

    if with_trials:
        shape = (n_layers, n_columns, n_times, n_trials)
    else:
        shape = (n_layers, n_columns, n_times)

    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    vertices = np.arange(
        n_layers * n_columns * 3,
        dtype=np.float32,
    ).reshape(
        n_layers,
        n_columns,
        3,
    )

    woi = np.array(
        [
            [-100.0, 50.0],
            [0.0, 150.0],
        ],
        dtype=np.float64,
    )

    window_count = np.array(
        [
            1,
            1,
            2,
            2,
            1,
            1,
        ],
        dtype=np.int32,
    )

    with h5py.File(fname, "w") as out_file:
        out_file.create_dataset("source_ts", data=source)
        out_file.create_dataset("time_ms",data=time)
        out_file.create_dataset("woi_ms", data=woi)
        out_file.create_dataset("window_count", data=window_count)

        if include_vertices:
            out_file.create_dataset("source_vertices", data=vertices)

        if include_metadata:
            out_file.attrs["axis_order"] = (
                "layer,column,time,trial"
                if with_trials
                else "layer,column,time"
            )
            out_file.attrs["n_layers"] = n_layers
            out_file.attrs["n_columns"] = n_columns
            out_file.attrs["n_windows"] = 2
            out_file.attrs["inversion_idx"] = 0
            out_file.attrs["source_order"] = "layer-major"
            out_file.attrs["overlap_combination"] = "sample-wise mean"

        out_file.attrs["lameg_schema_version"] = "1.0"

        out_file.attrs["lameg_version"] = "test"

        layer_ds = out_file.create_dataset(
            "layer_depth",
            data=np.array(
                [
                    0.0,
                    0.5,
                    1.0,
                ],
                dtype=np.float64,
            ),
        )

        layer_ds.attrs["axis_order"] = "layer"

        layer_ds.attrs["depth_convention"] = "0=pial,1=white"

    return {
        "source": source,
        "time": time,
        "vertices": vertices,
        "woi": woi,
        "window_count": (
            window_count
        ),
        "layer_depth": (
            layer_depth
        ),
    }