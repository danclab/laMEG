"""
Validation helpers for analysis-ready laMEG source HDF5 files.
"""

import numpy as np


def _validate_source_file(h5_file):
    """Validate the complete laMEG source-file schema."""
    source_ds = _validate_source_schema(
        h5_file
    )
    _validate_source_vertices(
        h5_file,
        source_ds,
    )
    _validate_bigbrain_schema(
        h5_file,
        source_ds,
    )


def _validate_source_schema(h5_file):
    """Validate required source/time datasets and core metadata."""
    required = ("source_ts", "time_ms")
    missing = [
        name
        for name in required
        if name not in h5_file
    ]
    if missing:
        raise ValueError(
            "Not a valid laMEG source file: "
            f"missing dataset(s) {missing}."
        )

    source_ds = h5_file["source_ts"]
    if source_ds.ndim not in (3, 4):
        raise ValueError(
            "`source_ts` must have shape "
            "layer x column x time or "
            "layer x column x time x trial; "
            f"got shape {source_ds.shape}."
        )

    time_ds = h5_file["time_ms"]
    if time_ds.ndim != 1:
        raise ValueError(
            "`time_ms` must be one-dimensional; "
            f"got shape {time_ds.shape}."
        )

    if source_ds.shape[2] != time_ds.shape[0]:
        raise ValueError(
            "Time dimension mismatch: "
            f"source_ts has {source_ds.shape[2]} samples but "
            f"time_ms has {time_ds.shape[0]}."
        )

    _validate_dimension_attribute(
        h5_file,
        "n_layers",
        source_ds.shape[0],
    )
    _validate_dimension_attribute(
        h5_file,
        "n_columns",
        source_ds.shape[1],
    )

    time = np.asarray(
        time_ds,
        dtype=float,
    )
    if (
        time.size > 1
        and np.any(np.diff(time) <= 0)
    ):
        raise ValueError(
            "`time_ms` must be strictly increasing."
        )

    return source_ds


def _validate_dimension_attribute(
    h5_file,
    name,
    expected,
):
    """Validate an optional integer dimension attribute."""
    value = h5_file.attrs.get(name)
    if (
        value is not None
        and int(value) != expected
    ):
        raise ValueError(
            f"`{name}` attribute does not match source_ts: "
            f"{int(value)} != {expected}."
        )


def _validate_source_vertices(
    h5_file,
    source_ds,
):
    """Validate optional source coordinates against source dimensions."""
    if "source_vertices" not in h5_file:
        return

    expected = (
        source_ds.shape[0],
        source_ds.shape[1],
        3,
    )
    actual = h5_file[
        "source_vertices"
    ].shape
    if actual != expected:
        raise ValueError(
            "`source_vertices` has incompatible shape: "
            f"expected {expected}, got {actual}."
        )


def _validate_bigbrain_schema(
    h5_file,
    source_ds,
):
    """Validate the optional stored BigBrain mapping."""
    has_layer_depth = "layer_depth" in h5_file
    has_bigbrain_group = "bigbrain" in h5_file

    if has_layer_depth != has_bigbrain_group:
        raise ValueError(
            "Incomplete BigBrain mapping: `layer_depth` and `bigbrain` "
            "must either both be present or both be absent."
        )

    if not has_bigbrain_group:
        return

    bigbrain_group = h5_file["bigbrain"]
    required = (
        "edges",
        "weights",
        "labels",
    )
    missing = [
        name
        for name in required
        if name not in bigbrain_group
    ]
    if missing:
        raise ValueError(
            "Incomplete BigBrain mapping: missing dataset(s) "
            f"{missing}."
        )

    labels_ds = bigbrain_group["labels"]
    if labels_ds.ndim != 1:
        raise ValueError(
            "`bigbrain/labels` must be one-dimensional; "
            f"got shape {labels_ds.shape}."
        )

    n_laminae = labels_ds.shape[0]
    if n_laminae < 1:
        raise ValueError(
            "`bigbrain/labels` must contain at least one lamina."
        )

    _validate_mapping_shapes(
        h5_file,
        bigbrain_group,
        source_ds,
        n_laminae,
    )

    if "valid_columns" in bigbrain_group:
        valid_ds = bigbrain_group[
            "valid_columns"
        ]
        expected_valid_shape = (
            source_ds.shape[1],
        )
        if valid_ds.shape != expected_valid_shape:
            raise ValueError(
                "`bigbrain/valid_columns` has incompatible shape: "
                f"expected {expected_valid_shape}, got {valid_ds.shape}."
            )

    n_laminae_attr = bigbrain_group.attrs.get(
        "n_laminae"
    )
    if (
        n_laminae_attr is not None
        and int(n_laminae_attr) != n_laminae
    ):
        raise ValueError(
            "`bigbrain` n_laminae attribute does not match labels: "
            f"{int(n_laminae_attr)} != {n_laminae}."
        )


def _validate_mapping_shapes(
    h5_file,
    bigbrain_group,
    source_ds,
    n_laminae,
):
    """Validate stored BigBrain array dimensions."""
    expected_shapes = {
        "layer_depth": (
            source_ds.shape[0],
        ),
        "edges": (
            source_ds.shape[1],
            n_laminae + 1,
        ),
        "weights": (
            source_ds.shape[1],
            n_laminae,
            source_ds.shape[0],
        ),
    }
    actual_shapes = {
        "layer_depth": h5_file[
            "layer_depth"
        ].shape,
        "edges": bigbrain_group[
            "edges"
        ].shape,
        "weights": bigbrain_group[
            "weights"
        ].shape,
    }

    for name, expected in expected_shapes.items():
        actual = actual_shapes[name]
        if actual == expected:
            continue

        dataset = (
            f"`bigbrain/{name}`"
            if name != "layer_depth"
            else "`layer_depth`"
        )
        raise ValueError(
            f"{dataset} has incompatible shape: "
            f"expected {expected}, got {actual}."
        )
