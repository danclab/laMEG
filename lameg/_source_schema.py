"""
Validation helpers for analysis-ready laMEG source HDF5 files.
"""

import numpy as np

SOURCE_SCHEMA_VERSION = "1.0"
LAYER_DEPTH_CONVENTION = "0=pial,1=white"

def _decode_attribute(value):
    """Return an HDF5 attribute value as a Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def _validate_source_file(h5_file):
    """Validate a complete laMEG source HDF5 file."""
    _validate_schema_version(h5_file)

    source_ds = _validate_source_schema(h5_file)

    _validate_source_metadata(h5_file, source_ds)

    _validate_layer_geometry(h5_file, source_ds)

    _validate_source_vertices(h5_file, source_ds)

    _validate_bigbrain_schema(h5_file, source_ds)


def _validate_schema_version(h5_file):
    """Validate the laMEG source schema version."""
    value = h5_file.attrs.get("lameg_schema_version")

    if value is None:
        raise ValueError(
            "Not a valid laMEG source file: "
            "missing `lameg_schema_version`."
        )

    version = _decode_attribute(value)

    if version != SOURCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported laMEG source schema version: "
            f"{version!r}. "
            f"Supported version is "
            f"{SOURCE_SCHEMA_VERSION!r}."
        )


def _validate_source_schema(h5_file):
    """Validate required source and time datasets."""
    required = (
        "source_ts",
        "time_ms",
    )

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

    if source_ds.ndim not in (3,4):
        raise ValueError(
            "`source_ts` must have shape "
            "layer x column x time or "
            "layer x column x time x trial; "
            f"got shape {source_ds.shape}."
        )

    time_ds = h5_file["time_ms"]

    if time_ds.ndim != 1:
        raise ValueError(
            "`time_ms` must be "
            "one-dimensional; "
            f"got shape {time_ds.shape}."
        )

    if source_ds.shape[2] != time_ds.shape[0]:
        raise ValueError(
            "Time dimension mismatch: "
            f"`source_ts` has "
            f"{source_ds.shape[2]} samples "
            f"but `time_ms` has "
            f"{time_ds.shape[0]}."
        )

    time = np.asarray(time_ds, dtype=float)

    if time.size > 1 and np.any(np.diff(time) <= 0):
        raise ValueError("`time_ms` must be strictly increasing.")

    return source_ds


def _validate_source_metadata(
    h5_file,
    source_ds,
):
    """Validate required source-file metadata."""
    required_attrs = (
        "lameg_version",
        "axis_order",
        "source_order",
        "n_layers",
        "n_columns",
    )

    missing = [
        name
        for name in required_attrs
        if name not in h5_file.attrs
    ]

    if missing:
        raise ValueError(
            "laMEG source schema "
            f"{SOURCE_SCHEMA_VERSION} "
            "requires attribute(s) "
            f"{missing}."
        )

    expected_axis_order = (
        "layer,column,time,trial"
        if source_ds.ndim == 4
        else "layer,column,time"
    )

    axis_order = _decode_attribute(h5_file.attrs["axis_order"])

    if axis_order != expected_axis_order:
        raise ValueError(
            "`axis_order` does not match "
            "`source_ts`: "
            f"expected "
            f"{expected_axis_order!r}, "
            f"got {axis_order!r}."
        )

    source_order = _decode_attribute(h5_file.attrs["source_order"])

    if source_order != "layer-major":
        raise ValueError(
            "Schema v1 requires "
            "`source_order='layer-major'`; "
            f"got {source_order!r}."
        )

    n_layers = int(h5_file.attrs["n_layers"])

    if n_layers != source_ds.shape[0]:
        raise ValueError(
            "`n_layers` does not match "
            "`source_ts`: "
            f"{n_layers} != "
            f"{source_ds.shape[0]}."
        )

    n_columns = int(h5_file.attrs["n_columns"])

    if n_columns != source_ds.shape[1]:
        raise ValueError(
            "`n_columns` does not match "
            "`source_ts`: "
            f"{n_columns} != "
            f"{source_ds.shape[1]}."
        )

    lameg_version = _decode_attribute(h5_file.attrs["lameg_version"])

    if not lameg_version.strip():
        raise ValueError(
            "`lameg_version` must be "
            "a non-empty string."
        )


def _validate_layer_geometry(
    h5_file,
    source_ds,
):
    """Validate reconstructed cortical-depth geometry."""
    if "layer_depth" not in h5_file:
        raise ValueError(
            "laMEG source schema v1 "
            "requires the "
            "`layer_depth` dataset."
        )

    layer_ds = h5_file["layer_depth"]

    expected_shape = (source_ds.shape[0],)

    if layer_ds.shape != expected_shape:
        raise ValueError(
            "`layer_depth` has "
            "incompatible shape: "
            f"expected {expected_shape}, "
            f"got {layer_ds.shape}."
        )

    depth = np.asarray(layer_ds, dtype=float)

    if np.any(~np.isfinite(depth)):
        raise ValueError("`layer_depth` contains non-finite values."
        )

    if np.any((depth < 0.0) | (depth > 1.0)):
        raise ValueError("`layer_depth` values must lie between 0 and 1.")

    if depth.size > 1 and np.any(np.diff(depth) <= 0):
        raise ValueError(
            "`layer_depth` must increase "
            "strictly from pial towards "
            "white matter."
        )

    if not np.isclose(depth[0], 0.0):
        raise ValueError("`layer_depth` must begin at the pial surface (0).")

    if not np.isclose(depth[-1], 1.0):
        raise ValueError("`layer_depth` must end at the white-matter surface (1).")

    convention = layer_ds.attrs.get("depth_convention")

    if convention is None:
        raise ValueError("`layer_depth` must define the `depth_convention` attribute.")

    convention = _decode_attribute(convention)

    if convention != LAYER_DEPTH_CONVENTION:
        raise ValueError(
            "Unsupported "
            "`layer_depth` convention: "
            f"{convention!r}; expected "
            f"{LAYER_DEPTH_CONVENTION!r}."
        )


def _validate_source_vertices(
    h5_file,
    source_ds,
):
    """Validate optional source coordinates."""
    if "source_vertices" not in h5_file:
        return

    expected = (source_ds.shape[0], source_ds.shape[1], 3)

    actual = h5_file["source_vertices"].shape

    if actual != expected:
        raise ValueError(
            "`source_vertices` has "
            "incompatible shape: "
            f"expected {expected}, "
            f"got {actual}."
        )


def _validate_bigbrain_schema(
    h5_file,
    source_ds,
):
    """Validate the optional stored BigBrain mapping."""
    if "bigbrain" not in h5_file:
        return

    group = h5_file["bigbrain"]

    required = (
        "edges",
        "weights",
        "labels",
    )

    missing = [
        name
        for name in required
        if name not in group
    ]

    if missing:
        raise ValueError(
            "Incomplete BigBrain mapping: "
            f"missing dataset(s) {missing}."
        )

    labels_ds = group["labels"]

    if labels_ds.ndim != 1:
        raise ValueError(
            "`bigbrain/labels` must be "
            "one-dimensional; "
            f"got shape "
            f"{labels_ds.shape}."
        )

    n_laminae = labels_ds.shape[0]

    if n_laminae < 1:
        raise ValueError("`bigbrain/labels` must contain at least one lamina.")

    expected_edges = (source_ds.shape[1], n_laminae + 1)

    if group["edges"].shape != expected_edges:
        raise ValueError(
            "`bigbrain/edges` has "
            "incompatible shape: "
            f"expected {expected_edges}, "
            f"got "
            f"{group['edges'].shape}."
        )

    expected_weights = (source_ds.shape[1], n_laminae, source_ds.shape[0])

    if group["weights"].shape != expected_weights:
        raise ValueError(
            "`bigbrain/weights` has "
            "incompatible shape: "
            f"expected "
            f"{expected_weights}, "
            f"got "
            f"{group['weights'].shape}."
        )

    if "valid_columns" in group:
        expected_valid = (source_ds.shape[1],)

        if group["valid_columns"].shape!= expected_valid:
            raise ValueError(
                "`bigbrain/valid_columns` "
                "has incompatible shape: "
                f"expected "
                f"{expected_valid}, "
                f"got "
                f"{group['valid_columns'].shape}."
            )

    n_laminae_attr = group.attrs.get("n_laminae")

    if n_laminae_attr is not None and int(n_laminae_attr) != n_laminae:
        raise ValueError(
            "`bigbrain` n_laminae "
            "attribute does not "
            "match labels: "
            f"{int(n_laminae_attr)} "
            f"!= {n_laminae}."
        )
