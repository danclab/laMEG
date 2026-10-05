"""
Source reconstruction, HDF5 access, and BigBrain mapping utilities.

The public API in this module provides analysis-ready source export,
``LaminarSourceData`` access, backward-compatible source reconstruction, and
persistent BigBrain layer-to-lamina mapping.
"""

import os

import h5py
import numpy as np

from lameg import laminar as _laminar
from lameg._source_export import export_source_time_series_hdf5
from lameg._source_io import load_source_time_series
from lameg._source_schema import _validate_source_file

__all__ = [
    "LaminarSourceData",
    "add_bigbrain_mapping",
    "export_source_time_series_hdf5",
    "load_source_time_series",
]


class LaminarSourceData:  # pylint: disable=too-many-public-methods
    """Reader for analysis-ready laminar source HDF5 files.

    Files are expected to contain ``source_ts`` with axis order
    ``layer, column, time`` or ``layer, column, time, trial`` and a
    corresponding ``time_ms`` dataset.

    Parameters
    ----------
    fname : str or path-like
        Path to an HDF5 file created by ``export_source_time_series_hdf5``.

    Notes
    -----
    ``layer`` refers to the reconstructed layered surfaces. ``lamina``
    refers to a BigBrain-defined histological compartment. Files without an
    added BigBrain mapping can still be read normally, but ``lamina()`` is
    available only after ``add_bigbrain_mapping`` has been run.
    """

    def __init__(self, fname):
        self.fname = os.path.abspath(os.fspath(fname))
        if not os.path.isfile(self.fname):
            raise FileNotFoundError(f"Source HDF5 file not found: {self.fname!r}.")

        self._file = h5py.File(self.fname, "r")

        try:
            self._validate()
            self._time = np.asarray(self._file["time_ms"], dtype=float)
        except Exception:
            self._file.close()
            self._file = None
            raise

    def _validate(self):
        """Validate the laMEG source-file schema."""
        _validate_source_file(self._file)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __repr__(self):
        state = "closed" if self.closed else "open"

        if self.closed:
            return (
                f"LaminarSourceData("
                f"{self.fname!r}, {state})"
            )

        shape = " x ".join(
            str(value)
            for value in self.shape
        )

        return (
            f"LaminarSourceData("
            f"{self.fname!r}, "
            f"shape={shape}, "
            f"{state})"
        )

    @property
    def closed(self):
        """Whether the underlying HDF5 file is closed."""
        return (
            self._file is None
            or not self._file.id.valid
        )

    def _require_open(self):
        """Raise if the underlying HDF5 file has been closed."""
        if self.closed:
            raise RuntimeError("LaminarSourceData is closed.")

    def close(self):
        """Close the underlying HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def shape(self):
        """Shape of ``source_ts``."""
        self._require_open()
        return self._file["source_ts"].shape

    @property
    def dtype(self):
        """On-disk datatype of ``source_ts``."""
        self._require_open()
        return self._file["source_ts"].dtype

    @property
    def n_layers(self):
        """Number of reconstructed layered surfaces."""
        return self.shape[0]

    @property
    def n_columns(self):
        """Number of cortical columns."""
        return self.shape[1]

    @property
    def n_times(self):
        """Number of time samples."""
        return self.shape[2]

    @property
    def has_trials(self):
        """Whether ``source_ts`` has an explicit trial axis."""
        return len(self.shape) == 4

    @property
    def n_trials(self):
        """Number of trials, or 1 if no explicit trial axis exists."""
        return (
            self.shape[3]
            if self.has_trials
            else 1
        )

    @property
    def time(self):
        """Stored time vector in milliseconds."""
        self._require_open()
        return self._time.copy()

    @property
    def woi(self):
        """Sliding inversion windows in milliseconds, if stored."""
        self._require_open()

        if "woi_ms" not in self._file:
            return None

        return np.asarray(self._file["woi_ms"], dtype=float)

    @property
    def window_count(self):
        """Number of inversion windows contributing to each time sample."""
        self._require_open()

        if "window_count" not in self._file:
            return None

        return np.asarray(self._file["window_count"], dtype=np.int32)

    @property
    def vertices(self):
        """Source vertex coordinates as layer x column x xyz, if stored."""
        self._require_open()

        if "source_vertices" not in self._file:
            return None

        return np.asarray(self._file["source_vertices"])

    @property
    def axis_order(self):
        """Stored source axis-order description."""
        self._require_open()

        value = self._file.attrs.get("axis_order")

        if value is None:
            return (
                "layer,column,time,trial"
                if self.has_trials
                else "layer,column,time"
            )

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        return str(value)

    @property
    def has_bigbrain_mapping(self):
        """Whether a complete BigBrain mapping is stored in the source file."""
        self._require_open()
        return (
            "layer_depth" in self._file
            and "bigbrain" in self._file
            and all(
                name in self._file["bigbrain"]
                for name in (
                    "edges",
                    "weights",
                    "labels",
                )
            )
        )

    def _require_bigbrain_mapping(self):
        """Raise if no BigBrain laminar mapping is stored."""
        if not self.has_bigbrain_mapping:
            raise ValueError(
                "This source file has no BigBrain mapping. "
                "Run `add_bigbrain_mapping(source_fname, surf_set)` first."
            )

    @staticmethod
    def _decode_string_array(values):
        """Decode an HDF5 string array to a tuple of Python strings."""
        decoded = []

        for value in np.asarray(values).ravel():
            if isinstance(value, bytes):
                decoded.append(value.decode("utf-8", errors="replace"))
            else:
                decoded.append(str(value))

        return tuple(decoded)

    @property
    def layer_depth(self):
        """Reconstructed layer depths, with 0=pial and 1=white, if stored."""
        self._require_open()

        if not self.has_bigbrain_mapping:
            return None

        return np.asarray(self._file["layer_depth"], dtype=float)

    @property
    def laminae(self):
        """Stored BigBrain lamina labels, or None if no mapping is present."""
        self._require_open()

        if not self.has_bigbrain_mapping:
            return None

        return self._decode_string_array(self._file["bigbrain"]["labels"][()])

    @property
    def n_laminae(self):
        """Number of stored BigBrain laminae, or 0 if no mapping is present."""
        labels = self.laminae
        return 0 if labels is None else len(labels)

    @property
    def bigbrain_edges(self):
        """BigBrain laminar boundaries as column x boundary, if stored."""
        self._require_open()

        if not self.has_bigbrain_mapping:
            return None

        return np.asarray(self._file["bigbrain"]["edges"], dtype=float)

    @property
    def bigbrain_weights(self):
        """Full BigBrain mapping weights as column x lamina x layer, if stored."""
        self._require_open()

        if not self.has_bigbrain_mapping:
            return None

        return np.asarray(self._file["bigbrain"]["weights"], dtype=float)


    @property
    def bigbrain_valid_columns(self):
        """Boolean mask of cortical columns with valid BigBrain mappings."""
        self._require_open()

        if not self.has_bigbrain_mapping:
            return None

        group = self._file["bigbrain"]
        if "valid_columns" in group:
            return np.asarray(
                group["valid_columns"],
                dtype=bool,
            )

        edges = np.asarray(
            group["edges"],
            dtype=float,
        )
        weights = np.asarray(
            group["weights"],
            dtype=float,
        )
        return (
            np.all(
                np.isfinite(edges),
                axis=1,
            )
            & np.all(
                np.isfinite(weights),
                axis=(1, 2),
            )
        )

    def _normalize_lamina_selector(self, lamina):
        """Normalize a BigBrain lamina label/index selector."""
        self._require_bigbrain_mapping()

        labels = self.laminae
        n_laminae = len(labels)

        if isinstance(lamina, str):
            label = lamina.strip().upper()

            try:
                return labels.index(label)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown lamina {lamina!r}. "
                    f"Valid labels are {labels}."
                ) from exc

        return self._normalize_axis_selector(lamina, n_laminae, "lamina")

    def _normalize_axis_selector(self, selector, size, name):
        """Normalize an integer/slice selector for one source axis."""
        if selector is None:
            return slice(0, size, 1, )

        if isinstance(selector, (int, np.integer)):
            index = int(selector)

            if index < 0:
                index += size

            if index < 0 or index >= size:
                raise IndexError(
                    f"{name} index {selector} "
                    f"out of range for size {size}."
                )

            return index

        if isinstance(selector, slice):
            start, stop, step = selector.indices(size)

            if step <= 0:
                raise ValueError(
                    f"{name} slices must have "
                    "a positive step."
                )

            return slice(start, stop, step)

        raise TypeError(
            f"`{name}` must be an integer, "
            "slice, or None; "
            f"got {type(selector).__name__}."
        )

    def _time_selector(self, time):
        """Convert a millisecond interval into a contiguous HDF5 slice."""
        if time is None:
            return slice(0, self.n_times, 1)

        if not isinstance(time, (tuple, list, np.ndarray)) or len(time) != 2:
            raise TypeError(
                "`time` must be None or a "
                "(start_ms, stop_ms) pair."
            )

        start_ms = float(time[0])
        stop_ms = float(time[1])

        if not np.isfinite(start_ms) or not np.isfinite(stop_ms):
            raise ValueError("Time limits must be finite.")

        if start_ms > stop_ms:
            raise ValueError(
                "Time interval start "
                f"({start_ms}) exceeds stop "
                f"({stop_ms})."
            )

        start_idx = int(np.searchsorted(self._time, start_ms, side="left"))
        stop_idx = int(np.searchsorted(self._time, stop_ms, side="right"))

        if start_idx >= stop_idx:
            raise ValueError(
                f"Time interval "
                f"({start_ms}, {stop_ms}) ms "
                "contains no samples within "
                f"[{self._time[0]}, "
                f"{self._time[-1]}] ms."
            )

        return slice(start_idx, stop_idx, 1)

    def time_values(self, time=None):
        """Return the stored time samples within a millisecond interval."""
        self._require_open()
        return self._time[self._time_selector(time)].copy()

    def layer(self, layer=None, column=None, time=None, trial=None):
        """Read layer-space source data without loading the full HDF5 array.

        Parameters
        ----------
        layer : int, slice, or None, optional
            Zero-based reconstructed layer index. An integer removes the layer
            axis from the returned array. A slice or None preserves it.
        column : int, slice, or None, optional
            Zero-based cortical-column index. An integer removes the column
            axis from the returned array. A slice or None preserves it.
        time : (float, float) or None, optional
            Inclusive time interval ``(start_ms, stop_ms)``. If None, all time
            samples are returned. Time selection is converted to a contiguous
            HDF5 slice using ``numpy.searchsorted``.
        trial : int, slice, or None, optional
            Zero-based trial index for files with an explicit trial axis. An
            integer removes the trial axis. If the file has no trial axis,
            ``trial`` must be None.

        Returns
        -------
        numpy.ndarray
            Requested source data. Integer selectors remove their corresponding
            axes, while unselected/sliced axes are retained.

        Examples
        --------
        ``src.layer(column=6620)``
            Returns ``layer x time`` for a file without a trial axis.

        ``src.layer(layer=4, time=(-75, -25))``
            Returns ``column x selected_time``.

        ``src.layer(layer=4, column=6620)``
            Returns one time series.
        """
        self._require_open()

        layer_sel = self._normalize_axis_selector(layer, self.n_layers, "layer")
        column_sel = self._normalize_axis_selector(column, self.n_columns, "column")
        time_sel = self._time_selector(time)

        source_ds = self._file["source_ts"]

        if self.has_trials:
            trial_sel = self._normalize_axis_selector(trial, self.n_trials, "trial")

            return np.asarray(source_ds[layer_sel, column_sel, time_sel, trial_sel])

        if trial is not None:
            raise ValueError(
                "This source file has no trial axis; "
                "`trial` must be None."
            )

        return np.asarray(source_ds[layer_sel, column_sel, time_sel])

    def lamina(self, lamina=None, column=None, time=None, trial=None):
        """Read BigBrain-mapped laminar source activity on demand.

        Parameters
        ----------
        lamina : {"I", "II", "III", "IV", "V", "VI"}, int, slice, or None, optional
            BigBrain lamina to return. String labels are matched
            case-insensitively. Integer indices are zero-based. An integer or
            string removes the lamina axis from the returned array; a slice or
            None preserves it.
        column : int, slice, or None, optional
            Zero-based cortical-column index. An integer removes the column
            axis from the returned array. A slice or None preserves it.
        time : (float, float) or None, optional
            Inclusive time interval ``(start_ms, stop_ms)``. If None, all time
            samples are returned.
        trial : int, slice, or None, optional
            Zero-based trial index for files with an explicit trial axis. An
            integer removes the trial axis. If the file has no trial axis,
            ``trial`` must be None.

        Returns
        -------
        numpy.ndarray
            BigBrain-mapped source activity. With no scalar selections, the
            axis order is ``lamina x column x time`` or
            ``lamina x column x time x trial``.

        Notes
        -----
        Laminar activity is not stored redundantly. The requested layer-space
        source data and column-specific BigBrain weights are read from HDF5 and
        combined on demand using ``surface_to_laminae``.

        Examples
        --------
        ``src.lamina(column=6620, time=(-300, 100))``
            Returns ``lamina x selected_time`` for one cortical column.

        ``src.lamina(lamina="V", time=(-75, -25))``
            Returns ``column x selected_time``.

        ``src.lamina(lamina="V", column=6620)``
            Returns one laminar time series.
        """
        self._require_open()
        self._require_bigbrain_mapping()

        lamina_sel = self._normalize_lamina_selector(lamina)
        column_sel = self._normalize_axis_selector(column, self.n_columns, "column")

        layer_data = self.layer(layer=None, column=column, time=time, trial=trial)

        weights_ds = self._file["bigbrain"]["weights"]

        weights = np.asarray(weights_ds[column_sel, :, :], dtype=float)

        lamina_data = _laminar.surface_to_laminae(
            layer_data,
            weights,
        )

        return np.asarray(lamina_data[lamina_sel, ...])


def _bigbrain_dataset_kwargs(compression):
    """Return HDF5 creation options for BigBrain mapping arrays."""
    if compression not in ("lzf", "gzip", None):
        raise ValueError(
            "`compression` must be 'lzf', 'gzip', or None."
        )

    if compression is None:
        return {}

    kwargs = {
        "compression": compression,
        "shuffle": True,
    }
    if compression == "gzip":
        kwargs["compression_opts"] = 4

    return kwargs


def _prepare_bigbrain_mapping(
    source_fname,
    surf_set,
    compression,
):
    """Validate inputs and compute a BigBrain mapping for a source file."""
    source_fname = os.path.abspath(
        os.fspath(source_fname)
    )
    dataset_kwargs = _bigbrain_dataset_kwargs(
        compression
    )

    with LaminarSourceData(source_fname) as source:
        n_layers = source.n_layers
        n_columns = source.n_columns

    layer_spacing = np.asarray(
        surf_set.layer_spacing,
        dtype=float,
    )
    if layer_spacing.ndim != 1:
        raise ValueError(
            "`surf_set.layer_spacing` must be one-dimensional."
        )
    if layer_spacing.size != n_layers:
        raise ValueError(
            "Layer-count mismatch between source file and surface set: "
            f"source file has {n_layers} layers but "
            f"`surf_set.layer_spacing` has {layer_spacing.size}."
        )

    layer_depth = 1.0 - layer_spacing
    if np.any(~np.isfinite(layer_depth)):
        raise ValueError(
            "`surf_set.layer_spacing` produced non-finite layer depths."
        )
    if (
        layer_depth.size > 1
        and np.any(np.diff(layer_depth) <= 0)
    ):
        raise ValueError(
            "Layer depths must increase monotonically from "
            "pial (0) to white (1)."
        )

    edges, weights = (
        _laminar.compute_bigbrain_laminar_weights(
            surf_set
        )
    )
    edges = np.asarray(
        edges,
        dtype=np.float64,
    )
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    expected_edges_shape = (
        n_columns,
        7,
    )
    expected_weights_shape = (
        n_columns,
        6,
        n_layers,
    )
    if edges.shape != expected_edges_shape:
        raise ValueError(
            "BigBrain edge dimensions do not match the source file: "
            f"expected {expected_edges_shape}, got {edges.shape}."
        )
    if weights.shape != expected_weights_shape:
        raise ValueError(
            "BigBrain weight dimensions do not match the source file: "
            f"expected {expected_weights_shape}, got {weights.shape}."
        )
    finite_edges = np.all(
        np.isfinite(edges),
        axis=1,
    )
    finite_weights = np.all(
        np.isfinite(weights),
        axis=(1, 2),
    )
    if not np.array_equal(
        finite_edges,
        finite_weights,
    ):
        raise ValueError(
            "BigBrain edge and weight validity masks do not match."
        )

    valid_columns = finite_edges
    if not np.any(valid_columns):
        raise ValueError(
            "BigBrain mapping contains no valid cortical columns."
        )

    invalid_columns = ~valid_columns
    if (
        np.any(invalid_columns)
        and (
            not np.all(
                np.isnan(
                    edges[invalid_columns]
                )
            )
            or not np.all(
                np.isnan(
                    weights[invalid_columns]
                )
            )
        )
    ):
        raise ValueError(
            "Invalid BigBrain columns must be represented by NaN "
            "edges and weights."
        )

    labels = np.asarray(
        ["I", "II", "III", "IV", "V", "VI"],
        dtype="S3",
    )
    return (
        source_fname,
        layer_depth,
        edges,
        weights,
        valid_columns,
        labels,
        dataset_kwargs,
    )


def _write_bigbrain_mapping(
    source_fname,
    layer_depth,
    edges,
    weights,
    valid_columns,
    labels,
    dataset_kwargs,
    subj_id,
    overwrite,
):
    """Atomically replace the stored BigBrain mapping."""
    temp_layer_name = "__layer_depth_tmp__"
    temp_group_name = "__bigbrain_tmp__"

    with h5py.File(
        source_fname,
        "r+",
    ) as source_file:
        mapping_exists = (
            "layer_depth" in source_file
            or "bigbrain" in source_file
        )
        if mapping_exists and not overwrite:
            raise FileExistsError(
                "BigBrain mapping already exists in "
                f"{source_fname!r}. Pass overwrite=True to replace it."
            )

        _remove_hdf5_objects(
            source_file,
            (
                temp_layer_name,
                temp_group_name,
            ),
        )

        try:
            _write_bigbrain_temporary_objects(
                source_file,
                temp_layer_name,
                temp_group_name,
                layer_depth,
                edges,
                weights,
                valid_columns,
                labels,
                dataset_kwargs,
                subj_id,
            )
            _remove_hdf5_objects(
                source_file,
                ("layer_depth", "bigbrain"),
            )
            source_file.move(
                temp_layer_name,
                "layer_depth",
            )
            source_file.move(
                temp_group_name,
                "bigbrain",
            )
        except Exception:
            _remove_hdf5_objects(
                source_file,
                (
                    temp_layer_name,
                    temp_group_name,
                ),
            )
            raise


def _remove_hdf5_objects(h5_file, names):
    """Delete named HDF5 objects when present."""
    for name in names:
        if name in h5_file:
            del h5_file[name]


def _write_bigbrain_temporary_objects(
    source_file,
    layer_name,
    group_name,
    layer_depth,
    edges,
    weights,
    valid_columns,
    labels,
    dataset_kwargs,
    subj_id,
):
    """Write a complete BigBrain mapping under temporary object names."""
    layer_ds = source_file.create_dataset(
        layer_name,
        data=layer_depth.astype(
            np.float64,
            copy=False,
        ),
    )
    layer_ds.attrs["axis_order"] = "layer"
    layer_ds.attrs[
        "depth_convention"
    ] = "0=pial,1=white"

    group = source_file.create_group(
        group_name
    )

    edge_ds = group.create_dataset(
        "edges",
        data=edges,
        **dataset_kwargs,
    )
    edge_ds.attrs["axis_order"] = "column,boundary"
    edge_ds.attrs[
        "depth_convention"
    ] = "0=pial,1=white"
    edge_ds.attrs["boundary_order"] = (
        "pial,end_I,end_II,end_III,"
        "end_IV,end_V,end_VI"
    )

    weight_ds = group.create_dataset(
        "weights",
        data=weights,
        **dataset_kwargs,
    )
    weight_ds.attrs[
        "axis_order"
    ] = "column,lamina,layer"
    weight_ds.attrs[
        "mapping"
    ] = "lamina = weights @ layer"

    valid_ds = group.create_dataset(
        "valid_columns",
        data=np.asarray(
            valid_columns,
            dtype=bool,
        ),
    )
    valid_ds.attrs["axis_order"] = "column"
    valid_ds.attrs["meaning"] = (
        "True where BigBrain laminar boundaries are valid"
    )

    label_ds = group.create_dataset(
        "labels",
        data=labels,
    )
    label_ds.attrs["axis_order"] = "lamina"

    group.attrs[
        "depth_convention"
    ] = "0=pial,1=white"
    group.attrs["n_laminae"] = 6
    group.attrs["n_valid_columns"] = int(
        np.count_nonzero(
            valid_columns
        )
    )
    group.attrs["n_invalid_columns"] = int(
        np.count_nonzero(
            ~np.asarray(
                valid_columns,
                dtype=bool,
            )
        )
    )
    if subj_id is not None:
        group.attrs["subject_id"] = str(
            subj_id
        )


def add_bigbrain_mapping(source_fname, surf_set, overwrite=False, compression="lzf"):
    """Add a BigBrain laminar mapping to an exported source HDF5 file.

    The mapping is stored independently of the source time series so that an
    existing source reconstruction does not need to be regenerated when the
    anatomical laminar mapping is added or updated.

    The resulting HDF5 layout is::

        /layer_depth
            (layer,)

        /bigbrain/edges
            (column, 7)

        /bigbrain/weights
            (column, 6, layer)

        /bigbrain/labels
            ["I", "II", "III", "IV", "V", "VI"]

        /bigbrain/valid_columns
            (column,)

    ``weights[column]`` is the linear transformation from reconstructed layer
    activity to mean activity in BigBrain laminae I-VI for that cortical
    column.

    Parameters
    ----------
    source_fname : str or path-like
        HDF5 source file created by ``export_source_time_series_hdf5``.
    surf_set : LayerSurfaceSet
        Surface set corresponding to the source reconstruction.
    overwrite : bool, optional
        Replace an existing ``layer_depth``/``bigbrain`` mapping when True.
        Default is False.
    compression : {"lzf", "gzip", None}, optional
        HDF5 compression used for ``edges`` and ``weights``. Default is
        ``"lzf"``.

    Returns
    -------
    str
        Absolute path to the modified source HDF5 file.

    Notes
    -----
    The depth convention used throughout the stored mapping is::

        0 = pial boundary
        1 = grey/white boundary

    Reconstructed layer depths are stored in the same coordinate system as
    the BigBrain cumulative laminar boundaries.
    """
    (
        source_fname,
        layer_depth,
        edges,
        weights,
        valid_columns,
        labels,
        dataset_kwargs,
    ) = _prepare_bigbrain_mapping(
        source_fname,
        surf_set,
        compression,
    )
    _write_bigbrain_mapping(
        source_fname,
        layer_depth,
        edges,
        weights,
        valid_columns,
        labels,
        dataset_kwargs,
        getattr(surf_set, "subj_id", None),
        overwrite,
    )
    return source_fname
