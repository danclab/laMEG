"""
Internal HDF5 exporter for sliding-window source reconstructions.

The public ``export_source_time_series_hdf5`` name is re-exported from
``lameg.source``.
"""

import os

import h5py
import numpy as np

from pkg_resources import (
    DistributionNotFound,
    get_distribution,
)

from lameg._source_io import (
    _ensure_m_orientation,
    _indices_for_woi,
    _load_inverse_components,
    _make_window_matrix_loaders,
)
from lameg._source_schema import (
    LAYER_DEPTH_CONVENTION,
    SOURCE_SCHEMA_VERSION,
)
from lameg.invert import (
    check_inversion_exists,
    load_forward_model_vertices,
)
from lameg.util import load_meg_sensor_data


def _get_lameg_version():
    """Return the installed laMEG package version."""
    try:
        return get_distribution("lameg").version
    except DistributionNotFound:
        return "unknown"


# pylint: disable=too-many-branches,too-many-statements
def export_source_time_series_hdf5(
    data_fname,
    out_fname,
    surf_set,
    inv_fname=None,
    inversion_idx=0,
    orientation="link_vector",
    fixed=True,
    block_size=512,
    dtype="float32",
    compression="lzf",
    compression_opts=4,
    overwrite=False,
    max_memory_gb=16.0,
):
    """Reconstruct a sliding-window inverse to an analysis-ready HDF5 file.

    The output source dataset is stored as ``layer x cortical_column x time``
    (or ``layer x cortical_column x time x trial`` for multi-trial data).

    For typical single-trial/averaged EBBlayer datasets this function uses a
    fast window-major reconstruction: each window-specific ``M`` matrix is
    opened exactly once, source activity is reconstructed for that window, and
    the overlapping estimates are accumulated in RAM.  This is substantially
    faster than repeatedly reopening every ``M`` file for small cortical
    blocks.  If the full accumulator would exceed ``max_memory_gb``, the
    function automatically falls back to the lower-memory blockwise strategy.

    Parameters
    ----------
    data_fname : str
        SPM-compatible M/EEG dataset containing the sensor data.
    out_fname : str
        HDF5 file to create.
    surf_set : LayerSurfaceSet
        Laminar surface hierarchy used to construct the forward model.
        The number of layers, number of cortical columns, and normalized
        cortical-depth coordinates are derived from this object.
    inv_fname : str, optional
        File containing the inverse. Defaults to ``data_fname``.
    inversion_idx : int, optional
        Zero-based inversion index.
    orientation : str, optional
        Orientation variant of the downsampled multilayer surface.
        Default is ``"link_vector"``.
    fixed : bool, optional
        Whether the forward model used fixed source orientations.
        Default is True.
    block_size : int, optional
        Cortical columns per block for the low-memory fallback path.
    dtype : str or numpy dtype, optional
        On-disk source datatype. Accumulation is performed in float64.
    compression : {"lzf", "gzip", None}, optional
        HDF5 compression filter. ``lzf`` is the default because it is much
        faster than gzip for floating-point source data. Use ``None`` for the
        fastest export and largest file.
    compression_opts : int, optional
        Gzip compression level. Ignored for LZF/None.
    overwrite : bool, optional
        Replace an existing output file when True.
    max_memory_gb : float, optional
        Maximum RAM allocated to the full source accumulator before falling
        back to blockwise export. Default is 16 GB.

    Returns
    -------
    str
        Absolute path to the created HDF5 file.

    Notes
    -----
    Source ordering is assumed to be layer-major, matching the multilayer SPM
    meshes used by laMEG: all columns from layer 0, followed by all columns
    from layer 1, and so on.

    For each inversion window the reconstruction uses associativity to avoid
    constructing the large ``M @ U`` matrix explicitly::

        (M @ U) @ Y == M @ (U @ Y)

    This is mathematically equivalent (up to floating-point rounding) and is
    much cheaper because the window contains far fewer time samples than there
    are sensors.
    """
    if inv_fname is None:
        inv_fname = data_fname

    out_fname = os.path.abspath(os.fspath(out_fname))
    if os.path.exists(out_fname) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_fname!r}. "
            "Pass overwrite=True to replace it."
        )

    n_layers = int(surf_set.n_layers)

    if n_layers < 1:
        raise ValueError("`surf_set.n_layers` must be a positive integer.")

    layer_spacing = np.asarray(surf_set.layer_spacing, dtype=float)

    if layer_spacing.ndim != 1 or layer_spacing.size != n_layers:
        raise ValueError(
            "`surf_set.layer_spacing` must contain "
            "one value per reconstructed layer."
        )

    layer_depth = 1.0 - layer_spacing

    if np.any(~np.isfinite(layer_depth)):
        raise ValueError("`surf_set.layer_spacing` produced non-finite cortical depths.")

    if np.any((layer_depth < 0.0) | (layer_depth > 1.0)):
        raise ValueError("Derived cortical depths must lie between 0 and 1.")

    if layer_depth.size > 1 and np.any(np.diff(layer_depth) <= 0):
        raise ValueError(
            "Derived cortical depths must increase "
            "from pial (0) to white matter (1)."
        )

    if not np.isclose(layer_depth[0], 0.0):
        raise ValueError("The first reconstructed surface must correspond to pial depth 0.")

    if not np.isclose(layer_depth[-1], 1.0):
        raise ValueError(
            "The final reconstructed surface must "
            "correspond to white-matter depth 1."
        )

    n_columns = int(surf_set.get_vertices_per_layer(orientation=orientation, fixed=fixed))

    if n_columns < 1:
        raise ValueError("`surf_set` contains no cortical columns.")

    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("`block_size` must be a positive integer.")
    if float(max_memory_gb) <= 0:
        raise ValueError("`max_memory_gb` must be > 0.")

    sensor_data, time_ms, _ = load_meg_sensor_data(data_fname)
    check_inversion_exists(inv_fname, inversion_idx=inversion_idx)
    invc = _load_inverse_components(inv_fname, inversion_idx=inversion_idx)

    if not any(key in invc for key in ("M_win_files", "M_win_hdf5_paths", "M_win")):
        raise ValueError("`export_source_time_series_hdf5` requires a sliding-window inverse.")

    u_matrix = invc["U"]
    n_spatial_modes = u_matrix.shape[0]
    woi = np.asarray(invc["woi"], dtype=float)
    n_time = sensor_data.shape[1]
    n_trials = sensor_data.shape[2] if sensor_data.ndim == 3 else 1

    # ------------------------------------------------------------------
    # Build full-matrix and row-selective loaders.
    # External filenames are resolved once, not once per block/window.
    # ------------------------------------------------------------------
    loaders = _make_window_matrix_loaders(invc, inv_fname, data_fname)
    n_windows = loaders["n_windows"]
    n_sources_total = loaders["n_sources"]
    load_full = loaders["load_full"]
    load_rows = loaders["load_rows"]

    if n_windows != woi.shape[0]:
        raise ValueError(
            f"Sliding-window inverse has {n_windows} M matrices but "
            f"`woi` has {woi.shape[0]} rows."
        )

    expected_sources = n_layers * n_columns

    if n_sources_total != expected_sources:
        raise ValueError(
            "Source geometry mismatch: "
            f"inverse contains {n_sources_total} sources, "
            f"but the supplied LayerSurfaceSet defines "
            f"{n_layers} layers x {n_columns} columns "
            f"= {expected_sources} sources."
        )

    win_indices = [_indices_for_woi(time_ms, window) for window in woi]

    # Source-independent overlap count.
    count = np.zeros((n_time,), dtype=np.int32)
    for time_idx in win_indices:
        if time_idx.size:
            count[time_idx] += 1
    nz_idx = count > 0

    output_dtype = np.dtype(dtype)
    source_shape = (
        (n_layers, n_columns, n_time, n_trials)
        if sensor_data.ndim == 3
        else (n_layers, n_columns, n_time)
    )

    chunk_columns = min(256, n_columns)
    chunk_time = min(256, n_time)
    chunks = (
        (1, chunk_columns, chunk_time, 1)
        if sensor_data.ndim == 3
        else (1, chunk_columns, chunk_time)
    )

    create_kwargs = {
        "shape": source_shape,
        "dtype": output_dtype,
        "chunks": chunks,
    }
    if compression is not None:
        create_kwargs["compression"] = compression
        if compression == "gzip":
            create_kwargs["compression_opts"] = compression_opts
            create_kwargs["shuffle"] = True
        elif compression == "lzf":
            create_kwargs["shuffle"] = True

    os.makedirs(os.path.dirname(out_fname) or ".", exist_ok=True)
    mode = "w" if overwrite else "x"

    accumulator_bytes = int(n_sources_total) * int(n_time) * int(n_trials) * 8
    accumulator_gb = accumulator_bytes / (1024.0 ** 3)
    use_fast_path = accumulator_gb <= float(max_memory_gb)

    if use_fast_path:
        print(
            "Exporting source time series using window-major reconstruction "
            f"(~{accumulator_gb:.2f} GB accumulator)."
        )
    else:
        print(
            "Full source accumulator would require "
            f"~{accumulator_gb:.2f} GB; using blockwise fallback "
            f"(block_size={block_size})."
        )

    with h5py.File(out_fname, mode) as out_file:

        out_file.attrs["lameg_schema_version"] = SOURCE_SCHEMA_VERSION
        out_file.attrs["lameg_version"] = _get_lameg_version()
        out_file.attrs["subject_id"] = str(surf_set.subj_id)
        out_file.attrs["orientation_method"] = str(orientation)
        out_file.attrs["fixed_orientation"] = bool(fixed)
        out_file.attrs["source_geometry"] = "LayerSurfaceSet"

        source_ds = out_file.create_dataset("source_ts", **create_kwargs)

        layer_ds = out_file.create_dataset(
            "layer_depth",
            data=layer_depth.astype(
                np.float64,
                copy=False
            )
        )
        layer_ds.attrs["axis_order"] = "layer"
        layer_ds.attrs["depth_convention"] = LAYER_DEPTH_CONVENTION

        out_file.create_dataset("time_ms", data=np.asarray(time_ms, dtype=np.float64))
        out_file.create_dataset("woi_ms", data=woi.astype(np.float64, copy=False))
        out_file.create_dataset("window_count", data=count)

        out_file.attrs["axis_order"] = (
            "layer,column,time,trial"
            if sensor_data.ndim == 3
            else "layer,column,time"
        )
        out_file.attrs["n_layers"] = int(n_layers)
        out_file.attrs["n_columns"] = int(n_columns)
        out_file.attrs["n_windows"] = int(n_windows)
        out_file.attrs["inversion_idx"] = int(inversion_idx)
        out_file.attrs["source_order"] = "layer-major"
        out_file.attrs["overlap_combination"] = "sample-wise mean"
        out_file.attrs["export_strategy"] = (
            "window-major" if use_fast_path else "blockwise"
        )

        # Store the SPM forward-model vertices in layer-major order.
        # These are tess_mni coordinates; they correspond one-to-one with
        # the LayerSurfaceSet vertices but are not numerically identical
        # to the GIFTI coordinates.
        try:
            vertices = np.asarray(
                load_forward_model_vertices(
                    inv_fname,
                    inversion_idx=inversion_idx,
                ),
                dtype=float,
            )
        except (
                KeyError,
                OSError,
                TypeError,
        ) as exc:
            raise ValueError(
                "Could not recover forward-model "
                "vertices for source export."
            ) from exc

        expected_vertex_shape = (expected_sources, 3)

        if vertices.shape != expected_vertex_shape:
            raise ValueError(
                "Forward-model vertex geometry does not "
                "match the LayerSurfaceSet: "
                f"expected {expected_vertex_shape}, "
                f"got {vertices.shape}."
            )

        vertex_kwargs = {
            "shape": (n_layers, n_columns, 3),
            "dtype": np.float32,
            "chunks": (1, min(1024, n_columns), 3),
        }
        if compression is not None:
            vertex_kwargs["compression"] = compression
            if compression == "gzip":
                vertex_kwargs["compression_opts"] = compression_opts
                vertex_kwargs["shuffle"] = True
            elif compression == "lzf":
                vertex_kwargs["shuffle"] = True
        vertex_ds = out_file.create_dataset("source_vertices", **vertex_kwargs)
        vertex_ds[...] = vertices.reshape(n_layers, n_columns, 3)

        if use_fast_path:
            # ----------------------------------------------------------
            # FAST PATH
            # Each M file is read exactly once.  Importantly, compute
            # M @ (U @ Y_window), not (M @ U) @ Y_window, avoiding a huge
            # sources x sensors intermediate matrix.
            # ----------------------------------------------------------
            if sensor_data.ndim == 3:
                source_sum = np.zeros((n_sources_total, n_time, n_trials), dtype=np.float64)
            else:
                source_sum = np.zeros((n_sources_total, n_time), dtype=np.float64)

            progress_step = max(1, n_windows // 20)

            for win_idx, time_idx in enumerate(win_indices):
                if time_idx.size == 0:
                    continue

                if win_idx == 0 or (win_idx + 1) % progress_step == 0 or win_idx == n_windows - 1:
                    print(f"  window {win_idx + 1}/{n_windows}")

                m_i = load_full(win_idx)

                if sensor_data.ndim == 3:
                    for trial_idx in range(n_trials):
                        reduced_data = u_matrix @ sensor_data[:, time_idx, trial_idx]
                        source_sum[:, time_idx, trial_idx] += np.asarray(m_i @ reduced_data)
                else:
                    reduced_data = u_matrix @ sensor_data[:, time_idx]
                    source_sum[:, time_idx] += np.asarray(m_i @ reduced_data)

                del m_i, reduced_data

            if sensor_data.ndim == 3:
                source_sum[:, nz_idx, :] /= count[nz_idx][None, :, None]
                # Write layer-by-layer to avoid another full-size cast/copy.
                for layer_idx in range(n_layers):
                    row0 = layer_idx * n_columns
                    row1 = row0 + n_columns
                    source_ds[layer_idx, :, :, :] = source_sum[
                        row0:row1, :, :
                    ].astype(output_dtype, copy=False)
            else:
                source_sum[:, nz_idx] /= count[nz_idx][None, :]
                for layer_idx in range(n_layers):
                    row0 = layer_idx * n_columns
                    row1 = row0 + n_columns
                    source_ds[layer_idx, :, :] = source_sum[
                        row0:row1, :
                    ].astype(output_dtype, copy=False)

            del source_sum

        else:
            # ----------------------------------------------------------
            # LOW-MEMORY FALLBACK
            # Retains the old column-block strategy but uses the cheaper
            # M @ (U @ Y_window) multiplication order.
            # ----------------------------------------------------------
            for col_start in range(0, n_columns, block_size):
                col_stop = min(col_start + block_size, n_columns)
                n_block_columns = col_stop - col_start
                print(f"  columns {col_start}:{col_stop} / {n_columns}")

                source_indices = np.concatenate([
                    np.arange(
                        layer_idx * n_columns + col_start,
                        layer_idx * n_columns + col_stop,
                        dtype=np.int64,
                    )
                    for layer_idx in range(n_layers)
                ])

                if sensor_data.ndim == 3:
                    block_sum = np.zeros((source_indices.size, n_time, n_trials), dtype=np.float64)
                else:
                    block_sum = np.zeros((source_indices.size, n_time), dtype=np.float64)

                for win_idx, time_idx in enumerate(win_indices):
                    if time_idx.size == 0:
                        continue

                    m_i = _ensure_m_orientation(
                        load_rows(win_idx, source_indices), n_spatial_modes
                    )

                    if sensor_data.ndim == 3:
                        for trial_idx in range(n_trials):
                            reduced_data = u_matrix @ sensor_data[:, time_idx, trial_idx]
                            block_sum[:, time_idx, trial_idx] += np.asarray(m_i @ reduced_data)
                    else:
                        reduced_data = u_matrix @ sensor_data[:, time_idx]
                        block_sum[:, time_idx] += np.asarray(m_i @ reduced_data)

                    del m_i, reduced_data

                if sensor_data.ndim == 3:
                    block_sum[:, nz_idx, :] /= count[nz_idx][None, :, None]
                    block_out = block_sum.reshape(
                        (
                            n_layers,
                            n_block_columns,
                            n_time,
                            n_trials,
                        )
                    )
                    source_ds[:, col_start:col_stop, :, :] = block_out.astype(
                        output_dtype, copy=False
                    )
                else:
                    block_sum[:, nz_idx] /= count[nz_idx][None, :]
                    block_out = block_sum.reshape(
                        (
                            n_layers,
                            n_block_columns,
                            n_time,
                        )
                    )
                    source_ds[:, col_start:col_stop, :] = block_out.astype(
                        output_dtype, copy=False
                    )

                del block_sum, block_out

    return out_fname
