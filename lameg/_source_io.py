"""
Internal source reconstruction I/O helpers.

This module contains MATLAB/HDF5 inverse loading and the legacy
``load_source_time_series`` reconstruction path. Public users should import
``load_source_time_series`` from ``lameg.source``.
"""

import os
from functools import partial

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, issparse

from lameg.invert import (
    _h5_to_csc,
    _mat_to_csc,
    check_inversion_exists,
)
from lameg.util import load_meg_sensor_data

# pylint: disable=too-many-branches,too-many-statements
def load_source_time_series(
    data_fname,
    mu_matrix=None,
    inv_fname=None,
    vertices=None,
    inversion_idx=0,
):
    """
    Load or compute source-space time series from MEG data.

    This function reconstructs source-level activity either from precomputed inverse solutions
    (stored in the MEG data file or an external inversion file) or directly from a provided
    forward/inverse matrix (`mu_matrix`). Optionally, the reconstruction can be restricted to a
    subset of vertices.

    Backward-compatible storage support
    -----------------------------------
    - Single-window inversions using ``inverse.M``.
    - Legacy multi-window inversions using embedded ``inverse.M_win``.
    - New disk-backed multi-window inversions using ``inverse.M_win_files``.

    For overlapping windows, source estimates are averaged sample-by-sample
    exactly as in the previous ``M_win`` implementation.

    Parameters
    ----------
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    mu_matrix : np.ndarray or scipy.sparse matrix, optional
        Precomputed source reconstruction matrix (sources × sensors). If provided, this matrix is
        used directly to compute source time series. Default is None.
    inv_fname : str, optional
        Path to a file containing the precomputed inverse solution. If None, the inversion stored
        within `data_fname` is used. Default is None.
    vertices : list of int, optional
        List of vertex indices to extract source time series from. If None, all vertices are used.
    inversion_idx: int, optional
        Index of the inversion to use within the SPM data object (default: 0).

    Returns
    -------
    source_ts : np.ndarray
        Source-space time series (sources × time × trial).
    time : np.ndarray
        Time vector in milliseconds.
    mu_matrix : np.ndarray or scipy.sparse matrix
        Matrix used to reconstruct source activity from sensor data.

    Notes
    -----
    - If both `mu_matrix` and `inv_fname` are None, the function attempts to load the inverse
      solution embedded in `data_fname`.
    - When `vertices` is specified, only the corresponding subset of the inverse matrix is used.
    - Supports both single-trial and multi-trial MEG data structures.

    Multi-woi behavior (when inverse has M_win and woi):
      - For each window i, compute mu_i = M_win[i] @ U (optionally vertex-subsetted),
        apply it to sensor data for the time indices in woi[i].
      - If windows overlap, average source estimates at overlapping time points
        (per source, per trial) using a per-timepoint contribution count.
      - Output has the SAME time axis as the sensor data (sources x time x trial).
    """

    sensor_data, time_ms, _ = load_meg_sensor_data(data_fname)

    if inv_fname is None:
        inv_fname = data_fname

    check_inversion_exists(inv_fname, inversion_idx=inversion_idx)
    invc = _load_inverse_components(inv_fname, inversion_idx=inversion_idx)

    u_matrix = invc["U"]
    n_spatial_modes = u_matrix.shape[0]

    n_time = sensor_data.shape[1]
    n_trials = 1
    if sensor_data.ndim == 3:
        n_trials = sensor_data.shape[2]

    # ------------------------------------------------------------------
    # Multi-WOI path: legacy embedded or new disk-backed M matrices.
    # ------------------------------------------------------------------
    multi_woi = (
            mu_matrix is None
            and any(
        key in invc
        for key in ("M_win", "M_win_hdf5_paths", "M_win_files")
    )
    )

    if multi_woi:
        woi = np.asarray(invc["woi"])
        loaders = _make_window_matrix_loaders(
            invc,
            inv_fname,
            data_fname,
        )
        n_windows = loaders["n_windows"]

        if n_windows != woi.shape[0]:
            raise ValueError(
                f"Sliding-window inverse has {n_windows} M matrices but "
                f"`woi` has {woi.shape[0]} rows."
            )

        win_indices = [_indices_for_woi(time_ms, window) for window in woi]

        # Determine output source count without constructing MU for all sources.
        first_matrix = _ensure_m_orientation(
            _load_window_matrix(loaders, 0, rows=vertices),
            n_spatial_modes,
        )
        n_sources = first_matrix.shape[0]
        del first_matrix

        if sensor_data.ndim == 3:
            source_sum = np.zeros((n_sources, n_time, n_trials), dtype=float)
            count = np.zeros((n_time,), dtype=np.int32)

            for index, time_idx in enumerate(win_indices):
                if time_idx.size == 0:
                    continue

                m_i = _ensure_m_orientation(
                    _load_window_matrix(loaders, index, rows=vertices), n_spatial_modes
                )

                mu_i = m_i @ u_matrix

                for trial_idx in range(n_trials):
                    src_seg = np.asarray(mu_i @ sensor_data[:, time_idx, trial_idx])
                    source_sum[:, time_idx, trial_idx] += src_seg

                count[time_idx] += 1
                del m_i, mu_i

            nz_idx = count > 0
            source_ts = np.zeros_like(source_sum)
            source_ts[:, nz_idx, :] = source_sum[:, nz_idx, :] / count[nz_idx][None, :, None]

        else:
            source_sum = np.zeros((n_sources, n_time), dtype=float)
            count = np.zeros((n_time,), dtype=np.int32)

            for index, time_idx in enumerate(win_indices):
                if time_idx.size == 0:
                    continue

                m_i = _ensure_m_orientation(
                    _load_window_matrix(loaders, index, rows=vertices), n_spatial_modes
                )

                mu_i = m_i @ u_matrix
                src_seg = np.asarray(mu_i @ sensor_data[:, time_idx])
                source_sum[:, time_idx] += src_seg
                count[time_idx] += 1

                del m_i, mu_i, src_seg

            nz_idx = count > 0
            source_ts = np.zeros_like(source_sum)
            source_ts[:, nz_idx] = source_sum[:, nz_idx] / count[nz_idx][None, :]

        # As before, there is no single MU matrix for a sliding-window inverse.
        return source_ts, time_ms, None

    # ------------------------------------------------------------------
    # Existing single-matrix path.
    # ------------------------------------------------------------------
    temp_projector_mat = invc["TT"]

    sensor_data_aligned, orig_n_time = _pad_or_trim_sensor_to_temp_projector(
        sensor_data, temp_projector_mat
    )

    if sensor_data_aligned.ndim == 3:
        yproj = np.empty(
            (sensor_data_aligned.shape[0], temp_projector_mat.shape[1], n_trials),
            dtype=float,
        )
        for trial_idx in range(n_trials):
            yproj[:, :, trial_idx] = sensor_data_aligned[:, :, trial_idx] @ temp_projector_mat
    else:
        yproj = sensor_data_aligned @ temp_projector_mat

    if mu_matrix is not None:
        if vertices is not None:
            mu_matrix = mu_matrix[vertices, :]
    else:
        m_matrix = _ensure_m_orientation(invc["M"], n_spatial_modes)
        if vertices is not None:
            m_matrix = m_matrix[vertices, :]
        mu_matrix = m_matrix @ u_matrix

    if yproj.ndim == 3:
        n_sources = mu_matrix.shape[0]
        source_ts = np.zeros((n_sources, orig_n_time, n_trials), dtype=float)

        for trial_idx in range(n_trials):
            trial_ts = np.asarray(mu_matrix @ yproj[:, :, trial_idx])
            source_ts[:, :, trial_idx] = _restore_time_length(trial_ts, orig_n_time)
    else:
        source_ts = np.asarray(mu_matrix @ yproj)
        source_ts = _restore_time_length(source_ts, orig_n_time)

    return source_ts, time_ms, mu_matrix

def _load_inverse_components(inv_fname, inversion_idx=0):
    """Load inverse metadata without eagerly loading window matrices."""
    try:
        return _load_hdf5_inverse_components(
            inv_fname,
            inversion_idx,
        )
    except OSError:
        return _load_mat_inverse_components(
            inv_fname,
            inversion_idx,
        )


def _load_hdf5_inverse_components(inv_fname, inversion_idx):
    """Load inverse metadata from a MATLAB v7.3/HDF5 file."""
    with h5py.File(inv_fname, "r") as file:
        inv_root = file[
            file["D"]["other"]["inv"][inversion_idx][0]
        ]["inverse"]

        u_ref = inv_root["U"][0][0]
        u_obj = file[u_ref]
        if isinstance(u_obj, h5py.Group):
            u_matrix = _h5_to_csc(u_obj)
        else:
            u_matrix = np.asarray(u_obj)

        temporal_projector = inv_root["T"][()].T
        temp_projector_mat = (
            temporal_projector
            @ temporal_projector.T
        )
        out = {
            "U": u_matrix,
            "TT": temp_projector_mat,
        }

        if "M_win_files" in inv_root:
            out["M_win_files"] = _h5_read_cellstr(
                file,
                inv_root["M_win_files"],
            )
            out["M_win_variable"] = (
                _h5_read_string(
                    file,
                    inv_root["M_win_variable"],
                )
                if "M_win_variable" in inv_root
                else "M_window"
            )
            out["M_win_storage"] = (
                _h5_read_string(
                    file,
                    inv_root["M_win_storage"],
                )
                if "M_win_storage" in inv_root
                else "external_per_window_v7.3"
            )
            if "woi" not in inv_root:
                raise ValueError(
                    "Found `M_win_files` but no `woi` field "
                    "in the inverse."
                )
            out["woi"] = np.asarray(
                inv_root["woi"][()]
            ).T

        elif "M_win" in inv_root:
            mwin_refs = np.asarray(
                inv_root["M_win"][()]
            ).squeeze()
            if mwin_refs.ndim == 0:
                mwin_refs = np.array([mwin_refs])

            out["M_win_hdf5_paths"] = [
                _h5_ref(file, ref).name
                for ref in np.asarray(
                    mwin_refs
                ).ravel(order="F")
            ]

            if "woi" not in inv_root:
                raise ValueError(
                    "Found `M_win` but no `woi` field "
                    "in the inverse."
                )
            out["woi"] = np.asarray(
                inv_root["woi"][()]
            ).T

        else:
            out["M"] = _h5_matrix_from_object(
                inv_root["M"]
            )

        return out


def _load_mat_inverse_components(inv_fname, inversion_idx):
    """Load inverse metadata from a pre-v7.3 MATLAB file."""
    mat = loadmat(
        inv_fname,
        simplify_cells=True,
    )
    inv = mat["D"]["other"]["inv"][
        inversion_idx
    ]["inverse"]

    u_raw = (
        inv["U"][0]
        if isinstance(
            inv["U"],
            (list, tuple, np.ndarray),
        )
        else inv["U"]
    )
    u_matrix = (
        u_raw
        if issparse(u_raw)
        else csc_matrix(u_raw)
    )

    temporal_projector = inv["T"].T
    temp_projector_mat = (
        temporal_projector
        @ temporal_projector.T
    )
    out = {
        "U": u_matrix,
        "TT": temp_projector_mat,
    }

    if (
        "M_win_files" in inv
        and inv["M_win_files"] is not None
    ):
        out["M_win_files"] = _normalize_string_list(
            inv["M_win_files"]
        )
        out["M_win_variable"] = _normalize_scalar_string(
            inv.get("M_win_variable"),
            default="M_window",
        )
        out["M_win_storage"] = _normalize_scalar_string(
            inv.get("M_win_storage"),
            default="external_per_window_v7.3",
        )
        if "woi" not in inv or inv["woi"] is None:
            raise ValueError(
                "Found `M_win_files` but no `woi` field "
                "in the inverse."
            )
        out["woi"] = np.asarray(inv["woi"]).T

    elif (
        "M_win" in inv
        and inv["M_win"] is not None
    ):
        m_win_raw = inv["M_win"]
        if not isinstance(
            m_win_raw,
            (list, tuple, np.ndarray),
        ):
            m_win_raw = [m_win_raw]

        out["M_win"] = [
            matrix
            if issparse(matrix)
            else csc_matrix(matrix)
            for matrix in list(m_win_raw)
        ]

        if "woi" not in inv or inv["woi"] is None:
            raise ValueError(
                "Found `M_win` but no `woi` field "
                "in the inverse."
            )
        out["woi"] = np.asarray(inv["woi"]).T

    else:
        out["M"] = _mat_to_csc(inv["M"])

    return out


def _load_external_window_full(
    index,
    *,
    files,
    variable,
    n_spatial_modes,
):
    """Load one external window matrix in source x mode orientation."""
    return _ensure_m_orientation(
        _load_h5_matrix(
            files[index],
            variable,
        ),
        n_spatial_modes,
    )


def _load_external_window_rows(
    index,
    rows,
    *,
    files,
    variable,
    n_spatial_modes,
):
    """Load selected rows from one external window matrix."""
    return _load_h5_matrix_rows(
        files[index],
        variable,
        rows,
        n_spatial_modes,
    )


def _load_embedded_h5_window_full(
    index,
    *,
    inv_fname,
    object_paths,
    n_spatial_modes,
):
    """Load one embedded v7.3 window matrix."""
    return _ensure_m_orientation(
        _load_h5_matrix_by_path(
            inv_fname,
            object_paths[index],
        ),
        n_spatial_modes,
    )


def _load_embedded_h5_window_rows(
    index,
    rows,
    *,
    inv_fname,
    object_paths,
    n_spatial_modes,
):
    """Load selected rows from one embedded v7.3 window matrix."""
    return _load_h5_matrix_rows_by_path(
        inv_fname,
        object_paths[index],
        rows,
        n_spatial_modes,
    )


def _load_legacy_window_full(
    index,
    *,
    matrices,
    n_spatial_modes,
):
    """Load one legacy in-memory window matrix."""
    return _ensure_m_orientation(
        matrices[index],
        n_spatial_modes,
    )


def _load_legacy_window_rows(
    index,
    rows,
    *,
    matrices,
    n_spatial_modes,
):
    """Load selected rows from one legacy in-memory window matrix."""
    matrix = _load_legacy_window_full(
        index,
        matrices=matrices,
        n_spatial_modes=n_spatial_modes,
    )
    return matrix[rows, :]


def _make_window_matrix_loaders(
    invc,
    inv_fname,
    data_fname,
):
    """Create lazy full/row loaders for a sliding-window inverse."""
    n_spatial_modes = invc["U"].shape[0]

    if "M_win_files" in invc:
        stored_files = invc["M_win_files"]
        m_variable = invc.get(
            "M_win_variable",
            "M_window",
        )
        resolved_files = [
            _resolve_m_win_file(
                fname,
                inv_fname=inv_fname,
                data_fname=data_fname,
            )
            for fname in stored_files
        ]

        load_full = partial(
            _load_external_window_full,
            files=resolved_files,
            variable=m_variable,
            n_spatial_modes=n_spatial_modes,
        )
        load_rows = partial(
            _load_external_window_rows,
            files=resolved_files,
            variable=m_variable,
            n_spatial_modes=n_spatial_modes,
        )
        n_sources, _ = _h5_matrix_shape(
            resolved_files[0],
            m_variable,
            n_spatial_modes,
        )

        return {
            "n_windows": len(resolved_files),
            "n_sources": n_sources,
            "load_full": load_full,
            "load_rows": load_rows,
        }

    if "M_win_hdf5_paths" in invc:
        object_paths = invc["M_win_hdf5_paths"]

        load_full = partial(
            _load_embedded_h5_window_full,
            inv_fname=inv_fname,
            object_paths=object_paths,
            n_spatial_modes=n_spatial_modes,
        )
        load_rows = partial(
            _load_embedded_h5_window_rows,
            inv_fname=inv_fname,
            object_paths=object_paths,
            n_spatial_modes=n_spatial_modes,
        )
        n_sources, _ = _h5_matrix_shape_by_path(
            inv_fname,
            object_paths[0],
            n_spatial_modes,
        )

        return {
            "n_windows": len(object_paths),
            "n_sources": n_sources,
            "load_full": load_full,
            "load_rows": load_rows,
        }

    matrices = invc["M_win"]
    load_full = partial(
        _load_legacy_window_full,
        matrices=matrices,
        n_spatial_modes=n_spatial_modes,
    )
    load_rows = partial(
        _load_legacy_window_rows,
        matrices=matrices,
        n_spatial_modes=n_spatial_modes,
    )
    first_matrix = load_full(0)

    return {
        "n_windows": len(matrices),
        "n_sources": first_matrix.shape[0],
        "load_full": load_full,
        "load_rows": load_rows,
    }


def _load_window_matrix(loaders, index, rows=None):
    """Load a whole window matrix or a selected row subset."""
    if rows is None:
        return loaders["load_full"](index)
    return loaders["load_rows"](index, rows)



def _h5_read_cellstr(file, dataset):
    """Read a MATLAB v7.3 cell array of character vectors."""
    refs = np.asarray(dataset[()]).ravel(order="F")
    values = []
    for ref in refs:
        if isinstance(ref, h5py.Reference):
            if bool(ref):
                values.append(_h5_read_string(file, ref))
        else:
            values.append(_decode_matlab_char_array(ref))
    return values

def _h5_read_string(file, obj):
    """Read a MATLAB v7.3 char value, including a cell/reference wrapper."""
    if isinstance(obj, h5py.Reference):
        obj = file[obj]

    if isinstance(obj, h5py.Dataset):
        ref_dtype = h5py.check_dtype(ref=obj.dtype)
        if ref_dtype is not None:
            refs = np.asarray(obj[()]).ravel(order="F")
            refs = [ref for ref in refs if bool(ref)]
            if not refs:
                return ""
            return _h5_read_string(file, refs[0])
        return _decode_matlab_char_array(obj[()])

    return _decode_matlab_char_array(obj)

def _normalize_scalar_string(value, default=""):
    values = _normalize_string_list(value)
    return values[0] if values else default

def _normalize_string_list(value):
    """Normalize scipy.io.loadmat cell/string output to list[str]."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="replace")]

    arr = np.asarray(value, dtype=object)
    values = []
    for item in arr.ravel(order="F"):
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()

        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, bytes):
            values.append(item.decode("utf-8", errors="replace"))
        elif isinstance(item, np.ndarray):
            values.append(_decode_matlab_char_array(item))
        else:
            values.append(str(item))

    return values

def _decode_matlab_char_array(value):
    """Decode a MATLAB char array read through h5py or scipy.io.loadmat."""
    arr = np.asarray(value)
    result = str(value)

    if arr.size == 0:
        result = ""

    elif arr.dtype.kind == "U":
        result = "".join(
            arr.ravel(order="F").tolist()
        ).rstrip("\x00")

    elif arr.dtype.kind == "S":
        result = b"".join(
            arr.ravel(order="F").tolist()
        ).decode(
            "utf-8",
            errors="replace",
        ).rstrip("\x00")

    elif np.issubdtype(
        arr.dtype,
        np.integer,
    ):
        codes = arr.astype(
            np.uint32,
            copy=False,
        ).ravel(order="F")
        result = "".join(
            chr(int(code))
            for code in codes
            if int(code) != 0
        )

    elif arr.size == 1:
        item = arr.item()
        result = (
            item.decode(
                "utf-8",
                errors="replace",
            )
            if isinstance(item, bytes)
            else str(item)
        )

    return result

def _load_h5_matrix_rows(file_name, object_name, rows, n_spatial_modes):
    """Read selected logical source rows from one v7.3 MAT/HDF5 matrix."""
    with h5py.File(file_name, "r") as file:
        if object_name not in file:
            raise KeyError(f"{object_name!r} not found in {file_name!r}.")
        return _h5_matrix_rows_from_object(file[object_name], rows, n_spatial_modes)

def _load_h5_matrix_rows_by_path(file_name, object_path, rows, n_spatial_modes):
    """Read selected logical source rows from an internal HDF5 object path."""
    with h5py.File(file_name, "r") as file:
        return _h5_matrix_rows_from_object(file[object_path], rows, n_spatial_modes)

def _h5_matrix_rows_from_object(obj, rows, n_spatial_modes):
    """Read selected logical source rows from a MATLAB v7.3 matrix object.

    Dense MATLAB arrays are commonly exposed by h5py with transposed dimensions.
    This helper detects that orientation from ``n_spatial_modes`` and performs
    the row selection before materializing the matrix in Python.
    """
    rows = np.asarray(rows, dtype=np.int64).ravel()
    if rows.size == 0:
        return np.empty((0, n_spatial_modes), dtype=float)
    if np.any(rows < 0):
        raise IndexError("Source row indices must be non-negative.")

    if isinstance(obj, h5py.Group) and all(k in obj for k in ("data", "ir", "jc")):
        matrix = _ensure_m_orientation(_h5_to_csc(obj), n_spatial_modes)
        if rows.max() >= matrix.shape[0]:
            raise IndexError("Source row index exceeds inverse matrix dimensions.")
        return matrix[rows, :]

    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Unsupported HDF5 MATLAB matrix object: {type(obj)!r}")

    if obj.ndim != 2:
        raise ValueError(f"Expected a 2-D inverse matrix, got shape {obj.shape}.")

    unique_rows, inverse_idx = np.unique(rows, return_inverse=True)

    if obj.shape[1] == n_spatial_modes:
        if unique_rows[-1] >= obj.shape[0]:
            raise IndexError("Source row index exceeds inverse matrix dimensions.")
        selected = np.asarray(obj[unique_rows, :])

    elif obj.shape[0] == n_spatial_modes:
        if unique_rows[-1] >= obj.shape[1]:
            raise IndexError("Source row index exceeds inverse matrix dimensions.")
        selected = np.asarray(obj[:, unique_rows]).T

    else:
        raise ValueError(
            "Window inverse matrix is incompatible with inverse.U: "
            f"stored shape={obj.shape}, expected one dimension to equal "
            f"{n_spatial_modes} spatial modes."
        )

    return selected[inverse_idx, :]

def _h5_matrix_shape(file_name, object_name, n_spatial_modes):
    """Return logical matrix shape without loading a dense matrix."""
    with h5py.File(file_name, "r") as file:
        if object_name not in file:
            raise KeyError(f"{object_name!r} not found in {file_name!r}.")
        return _h5_matrix_shape_from_object(file[object_name], n_spatial_modes)

def _h5_matrix_shape_by_path(file_name, object_path, n_spatial_modes):
    """Return logical matrix shape for a known internal HDF5 object path."""
    with h5py.File(file_name, "r") as file:
        return _h5_matrix_shape_from_object(file[object_path], n_spatial_modes)

def _h5_matrix_shape_from_object(obj, n_spatial_modes):
    """Return logical ``(n_sources, n_spatial_modes)`` for a v7.3 matrix."""
    if isinstance(obj, h5py.Group) and all(k in obj for k in ("data", "ir", "jc")):
        matrix = _ensure_m_orientation(_h5_to_csc(obj), n_spatial_modes)
        return matrix.shape

    if not isinstance(obj, h5py.Dataset) or obj.ndim != 2:
        raise TypeError(f"Unsupported HDF5 MATLAB matrix object: {type(obj)!r}")

    if obj.shape[1] == n_spatial_modes:
        return obj.shape
    if obj.shape[0] == n_spatial_modes:
        return (obj.shape[1], obj.shape[0])

    raise ValueError(
        "Window inverse matrix is incompatible with inverse.U: "
        f"stored shape={obj.shape}, expected one dimension to equal "
        f"{n_spatial_modes} spatial modes."
    )

def _ensure_m_orientation(m_matrix, n_spatial_modes):
    """Ensure M is sources x spatial_modes.

    MATLAB v7.3 dense arrays can appear transposed when accessed directly via
    h5py. Sparse MATLAB matrices retain their logical dimensions. This check
    makes the loader robust to either representation.
    """
    if m_matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D inverse matrix, got shape {m_matrix.shape}.")

    if m_matrix.shape[1] == n_spatial_modes:
        return m_matrix

    if m_matrix.shape[0] == n_spatial_modes:
        return m_matrix.T.tocsc() if issparse(m_matrix) else m_matrix.T

    raise ValueError(
        "Window inverse matrix is incompatible with inverse.U: "
        f"M shape={m_matrix.shape}, expected one dimension to equal "
        f"{n_spatial_modes} spatial modes."
    )

def _load_h5_matrix(file_name, object_name):
    """Load one dense/sparse matrix from a MATLAB v7.3 HDF5 file."""
    with h5py.File(file_name, "r") as file:
        if object_name not in file:
            raise KeyError(f"{object_name!r} not found in {file_name!r}.")
        return _h5_matrix_from_object(file[object_name])

def _load_h5_matrix_by_path(file_name, object_path):
    """Load one matrix from a known internal HDF5 object path."""
    with h5py.File(file_name, "r") as file:
        return _h5_matrix_from_object(file[object_path])

def _resolve_m_win_file(stored_path, inv_fname, data_fname):
    """Resolve a stored M-window path relative to inverse/data locations."""
    stored_path = os.fspath(stored_path).strip()
    if not stored_path:
        raise ValueError("Encountered an empty entry in inverse.M_win_files.")

    # Files are normally generated on POSIX, but normalizing backslashes makes
    # copied inversions a little more portable.
    stored_path = stored_path.replace("\\", os.sep)

    if os.path.isabs(stored_path):
        candidates = [stored_path]
    else:
        candidates = []
        for base_file in (inv_fname, data_fname):
            if base_file is None:
                continue
            candidate = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(base_file)), stored_path)
            )
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate external sliding-window inverse matrix. "
        f"Stored path: {stored_path!r}; tried: {candidates!r}."
    )

def _h5_matrix_from_object(obj):
    """Convert a MATLAB v7.3 dense/sparse matrix object to NumPy/SciPy."""
    if isinstance(obj, h5py.Group) and all(k in obj for k in ("data", "ir", "jc")):
        return _h5_to_csc(obj)
    if isinstance(obj, h5py.Dataset):
        return np.asarray(obj[()])
    raise TypeError(f"Unsupported HDF5 MATLAB matrix object: {type(obj)!r}")

def _h5_ref(file, ref):
    return file[ref]

def _pad_or_trim_sensor_to_temp_projector(sensor_data, temp_projector_mat):
    """Pad/trim sensor_data time dimension to match TT if off by one sample."""
    n_time = sensor_data.shape[1]
    if temp_projector_mat.shape[0] == n_time:
        return sensor_data, n_time

    diff = temp_projector_mat.shape[0] - n_time
    if abs(diff) != 1:
        raise ValueError(
            f"Temporal projector ({temp_projector_mat.shape}) and sensor data "
            f"({sensor_data.shape}) differ by >1 sample."
        )

    if sensor_data.ndim == 2:
        if diff > 0:
            sensor_data = np.pad(sensor_data, ((0, 0), (0, diff)), mode="constant")
        else:
            sensor_data = sensor_data[:, :temp_projector_mat.shape[0]]
    else:
        if diff > 0:
            sensor_data = np.pad(sensor_data, ((0, 0), (0, diff), (0, 0)), mode="constant")
        else:
            sensor_data = sensor_data[:, :temp_projector_mat.shape[0], :]

    return sensor_data, n_time

def _restore_time_length(time_series, orig_n_time):
    """Ensure time_series has exactly orig_n_time along axis 1."""
    cur = time_series.shape[1]
    if cur == orig_n_time:
        return time_series
    if cur < orig_n_time:
        return np.pad(time_series, ((0, 0), (0, orig_n_time - cur)), mode="constant")
    return time_series[:, :orig_n_time]

def _indices_for_woi(time_ms, woi_row):
    """Indices where time is within [start, end] (inclusive)."""
    time_0, time_1 = float(woi_row[0]), float(woi_row[1])
    lo_idx, hi_idx = (time_0, time_1) if time_0 <= time_1 else (time_1, time_0)
    return np.where((time_ms >= lo_idx) & (time_ms <= hi_idx))[0]
