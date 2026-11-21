"""Read and verify the gain cache selected on an SPM inversion slot."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import PureWindowsPath

import h5py
import numpy as np
from scipy.io import loadmat


def validate_slot_index(index):
    """Require a zero-based explicit inversion index."""
    if (not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
            or index < 0):
        raise ValueError("inversion_idx must be a nonnegative integer.")


@contextmanager
def open_spm_header(path):
    """Yield an HDF5 header or the SciPy representation of a pre-v7.3 D struct."""
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as header:
            yield header
    else:
        yield loadmat(path, variable_names=['D'], struct_as_record=False, squeeze_me=True)['D']


def active_slot_index(header):
    """Read SPM's one-based D.val as a zero-based inversion index."""
    if isinstance(header, h5py.File):
        return int(np.asarray(header.get('D/other/val', 1)).item()) - 1
    return int(getattr(getattr(header, 'other', None), 'val', 1)) - 1


def inversion_slot(header, index):
    """Return the requested slot struct, or None for an absent or MATLAB-empty cell."""
    if isinstance(header, h5py.File):
        cells = header.get('D/other/inv')
        if (not isinstance(cells, h5py.Dataset) or h5py.check_dtype(ref=cells.dtype) is None
                or not 0 <= index < cells.size):
            return None
        ref = cells[()].ravel()[index]
        slot = header[ref] if ref else None
        return slot if isinstance(slot, h5py.Group) else None
    cells = np.atleast_1d(getattr(getattr(header, 'other', None), 'inv', [])).ravel()
    if 0 <= index < cells.size and hasattr(cells[index], '_fieldnames'):
        return cells[index]
    return None


def mat_string(dataset):
    """Decode one HDF5 MATLAB character vector."""
    chars = dataset[()]
    if chars.dtype.kind not in 'ui':
        raise ValueError("Expected a single MATLAB character vector in the SPM header.")
    return ''.join(chr(c) for c in chars.ravel()).rstrip('\x00 ')


def cell_strings(cells):
    """Decode a cell array of character vectors as loaded by scipy.io.loadmat."""
    return tuple(str(np.asarray(value).item()) for value in cells.ravel())


def _field(struct, key):
    return struct.get(key) if isinstance(struct, h5py.Group) else getattr(struct, key, None)


def _text_field(struct, key):
    value = _field(struct, key)
    if isinstance(value, h5py.Dataset):
        return '' if value.attrs.get('MATLAB_empty', False) else mat_string(value)
    return '' if value is None or not np.size(value) else str(value)


def gain_reference(slot, external_only=False):
    """Read (gainmat, optional checksum) and validate any external registration."""
    marker = _field(slot, 'lameg_external_gain')
    if marker is None and external_only:
        return None
    name = _text_field(slot, 'gainmat')
    if marker is None:
        return name, None
    version = np.asarray(_field(marker, 'format_version'))
    digest = _text_field(marker, 'sha256')
    if version.size != 1 or version.item() != 1 or not digest:
        raise ValueError("Invalid external-gain registration in the SPM inversion slot.")
    if not name or PureWindowsPath(name).name != name or name != _text_field(marker, 'file'):
        raise ValueError("The registered external gain cache is not selected.")
    return name, digest


def file_identity(path):
    """Identify the file version used for memoized metadata and checksums."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=128)
def cache_hash(path, _file_identity):
    """Hash the exact cache bytes once per file version."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def verified_gain(path, expected_hash):
    """Verify a registered cache before use and detect replacement during the read."""
    if expected_hash is None:
        if not path.is_file():
            raise FileNotFoundError(f"SPM gain cache does not exist: {path}")
        yield None
        return
    if not path.is_file():
        raise FileNotFoundError(f"The registered external gain cache is missing: {path}")
    identity = file_identity(path)
    if cache_hash(path, identity) != expected_hash:
        raise ValueError(f"External gain cache has changed (checksum mismatch): {path}")
    yield identity
    if file_identity(path) != identity:
        raise ValueError(f"The registered external gain cache changed during the read: {path}")
