"""The selected SPM cache determines the pial–white RMS contrast."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from lameg.invert import get_lead_field_rms_diff


SURFACES = SimpleNamespace(n_layers=2, get_vertices_per_layer=lambda: 2)


def make_header(path, names, hdf5):
    """Create a header with the specified gain filename in each inversion slot."""
    if hdf5:
        with h5py.File(path, 'w') as file:
            refs = file.create_dataset('D/other/inv', (len(names), 1), dtype=h5py.ref_dtype)
            for i, name in enumerate(names):
                slot = file.create_group(f'slot_{i + 1}')
                refs[i, 0] = slot.ref
                if name:
                    slot['gainmat'] = np.asarray([ord(c) for c in name], dtype='uint16')[:, None]
    else:
        cells = np.empty((1, len(names)), dtype=object)
        for i, name in enumerate(names):
            cells[0, i] = {'forward': 1, **({'gainmat': name} if name else {})}
        savemat(path, {'D': {'other': {'inv': cells}}})


@pytest.mark.parametrize('hdf5_header', [False, True])
@pytest.mark.parametrize('hdf5_cache', [False, True])
def test_rms_uses_recorded_cache_and_selected_slot(tmp_path, hdf5_header, hdf5_cache):
    """Resolve the selected inversion across both header and cache formats."""
    header = tmp_path / 'subject.mat'
    make_header(header, ['first.mat', 'selected.mat'], hdf5_header)
    # The old conventional gain remains on disk beside the explicitly selected one.
    savemat(tmp_path / 'SPMgainmatrix_subject_2.mat', {'G': np.zeros((2, 4))})
    gain = np.array([[1., 4., 4., 0.], [2., 4., 6., 4.]])
    if hdf5_cache:
        with h5py.File(tmp_path / 'selected.mat', 'w') as cache:
            cache['G'] = gain.T
    else:
        savemat(tmp_path / 'selected.mat', {'G': gain})
    result = get_lead_field_rms_diff(header, SURFACES, inversion_idx=1)
    np.testing.assert_allclose(result, np.sqrt([12.5, 8.]))


def test_rms_missing_recorded_cache_never_uses_conventional_gain(tmp_path):
    """An explicitly selected but missing cache must produce an error."""
    header = tmp_path / 'subject.mat'
    make_header(header, ['missing.mat'], True)
    savemat(tmp_path / 'SPMgainmatrix_subject_1.mat', {'G': np.zeros((2, 4))})
    with pytest.raises(FileNotFoundError, match='missing.mat'):
        get_lead_field_rms_diff(header, SURFACES)


@pytest.mark.parametrize('hdf5_header', [False, True])
def test_rms_legacy_fallback_checks_entire_layer_shape(tmp_path, hdf5_header):
    """Support conventional legacy names and reject additional source columns."""
    header = tmp_path / 'subject.mat'
    make_header(header, [None], hdf5_header)
    cache = tmp_path / 'SPMgainmatrix_subject_1.mat'
    savemat(cache, {'G': np.arange(8).reshape(2, 4)})
    np.testing.assert_allclose(get_lead_field_rms_diff(header, SURFACES), [2., 2.])
    # Extra columns used to be ignored, silently misidentifying the last layer.
    savemat(cache, {'G': np.arange(12).reshape(2, 6)})
    with pytest.raises(ValueError, match='layer and vertex counts'):
        get_lead_field_rms_diff(header, SURFACES)
