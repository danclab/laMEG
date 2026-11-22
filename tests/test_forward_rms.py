"""The selected SPM cache determines the pial–white RMS contrast."""
import hashlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from lameg.invert import get_lead_field_rms_diff


SURFACES = SimpleNamespace(n_layers=2, get_vertices_per_layer=lambda: 2)


def make_header(path, names, hdf5, marker=None):
    """Create a header with the specified gain filename in each inversion slot."""
    def chars(text):
        return np.asarray([ord(c) for c in text], dtype='uint16')[:, None]

    if hdf5:
        with h5py.File(path, 'w') as file:
            refs = file.create_dataset('D/other/inv', (len(names), 1), dtype=h5py.ref_dtype)
            for i, name in enumerate(names):
                slot = file.create_group(f'slot_{i + 1}')
                refs[i, 0] = slot.ref
                if name:
                    slot['gainmat'] = chars(name)
            if marker is not None:
                external = slot.create_group('lameg_external_gain')
                for key, value in marker.items():
                    external[key] = chars(value) if isinstance(value, str) else value
    else:
        cells = np.empty((1, len(names)), dtype=object)
        for i, name in enumerate(names):
            cells[0, i] = {'forward': 1, **({'gainmat': name} if name else {})}
        if marker is not None:
            cells[0, -1]['lameg_external_gain'] = marker
        savemat(path, {'D': {'other': {'inv': cells}}})


@pytest.mark.parametrize('hdf5_header', [False, True])
@pytest.mark.parametrize('hdf5_cache', [False, True])
def test_rms_uses_recorded_cache_and_selected_slot(tmp_path, hdf5_header, hdf5_cache):
    """Resolve the selected inversion across both header and cache formats."""
    header = tmp_path / 'subject.mat'
    make_header(header, ['first.mat', 'selected.mat'], hdf5_header)
    # The old conventional gain remains on disk beside the explicitly selected one.
    savemat(tmp_path / 'SPMgainmatrix_subject_2.mat', {'G': np.zeros((2, 4))})
    # Each row is a sensor. The first two columns are pial vertices and the next two
    # are corresponding white vertices. White minus pial gives [3, 4] and [-4, 0]
    # across sensors, with mean squares 12.5 and 8 for the two vertices.
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


@pytest.mark.parametrize('hdf5_header', [False, True])
def test_rms_requires_selected_external_gain_bytes(tmp_path, hdf5_header):
    """An explicit, inactive slot needs its registered gain and no active-channel metadata."""
    header = tmp_path / 'subject.mat'
    cache = tmp_path / 'external.mat'
    gain = np.arange(8).reshape(2, 4)
    savemat(cache, {'G': gain})
    # The saved digest ties this diagnostic to the gain actually attached for inversion.
    # A polarity reversal later in the test leaves the RMS contrast itself unchanged.
    marker = {'file': cache.name, 'format_version': 1,
              'sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}
    make_header(header, ['first.mat', cache.name], hdf5_header, marker)
    np.testing.assert_allclose(get_lead_field_rms_diff(header, SURFACES, 1), [2., 2.])
    make_header(header, ['first.mat', cache.name], hdf5_header, dict(marker, format_version=2))
    with pytest.raises(ValueError, match='Invalid external-gain registration'):
        get_lead_field_rms_diff(header, SURFACES, 1)
    make_header(header, ['first.mat', cache.name], hdf5_header, marker)
    # A readable conventional cache cannot substitute for an altered or missing registered gain.
    savemat(tmp_path / 'SPMgainmatrix_subject_2.mat', {'G': gain})
    savemat(cache, {'G': -gain})
    with pytest.raises(ValueError, match='checksum'):
        get_lead_field_rms_diff(header, SURFACES, 1)
    cache.unlink()
    with pytest.raises(FileNotFoundError, match='external.mat'):
        get_lead_field_rms_diff(header, SURFACES, 1)
    make_header(header, ['first.mat', 'SPMgainmatrix_subject_2.mat'], hdf5_header, marker)
    with pytest.raises(ValueError, match='not selected'):
        get_lead_field_rms_diff(header, SURFACES, 1)


@pytest.mark.parametrize('hdf5_header', [False, True])
def test_rms_legacy_relative_cache_path(tmp_path, hdf5_header):
    """Preserve a native SPM cache's relative subdirectory after moving a dataset."""
    header = tmp_path / 'subject.mat'
    folder = tmp_path / 'gains'
    folder.mkdir()
    savemat(folder / 'nested.mat', {'G': np.arange(8).reshape(2, 4)})
    # This Windows-style relative path models a dataset copied between machines. The
    # lookup should still resolve the gain directory relative to the new header.
    make_header(header, [r'gains\nested.mat'], hdf5_header)
    np.testing.assert_allclose(get_lead_field_rms_diff(header, SURFACES), [2., 2.])
