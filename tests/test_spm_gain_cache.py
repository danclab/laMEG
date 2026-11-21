"""Exercise the preflight SPM uses before selecting spatial modes."""
import hashlib

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from lameg._spm_gain import check_spm_gain_cache


def write_header(path, external, version, active=2, bad_extra=True):
    """Set up two inversion slots, with a MATLAB-empty first cell."""
    labels = ['A', 'B', 'C', 'EOG', 'X']
    kinds = ['MEGMAG', 'MEGGRAD', 'MEGPLANAR', 'EOG', 'MEG']
    flags = [0, 0, 0, 0, int(bad_extra)]
    if version == 'hdf5':
        with h5py.File(path, 'w') as handle:
            other = handle.create_group('D/other')
            other.create_dataset('val', data=[[active]])
            slots = other.create_dataset('inv', (2, 1), dtype=h5py.ref_dtype)
            empty = handle.create_dataset('empty', data=np.zeros((2, 1), dtype=np.uint64))
            empty.attrs['MATLAB_empty'] = 1
            native = handle.create_group('slot2')
            def text(value):
                return np.frombuffer(value.encode('utf-16-le'), dtype='<u2')

            channels = handle.create_group('D/channels')
            for key, content in [('label', labels), ('type', kinds), ('bad', flags)]:
                refs = channels.create_dataset(key, (len(labels), 1), dtype=h5py.ref_dtype)
                for index, item in enumerate(content):
                    value = text(item) if isinstance(item, str) else [[item]]
                    refs[index, 0] = handle.create_dataset(f'{key}{index}', data=value).ref
            if external:
                native.create_dataset('gainmat', data=text(external['file']))
                marker = native.create_group('lameg_external_gain')
                for key, value in external.items():
                    marker.create_dataset(key, data=text(value))
            slots[:, 0] = [empty.ref, native.ref]
    else:
        slots = np.empty((1, 2), dtype=object)
        slots[0, 0] = np.empty((0, 0))
        slots[0, 1] = ({'lameg_external_gain': external, 'gainmat': external['file']}
                       if external else {'native': 1})
        channels = np.empty(len(labels), dtype=[('label','O'), ('type','O'), ('bad','O')])
        channels['label'], channels['type'], channels['bad'] = labels, kinds, flags
        savemat(path, {'D': {'channels': channels, 'other': {'inv': slots, 'val': active}}})


@pytest.mark.parametrize('version', ['hdf5', 'legacy'])
def test_marked_cache_enforces_slot_and_exact_bytes(tmp_path, version):
    """A missing or damaged external operator must fail before native SPM fallback."""
    header = tmp_path / 'header.mat'
    cache = tmp_path / 'lameg_gain_fixture.mat'
    savemat(cache, {'G': np.eye(3), 'label': np.array(['A', 'B', 'C'], dtype=object)})
    checksum = hashlib.sha256(cache.read_bytes()).hexdigest()
    write_header(header, {'file': cache.name, 'sha256': checksum}, version)
    check_spm_gain_cache(header)
    check_spm_gain_cache(header, inversion_idx=1)
    with pytest.raises(ValueError, match='active SPM slot'):
        check_spm_gain_cache(header, inversion_idx=0)
    cache.write_bytes(cache.read_bytes()[:-10])
    with pytest.raises(ValueError, match='has changed'):
        check_spm_gain_cache(header)
    cache.unlink()
    with pytest.raises(FileNotFoundError, match='is missing'):
        check_spm_gain_cache(header)


@pytest.mark.parametrize('version', ['hdf5', 'legacy'])
def test_native_and_empty_slots_do_not_require_a_cache(tmp_path, version):
    """Native SPM may compute a fresh gain when no cache has been selected."""
    header = tmp_path / 'native.mat'
    write_header(header, None, version, active=1)
    check_spm_gain_cache(header)
    check_spm_gain_cache(header, inversion_idx=1)


@pytest.mark.parametrize('version', ['hdf5', 'legacy'])
def test_new_good_meg_channel_cannot_trigger_native_fallback(tmp_path, version):
    """Marking a previously excluded MEG channel good requires a gain for that channel."""
    header = tmp_path / 'header.mat'
    cache = tmp_path / 'lameg_gain_fixture.mat'
    savemat(cache, {'G': np.eye(3), 'label': np.array(['A', 'B', 'C'], dtype=object)})
    marker = {'file': cache.name, 'sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}
    write_header(header, marker, version, bad_extra=True)
    check_spm_gain_cache(header)  # EOG is good, but never enters the MEG forward.
    write_header(header, marker, version, bad_extra=False)
    with pytest.raises(ValueError, match="Good MEG channels missing.*'X'"):
        check_spm_gain_cache(header)
