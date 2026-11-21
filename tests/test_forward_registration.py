"""Registration for preparing MNE gains without computing a native gain cache."""
import numpy as np
import pytest
from mne.transforms import Transform
from scipy.io import savemat

from lameg._forward_registration import mne_registration


def test_registered_header_supplies_montage_units_and_frame(tmp_path):
    """Use the requested SPM slot, active montage and explicitly identified CTF head frame."""
    affine = np.eye(4)
    affine[2, 3] = 24.
    channel = {'label': 'M1', 'type': 'MEG', 'units': 'fT', 'bad': 0}
    registered = {'datareg': {'modality': 'MEG', 'fromMNI': np.eye(4)}, 'mesh': {'Affine': affine},
                  'forward': {'modality': 'MEG', 'sensors': {'coordsys': 'ctf'}}}
    path = tmp_path / 'recording.mat'
    savemat(path, {'D': {'channels': dict(channel, units='T'), 'other': {'inv': [{}, registered]},
                         'montage': {'Mind': 1, 'M': {'channels': channel}}}})
    ctf_to_mne = np.eye(4)
    ctf_to_mne[0, 3] = 0.021
    info = {'ctf_head_t': Transform('ctf_head', 'head', ctf_to_mne)}
    units, used, head = mne_registration(path, info, None, None, None, inversion_idx=1)
    assert units == {'M1': 'fT'}
    np.testing.assert_array_equal(used, affine)
    np.testing.assert_allclose(head[:3, 3], [-21., 0, 0])
    with pytest.raises(ValueError, match='Coregister'):
        mne_registration(path, info, None, None, None)
    with pytest.raises(ValueError, match='units'):
        mne_registration(path, info, affine, head, {'M1': 'T'}, inversion_idx=1)
    with pytest.raises(ValueError, match='mne_head_to_head_mm'):
        mne_registration(path, None, None, None, None, inversion_idx=1)
