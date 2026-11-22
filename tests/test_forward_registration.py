"""Registration for preparing MNE gains without computing a native gain cache."""
import numpy as np
import pytest
from mne.transforms import Transform
from scipy.io import savemat

from lameg._forward_registration import mne_registration


def test_registered_header_supplies_montage_units_and_frame(tmp_path):
    """Use the requested SPM slot, active montage and explicitly identified CTF head frame."""
    # Only the second SPM slot contains a registration. Its 24 mm MRI translation
    # makes accidental use of an identity affine observable in the test.
    affine = np.eye(4)
    affine[2, 3] = 24.
    channel = {'label': 'M1', 'type': 'MEG', 'units': 'fT', 'bad': 0}
    registered = {'datareg': {'modality': 'MEG', 'fromMNI': np.eye(4)}, 'mesh': {'Affine': affine},
                  'forward': {'modality': 'MEG', 'sensors': {'coordsys': 'ctf'}}}
    path = tmp_path / 'recording.mat'
    # Use different units for the stored channel and the active montage so that the
    # expected fT gain row can only be obtained from the montage.
    savemat(path, {'D': {'channels': dict(channel, units='T'), 'other': {'inv': [{}, registered]},
                         'montage': {'Mind': 1, 'M': {'channels': channel}}}})
    # A +0.021 m CTF-to-MNE translation must become -21 mm when carrying MNE
    # points into the head coordinates of the recipient SPM recording.
    ctf_to_mne = np.eye(4)
    ctf_to_mne[0, 3] = 0.021
    info = {'ctf_head_t': Transform('ctf_head', 'head', ctf_to_mne)}
    units, used, head = mne_registration(path, info, None, None, None, inversion_idx=1)
    assert units == {'M1': 'fT'}
    np.testing.assert_array_equal(used, affine)
    np.testing.assert_allclose(head[:3, 3], [-21., 0, 0])
    # These three failures distinguish an unregistered slot, a gain scaling conflict,
    # and missing acquisition information for the head-to-head transform.
    with pytest.raises(ValueError, match='Coregister'):
        mne_registration(path, info, None, None, None)
    with pytest.raises(ValueError, match='units'):
        mne_registration(path, info, affine, head, {'M1': 'T'}, inversion_idx=1)
    with pytest.raises(ValueError, match='mne_head_to_head_mm'):
        mne_registration(path, None, None, None, None, inversion_idx=1)
