"""Reference compensation and pickup geometry for CTF-derived MNE gains."""

import mne
from mne.io.constants import FIFF
import numpy as np
import pytest

from lameg._mne_sensors import mne_channel_geometry


def test_compensated_ctf_forward(tmp_path):
    """Keep the signed reference pickup in the physical channel response after FIF reload."""
    info = mne.create_info(['MLC11', 'BREF'], 200., ['mag', 'ref_meg'])
    primary, reference = info['chs']
    # The upper bits encode CTF compensation grade 3. Use one primary and one reference
    # so the expected sensor transfer can be checked by hand.
    primary['coil_type'] = FIFF.FIFFV_COIL_CTF_GRAD | (3 << 16)
    reference['coil_type'] = FIFF.FIFFV_COIL_CTF_REF_MAG
    primary['loc'] = np.r_[0.01, 0.0, 0.10, np.eye(3).ravel()]
    reference['loc'] = np.r_[0.03, 0.01, 0.13, np.eye(3).ravel()]
    # A coefficient of 0.125 subtracts one eighth of the reference signal from the
    # primary. This produces a signed contribution distinct from either primary pickup.
    info['comps'].append({
        'kind': 3, 'rowcals': np.ones(1), 'colcals': np.ones(1),
        'data': {'nrow': 1, 'ncol': 1, 'row_names': ['MLC11'],
                 'col_names': ['BREF'], 'data': np.array([[0.125]])}})
    src = mne.setup_volume_source_space(
        pos={'rr': np.array([[0.01, 0.02, 0.04]]), 'nn': np.array([[1., 0., 0.]])},
        verbose=False)
    sphere = mne.make_sphere_model(r0=(0, 0, 0), head_radius=None, verbose=False)
    fwd = mne.make_forward_solution(info, None, src, sphere, eeg=False, ignore_ref=False,
                                    verbose=False)
    ignored = mne.make_forward_solution(info, None, src, sphere, eeg=False, ignore_ref=True,
                                        verbose=False)
    # Confirm that including the reference changes the actual MNE gain for this fixture.
    assert np.linalg.norm(fwd['sol']['data'] - ignored['sol']['data']) > 0
    fname = tmp_path / 'ctf-fwd.fif'
    mne.write_forward_solution(fname, fwd, verbose=False)
    fwd = mne.read_forward_solution(fname, verbose=False)
    # The FIF round trip retains the primary label and grade. Check that importing still
    # requires enough acquisition metadata to reconstruct the reference response.
    args = (fwd, ('MLC11',), ('fT',), np.eye(4))
    with pytest.raises(ValueError, match='original acquisition Info'):
        mne_channel_geometry(*args, ignore_ref=False)
    with pytest.raises(ValueError, match='ignore_ref=False'):
        mne_channel_geometry(*args, info=info)
    with pytest.raises(ValueError, match='ignore_ref=False'):
        mne_channel_geometry(ignored, *args[1:], info=info, ignore_ref=True)
    shifted = info.copy()
    shifted['chs'][0]['loc'][0] += 0.001
    with pytest.raises(ValueError, match='MEG coil'):
        mne_channel_geometry(*args, info=shifted, ignore_ref=False)

    # This reference contributes to the requested primary. Nominal CTF matching must
    # therefore verify the reference coil type as well as the output channel's hardware.
    other = info.copy()
    other['chs'][1]['coil_type'] = FIFF.FIFFV_COIL_MAGNES_REF_MAG
    with pytest.raises(ValueError, match='requires CTF sensors'):
        mne_channel_geometry(*args, info=other, ignore_ref=False, allow_nominal_ctf_coils=True)

    rows, scales, geometry = mne_channel_geometry(
        *args, info=info, ignore_ref=False, allow_nominal_ctf_coils=True)
    np.testing.assert_array_equal(rows, [0])
    # The primary is calibrated in tesla. Per nA m its fT gain has scale 1e15 times 1e-9.
    np.testing.assert_allclose(scales, [1e6])
    # Reference magnetometer, then both coils of the signed axial gradiometer.
    np.testing.assert_allclose(geometry['coilpos'],
                               [[0.03, 0.01, 0.13], [0.01, 0, 0.10], [0.01, 0, 0.15]])
    np.testing.assert_allclose(geometry['coilori'], [[0, 0, 1]] * 3)
    # All axes point along +z. The weights encode both the opposing primary pickups
    # and the subtraction of the separate reference magnetometer.
    np.testing.assert_allclose(geometry['tra'], [[-0.125, 1, -1]])
