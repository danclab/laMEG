"""Recording geometry for MEG forwards without a native gain cache."""
import h5py
import numpy as np
import pytest
from mne.transforms import Transform
from scipy.io import savemat

from lameg._forward_registration import fem_recording_geometry, mne_registration
from tests.test_forward import _char, _strings


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


def test_fem_recording_hdf5_before_any_native_gain(tmp_path):
    """Read good channels from the active montage, with signed weights from its registration."""
    path = tmp_path / 'recording.mat'
    native_to_mni, mni_to_head = np.eye(4), np.eye(4)
    native_to_mni[0, 3], mni_to_head[1, 3] = 8., 19.
    pos = np.array([[0., 0., .10], [.01, 0., .10], [0., .01, .15]])
    ori = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])

    with h5py.File(path, 'w') as header:
        header.create_group('D/channels')
        header['D/montage/Mind'] = 1
        channels = header.create_group('D/montage/M/channels')
        _strings(channels, 'label', ['B', 'BAD', 'A'])
        _strings(channels, 'units', ['fT/cm', 'T', 'fT'])
        _strings(channels, 'type', ['MEGGRAD', 'MEG', 'MEGMAG'])
        bad = channels.create_dataset('bad', (3, 1), dtype=h5py.ref_dtype)
        for i, flag in enumerate([0, 1, 0]):
            bad[i, 0] = header.create_dataset(f'bad_{i}', data=flag).ref

        # An inversion slot with just the registration. There is no gainmat,
        # scale, native source tessellation or precomputed list of gain rows.
        slot = header.create_group('slot')
        inv = header.create_dataset('D/other/inv', (1, 1), dtype=h5py.ref_dtype)
        inv[0, 0] = slot.ref
        reg = slot.create_group('datareg')
        _char(reg, 'modality', 'MEG')
        reg['fromMNI'] = mni_to_head.T
        slot['mesh/Affine'] = native_to_mni.T
        fwd = slot.create_group('forward')
        _char(fwd, 'modality', 'MEG')
        fwd['siunits'] = 1
        sensors = fwd.create_group('sensors')
        _char(sensors, 'unit', 'm')
        _strings(sensors, 'label', ['A', 'B', 'BAD'])
        _strings(sensors, 'chanunit', ['T', 'T/m', 'T'])
        sensors['coilpos'], sensors['coilori'] = pos.T, ori.T
        sensors['tra'] = np.array([[1., 0., -.2], [0., 4., -4.], [1., 1., 1.]]).T

    labels, units, geometry, affine = fem_recording_geometry(path)
    assert labels == ('B', 'A')
    assert units == ('fT/cm', 'fT')
    np.testing.assert_array_equal(affine, mni_to_head @ native_to_mni)
    np.testing.assert_array_equal(geometry['coilpos'], pos)
    np.testing.assert_array_equal(geometry['coilori'], ori)
    np.testing.assert_array_equal(geometry['tra'], [[0., 4., -4.], [1., 0., -.2]])

    with h5py.File(path, 'a') as header:
        header['slot/forward/siunits'][()] = 0
    with pytest.raises(ValueError, match='physical sensor units'):
        fem_recording_geometry(path)
    with pytest.raises(ValueError, match='Coregister'):
        fem_recording_geometry(path, inversion_idx=1)


def test_single_pickup_classic_mat(tmp_path):
    """Keep singleton channel and pickup axes that the classic MAT reader squeezes."""
    path = tmp_path / 'recording.mat'
    sensors = {'unit': 'm', 'label': 'M1', 'chanunit': 'T',
               'coilpos': [[0., .1, .1]], 'coilori': [[0., 0., 1.]], 'tra': [[-1.]]}
    slot = {'mesh': {'Affine': np.eye(4)}, 'datareg': {'modality': 'MEG', 'fromMNI': np.eye(4)},
            'forward': {'modality': 'MEG', 'siunits': 1, 'sensors': sensors}}
    channel = {'label': 'M1', 'type': 'MEGMAG', 'units': 'fT', 'bad': 0}
    savemat(path, {'D': {'channels': channel, 'other': {'inv': slot}}})
    labels, units, geometry, _ = fem_recording_geometry(path)
    assert (labels, units) == (('M1',), ('fT',))
    np.testing.assert_array_equal(geometry['coilpos'], [[0., .1, .1]])
    np.testing.assert_array_equal(geometry['tra'], [[-1.]])
