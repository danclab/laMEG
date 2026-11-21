"""Check physical-unit conversion, MAT serialization and failed-attachment behavior."""
from pathlib import Path
from unittest.mock import Mock
import re

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from lameg.laminar import _attach_layer_forward
from lameg._spm_gain import attach_spm_gain
from lameg.forward import ForwardOperator


def source_model():
    """Three signed sources whose effective coordinates differ from their requested positions."""
    return ForwardOperator(
        gain=np.array([[-4., 9., 3.], [7., -3., 1.]]),
        channel_names=('MEG2', 'MEG1'), channel_units=('fT', 'T/m'), layer_names=('white',),
        requested_mri_mm=np.array([[1., 2., 3.], [4., 5., 6.], [4., 4., 7.]]),
        effective_head_mm=np.array([[12., 24., 31.], [12., 23., 36.], [11., 21., 34.]]),
        orientations_head=np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]),
        mri_to_head_mm=np.array([[0., -1., 0., 15.], [1., 0., 0., 20.],
                                [0., 0., 1., 28.], [0., 0., 0., 1.]]),
        faces=np.array([[0, 1, 2]]), provenance='synthetic',
        sensor_geometry={'coilpos': np.array([[0., .1, .2], [.1, .2, 0.]]),
                         'coilori': np.array([[1., 0., 0.], [0., 1., 0.]]),
                         'tra': np.array([[1., -.8], [.3, 1.]])},
    )


def test_attachment_stages_signed_physical_gain_and_two_geometries(tmp_path):
    """Serialize one physical gain with both source geometries and the sensor definition."""
    source = source_model()
    dataset = tmp_path / "S' dataset.mat"
    dataset.write_bytes(b'old header')
    caches = []

    def stage(command, script, nargout):
        assert command == 'eval' and nargout == 0
        config_path = re.search(r"config_file='((?:''|[^'])*)';", script).group(1)
        config = loadmat(config_path.replace("''", "'"), squeeze_me=True)
        assert config['slot'] == 3
        assert config['dataset'] == str(dataset)
        staged = loadmat(config['staged_gain'], squeeze_me=True)
        cache = dataset.parent / config['read_cache']
        assert cache.is_file()
        assert np.array_equal(loadmat(str(cache))['G'], staged['G'])
        np.testing.assert_array_equal(staged['G'], source.gain)
        assert staged['label'].tolist() == list(source.channel_names)
        assert staged['channel_units'].tolist() == list(source.channel_units)
        np.testing.assert_array_equal(staged['normals'], source.orientations_head)
        expected = [[.013, .021, .031], [.010, .024, .034], [.011, .024, .035]]
        np.testing.assert_allclose(staged['requested_m'], expected)
        np.testing.assert_allclose(staged['positions_m'], source.effective_head_mm / 1000.)
        for key in ('coilpos', 'coilori', 'tra'):
            np.testing.assert_array_equal(staged[key], source.sensor_geometry[key])
        caches.append(config['cache_name'])
        # Simulate a successful verified MATLAB save; native acceptance is covered separately.
        Path(config['staged_header']).write_bytes(b'validated new header')
        savemat(config['status_file'], {'message': ''})

    runtime = Mock()
    runtime.spm_standalone.side_effect = stage
    for _ in range(2):
        _attach_layer_forward(str(dataset), source, 2, runtime)
    assert dataset.read_bytes() == b'validated new header'
    assert caches[0] == caches[1]
    assert len(list(tmp_path.glob('lameg_gain_*.mat'))) == 1


def test_rejected_runtime_attachment_preserves_header(tmp_path):
    """Leave the original SPM header intact when target metadata fails validation."""
    forward = source_model()
    dataset = tmp_path / 'sample.mat'
    dataset.write_bytes(b'original header')

    def reject(_command, script, nargout):
        assert nargout == 0
        config = loadmat(re.search(r"config_file='([^']+)';", script).group(1), squeeze_me=True)
        savemat(config['status_file'], {'message': 'MEG coil positions differ.'})

    runtime = Mock()
    runtime.spm_standalone.side_effect = reject
    with pytest.raises(ValueError, match='coil positions'):
        _attach_layer_forward(str(dataset), forward, 0, runtime)
    assert dataset.read_bytes() == b'original header'
    assert not list(tmp_path.glob('lameg_gain_*.mat'))


def test_wrong_coil_transfer_dimensions_rejected_before_runtime(tmp_path):
    """Reject a transfer matrix with the wrong number of channels or coils."""
    forward = source_model()
    dataset = tmp_path / 'sample.mat'
    dataset.write_bytes(b'original header')
    wrong = dict(forward.sensor_geometry, tra=np.eye(3))
    runtime = Mock()
    with pytest.raises(ValueError, match='coil geometry|transfer matrix'):
        attach_spm_gain(str(dataset), forward.gain, forward.channel_names,
                        forward.effective_head_mm / 1000., forward.orientations_head,
                        forward.channel_units, sensor_geometry=wrong, spm_instance=runtime)
    assert not runtime.spm_standalone.called
    assert dataset.read_bytes() == b'original header'
