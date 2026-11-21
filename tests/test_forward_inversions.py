"""Check supplied gains and source indices at the public inversion entry points."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.io import loadmat

from lameg import _spm_gain, invert
from lameg._forward_support import ForwardSupport
from tests.test_forward import _mesh
from tests.test_spm_gain_staging import source_model


@pytest.fixture(name='cortex')
def make_cortex(tmp_path):
    """Signed, distinct gain blocks on three registered layers and their combined GIfTI."""
    white = source_model()
    names = ('pial', '0.500', 'white')
    requested = np.vstack([white.requested_mri_mm + [0, 0, depth] for depth in (2, 1, 0)])
    effective = np.vstack([white.effective_head_mm + [0, 0, depth] for depth in (2, 1, 0)])
    operator = replace(white, layer_names=names, requested_mri_mm=requested,
                       effective_head_mm=effective, orientations_head=np.tile(
                           white.orientations_head, (len(names), 1)),
                       gain=np.hstack([-2 * white.gain, .5 * white.gain, white.gain]))
    paths = {name: str(tmp_path / ((name or 'combined') + '.gii')) for name in (*names, None)}
    for name, path in paths.items():
        layer = operator if name is None else operator.layer(name)
        triangles = np.vstack([layer.faces + i * layer.n_columns for i in range(
            len(layer.layer_names))])
        _mesh(path, layer.requested_mri_mm, triangles,
              layer.orientations_head @ layer.mri_to_head_mm[:3, :3])
    surfaces = SimpleNamespace(get_layer_names=lambda: names, n_layers=len(names),
                               get_mesh_path=lambda layer_name=None, **_: paths[layer_name])
    return surfaces, operator


def _check_job(cfg, method, sliding, windows):
    """Confirm SPM uses the same slot, mesh index convention and windows as the public call."""
    job = cfg['spm']['meeg']['source']['invertiter']
    custom = job['isstandard']['custom']
    assert job['val'] == 3 and custom['invtype'] == method
    assert custom.get('nlayers') == (3 if method == 'EBBlayer' else None)
    if method == 'EBBlayer' and not sliding:
        assert custom['ebblayer_diag_vertex'] == 2  # original 2 -> compact 1 -> MATLAB 2
    if sliding:
        np.testing.assert_array_equal(custom['woi'], windows)
    if method == 'MSP':
        path = custom['isfixedpatch']['fixedpatch']['fixedfile'][0]
        # Combined MSP priors can address any layer block.
        np.testing.assert_array_equal(loadmat(path)['Ip'].ravel(), [3] if sliding else [8, 2])


@pytest.mark.parametrize('inversion, layer_name, method', [
    (invert.invert_ebb, 'white', 'EBB'),
    (invert.invert_ebb_layer, None, 'EBBlayer'),
    (invert.invert_msp, None, 'MSP'),
    (invert.invert_sliding_window_ebb, None, 'EBB'),
    (invert.invert_sliding_window_ebb_layer, None, 'EBBlayer'),
    (invert.invert_sliding_window_msp, 'pial', 'MSP'),
])
def test_direct_operator_precedes_spatial_modes(monkeypatch, cortex, tmp_path, inversion,
                                               layer_name, method):
    """Each inverse computes its modes from the requested signed gain in the requested slot."""
    surfaces, operator = cortex
    # Original vertex 1 was excluded from a four-vertex mesh. Priors retain anatomical IDs.
    surfaces.forward_support = ForwardSupport(4, np.array([0, 2, 3]), {
        'pial': [1], '0.500': [], 'white': []})
    expected = operator if layer_name is None else operator.layer(layer_name)
    runtime = Mock()
    dataset = str(tmp_path / 'recording.mat')
    events = []
    windows = np.array([[-20., 0.], [0., 20.]])
    sliding = 'sliding_window' in inversion.__name__

    def install(path, gain, channels, positions, normals, units, **kwargs):
        assert path == dataset and kwargs['inversion_idx'] == 2
        assert kwargs['spm_instance'] is runtime
        assert kwargs['mesh_fname'] == surfaces.get_mesh_path(layer_name)
        assert channels == expected.channel_names and units == expected.channel_units
        np.testing.assert_array_equal(gain, expected.gain)
        np.testing.assert_allclose(positions, expected.effective_head_mm / 1000)
        np.testing.assert_array_equal(normals, expected.orientations_head)
        events.append('attach')

    def modes(*_args, **_kwargs):
        events.append('modes')
        return 'modes.mat', 2, 0.

    def run(cfg, **kwargs):
        assert kwargs['spm_instance'] is runtime
        _check_job(cfg, method, sliding, windows)
        events.append('batch')

    monkeypatch.setattr(_spm_gain, 'attach_spm_gain', install)
    runtime.spm_eeg_inv_prep_modes_xval.side_effect = modes
    runtime.spm_eeg_smoothmesh_multilayer_mm.side_effect = lambda *_args, **_kw: events.append(
        'smooth')
    monkeypatch.setattr(invert, 'batch', run)
    read_one = Mock(return_value=[-9., .1, 'source projector'])
    read_windows = Mock(return_value=np.array([-2., -3.]))
    monkeypatch.setattr(invert, '_read_inversion_results', read_one)
    monkeypatch.setattr(invert, '_read_sliding_window_free_energy', read_windows)
    options = {'data_fname': dataset, 'surf_set': surfaces, 'layer_name': layer_name,
               'inversion_idx': 2, 'spm_instance': runtime, 'forward_operator': operator}
    if method == 'MSP':
        options.update({'prior': 3} if sliding else {'priors': [10, 2]})
    if method == 'EBBlayer' and not sliding:
        options['diag_vertex'] = 2
    result = inversion(**options, **({'wois': windows} if sliding else {'return_mu_matrix': True}))

    assert events == ['attach', 'smooth', 'modes', 'batch']
    runtime.spm_eeg_inv_prep_modes_xval.assert_called_once()
    if sliding:
        np.testing.assert_array_equal(result[0], [-2., -3.])
        np.testing.assert_array_equal(result[1], windows)
        read_windows.assert_called_once_with(dataset, inversion_idx=2)
    else:
        assert result == [-9., .1, 'source projector']
        read_one.assert_called_once_with(dataset, inversion_idx=2, return_mu_matrix=True)


@pytest.mark.parametrize('invalid', ['combined_layer_order', 'layer_prior',
                                     'combined_prior', 'excluded_prior', 'bad_slot'])
def test_direct_invalid_operator_or_prior_leaves_slot_untouched(monkeypatch, cortex, tmp_path,
                                                                invalid):
    """Wrong layer identity and out-of-range source indices fail before replacing any gain."""
    surfaces, operator = cortex
    if invalid == 'combined_layer_order':
        operator = replace(operator, layer_names=operator.layer_names[::-1])
    if invalid == 'excluded_prior':
        surfaces.forward_support = ForwardSupport(4, np.array([0, 2, 3]), {
            'pial': [1], '0.500': [], 'white': []})
    install = Mock(side_effect=AssertionError('invalid gain must not be attached'))
    prepare = Mock(side_effect=AssertionError('invalid gain must not reach the SPM runtime'))
    monkeypatch.setattr(_spm_gain, 'attach_spm_gain', install)
    monkeypatch.setattr(invert, 'spm_context', prepare)
    layer = 'pial' if invalid == 'layer_prior' else None
    prior = {'combined_layer_order': 0, 'layer_prior': 3, 'combined_prior': 9,
             'excluded_prior': 9, 'bad_slot': 0}[invalid]
    with pytest.raises(ValueError):
        invert.invert_msp(str(tmp_path / 'data.mat'), surfaces, layer_name=layer, priors=[prior],
                          forward_operator=operator,
                          inversion_idx=-1 if invalid == 'bad_slot' else 0)
    install.assert_not_called()
    prepare.assert_not_called()
