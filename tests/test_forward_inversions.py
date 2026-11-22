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
    # Distinct amplitudes and polarities identify each layer's gain even when
    # its channels, triangle topology and within-layer ordering match the others.
    operator = replace(white, layer_names=names, requested_mri_mm=requested,
                       effective_head_mm=effective, orientations_head=np.tile(
                           white.orientations_head, (len(names), 1)),
                       gain=np.hstack([-2 * white.gain, .5 * white.gain, white.gain]))
    paths = {name: str(tmp_path / ((name or 'combined') + '.gii')) for name in (*names, None)}
    for name, path in paths.items():
        layer = operator if name is None else operator.layer(name)
        # Each face indexes vertices in its own layer. In the concatenated GIfTI
        # shift each copy by a full layer so no triangle connects two depths.
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
        # With four original vertices per layer, ID 10 is white vertex 2.
        # Keeping [0, 2, 3] gives a white offset of 6 plus local index 1,
        # so SPM should read Ip=8 after conversion to one-based indexing.
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

    # Record the values SPM would receive while keeping mesh validation,
    # layer selection, source remapping and construction of the batch real.
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

    # Two time windows still require just one gain installation and one set
    # of spatial modes. Preparing those modes earlier would use the old gain.
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
    # Original combined ID 9 is white vertex 1. Losing that cortical location
    # on pial also removes it from the shared support on white.
    prior = {'combined_layer_order': 0, 'layer_prior': 3, 'combined_prior': 9,
             'excluded_prior': 9, 'bad_slot': 0}[invalid]
    with pytest.raises(ValueError):
        invert.invert_msp(str(tmp_path / 'data.mat'), surfaces, layer_name=layer, priors=[prior],
                          forward_operator=operator,
                          inversion_idx=-1 if invalid == 'bad_slot' else 0)
    install.assert_not_called()
    prepare.assert_not_called()


@pytest.mark.parametrize('sliding', [False, True], ids=['static', 'sliding'])
@pytest.mark.parametrize('n_layers, pair_options, expected', [
    (2, {}, (1, 1)),
    (3, {}, (2, 2)),
    (3, {'sum_pair_topk': 3, 'diff_pair_topk': 1}, (3, 1)),
])
def test_ebblayer_pair_counts_in_spm_job(monkeypatch, sliding, n_layers, pair_options, expected):
    """Both EBBlayer entry points use only available pairs and honor independent overrides."""
    run = Mock()
    monkeypatch.setattr(invert, '_prepare_spatial_modes', Mock(return_value=('modes.mat', 2, 0.)))
    monkeypatch.setattr(invert, 'batch', run)
    monkeypatch.setattr(invert, '_read_inversion_results', Mock(return_value=[0., 0.]))
    monkeypatch.setattr(invert, '_read_sliding_window_free_energy', Mock(return_value=[0., 0.]))
    surfaces = SimpleNamespace(n_layers=n_layers)
    if sliding:
        invert.invert_sliding_window_ebb_layer('data.mat', surfaces, wois=[[0, 10], [10, 20]],
                                               **pair_options)
    else:
        invert.invert_ebb_layer('data.mat', surfaces, **pair_options)
    # Inspect the actual batch fields consumed by compiled SPM. A two-layer fit
    # that requests two pairs fails there even if the call accepts its arguments.
    job = run.call_args[0][0]['spm']['meeg']['source']['invertiter']['isstandard']['custom']
    assert job['invtype'] == 'EBBlayer' and job['nlayers'] == n_layers
    assert (job['ebblayer_sum_pair_topk'], job['ebblayer_diff_pair_topk']) == expected


@pytest.mark.parametrize('sliding', [False, True], ids=['static', 'sliding'])
@pytest.mark.parametrize('layer, pair_options', [
    ('pial', {}), (None, {'sum_pair_topk': 2}), (None, {'diff_pair_topk': 2}),
    (None, {'sum_pair_topk': 0}), (None, {'diff_pair_topk': -1}),
    (None, {'sum_pair_topk': 0.5}),
])
def test_ebblayer_invalid_pair_counts_stop_before_spm(monkeypatch, sliding, layer, pair_options):
    """Reject impossible layer pairs before attaching a gain or creating spatial modes."""
    prepare = Mock(side_effect=AssertionError('invalid pair counts reached mode preparation'))
    monkeypatch.setattr(invert, '_prepare_spatial_modes', prepare)
    surfaces = SimpleNamespace(n_layers=2)
    with pytest.raises(ValueError, match='layer|pair'):
        if sliding:
            invert.invert_sliding_window_ebb_layer('data.mat', surfaces, layer_name=layer,
                                                   wois=[[0, 10]], **pair_options)
        else:
            invert.invert_ebb_layer('data.mat', surfaces, layer_name=layer, **pair_options)
    prepare.assert_not_called()
