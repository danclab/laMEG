"""Regression tests for forward-model selection during laminar comparison."""
from dataclasses import replace
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest

from lameg import laminar
from lameg.forward import ForwardOperator


class TwoLayerSurfaces:
    """Two layers in the order used by the existing reconstructions."""

    n_layers = 2
    forward_support = None

    def get_layer_names(self):
        """Surface order used for layer comparisons."""
        return ['pial', 'white']

    def get_vertices_per_layer(self):
        """Number of corresponding vertices in each mesh."""
        return 3


def record_reconstruction(monkeypatch, method, events, get_scores):
    """Replace only the inversion call and keep its keyword arguments observable."""
    def invert(data_fname, surf_set, **kwargs):
        layer = kwargs['layer_name']
        events.append(('invert', layer))
        assert data_fname == 'data.mat'
        assert surf_set.get_layer_names() == ['pial', 'white']
        assert kwargs['n_spatial_modes'] == 'all'
        assert kwargs['patch_size'] == 5
        return get_scores(layer)

    monkeypatch.setattr(laminar, 'invert_' + method.lower(), invert)


@pytest.mark.parametrize('device', ['squid', 'opm'])
@pytest.mark.parametrize('method', ['EBB', 'MSP'])
def test_native_comparison_preserves_setup_and_output(monkeypatch, device, method):
    """Existing device and inverse-method combinations preserve their ordering and outputs."""
    events = []
    surfaces = TwoLayerSurfaces()
    fids = {'nas': [0, 1, 0], 'lpa': [-1, 0, 0], 'rpa': [1, 0, 0]}
    spm = object()
    if device == 'squid':
        forward_kwargs = {'forward_model_type': 'MEG OpenMEEG BEM'}
    else:
        forward_kwargs = {'voltype': 'Single Shell'}

    def build(*args, **kwargs):
        expected = (fids, 'data.mat', surfaces) if device == 'squid' else ('data.mat', surfaces)
        assert args == expected
        assert kwargs['spm_instance'] is spm
        assert kwargs['viz'] is False
        assert kwargs['stage'] == 'ds' and kwargs['orientation'] == 'link_vector'
        assert kwargs['fixed'] is True
        for key, value in forward_kwargs.items():
            assert kwargs[key] == value
        events.append(('native', kwargs['layer_name']))

    squid = build if device == 'squid' else Mock(side_effect=AssertionError('unexpected SQUID'))
    opm = build if device == 'opm' else Mock(side_effect=AssertionError('unexpected OPM'))
    monkeypatch.setattr(laminar, 'coregister', squid)
    monkeypatch.setattr(laminar, 'opm_headmodel', opm)
    scores = {'pial': (-4., .25), 'white': (-2., .5)}
    record_reconstruction(monkeypatch, method, events, scores.__getitem__)
    scores_f, scores_cv = laminar.model_comparison(
        fids if device == 'squid' else None, 'data.mat', surfaces, method=method,
        device=device, viz=False, spm_instance=spm, forward_kwargs=forward_kwargs,
        invert_kwargs={'patch_size': 5},
    )
    assert events == [('native', 'pial'), ('invert', 'pial'),
                      ('native', 'white'), ('invert', 'white')]
    np.testing.assert_array_equal(scores_f, [-4., -2.])
    np.testing.assert_array_equal(scores_cv, [.25, .5])


@pytest.fixture(name='input_forward')
def build_input_forward(tmp_path):
    """One asymmetric signed gain matrix and two source meshes in MRI mm."""
    class Surfaces(TwoLayerSurfaces):
        """Resolve the two synthetic test meshes."""
        def get_mesh_path(self, layer_name=None, stage='ds', orientation='link_vector',
                          fixed=True):
            """Return the path to the requested oriented cortex."""
            assert (stage, orientation, fixed) == ('ds', 'link_vector', True)
            return str(tmp_path / (layer_name + '.gii'))

    surfaces = Surfaces()
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    vertices = np.array([[10, 0, 12], [10, 1, 12], [9, 1, 12]], dtype=float)
    normals = np.tile([0., 0., 1.], (3, 1))
    layers = surfaces.get_layer_names()
    geometry = [vertices, vertices - [0., 0., 2.]]
    for name, points in zip(layers, geometry):
        mesh = nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(points.astype('float32'), intent='NIFTI_INTENT_POINTSET'),
            nib.gifti.GiftiDataArray(faces, intent='NIFTI_INTENT_TRIANGLE'),
            nib.gifti.GiftiDataArray(normals.astype('float32'), intent='NIFTI_INTENT_VECTOR')
        ])
        nib.save(mesh, surfaces.get_mesh_path(name))

    # Different signs within each layer make row/column transposes and sign loss observable.
    gain = np.array([[-1., 21., -3., 5., -5., 9.], [3., -6., -2., 7., 4., -3.]])
    points = np.concatenate(geometry)
    operator = ForwardOperator(
        gain=gain, channel_names=['MEG001', 'MEG002'], channel_units=['fT', 'fT'],
        layer_names=layers, requested_mri_mm=points, effective_head_mm=points.copy(),
        orientations_head=np.tile(normals, (2, 1)), mri_to_head_mm=np.eye(4),
        faces=faces, sensor_geometry={
            'coilpos': np.array([[0., 0.1, 0.2], [0.1, 0.2, 0.]]),
            'coilori': np.array([[1., 0., 0.], [0., 1., 0.]]), 'tra': np.eye(2),
        }, provenance='synthetic two-layer fixed-orientation MEG test'
    )
    return surfaces, operator


@pytest.mark.parametrize('device, method, slot, specified_in', [
    ('squid', 'MSP', 2, 'both'),
    ('squid', 'MSP', 2, 'forward'),
    ('squid', 'MSP', 2, 'inverse'),
    ('opm', 'EBB', 0, 'inverse'),
])
def test_supplied_gain_attaches_after_setup_before_each_inversion(
        monkeypatch, input_forward, device, method, slot, specified_in):
    """Each inversion receives the signed gain for its own layer and slot."""
    surfaces, operator = input_forward
    events = []
    runtime = object()
    attached = {}
    fids = {'nas': [0, 1, 0], 'lpa': [-1, 0, 0], 'rpa': [1, 0, 0]}
    # Exercise both places where callers can specify an SPM slot. Registration
    # and inversion must end up targeting the same slot in either case.
    native_kwargs = {'inversion_idx': slot} if specified_in in ('both', 'forward') else {}
    inverse_kwargs = {'patch_size': 5}
    if specified_in in ('both', 'inverse'):
        inverse_kwargs['inversion_idx'] = slot
    original_options = native_kwargs.copy(), inverse_kwargs.copy()

    def setup(*_args, **kwargs):
        assert kwargs['spm_instance'] is runtime
        assert kwargs.get('inversion_idx', 0) == slot
        events.append(('setup', kwargs['layer_name']))

    def attach(data_fname, layer_operator, inversion_idx, spm_instance, mesh_fname):
        assert data_fname == 'data.mat' and spm_instance is runtime
        assert inversion_idx == slot
        assert len(layer_operator.layer_names) == 1
        name = layer_operator.layer_names[0]
        assert mesh_fname == surfaces.get_mesh_path(name)
        events.append(('attach', name))
        attached[name] = layer_operator.gain.copy()

    def invert(data_fname, surf_set, **kwargs):
        assert data_fname == 'data.mat' and surf_set is surfaces
        assert kwargs.get('inversion_idx', 0) == slot
        assert kwargs['spm_instance'] is runtime
        name = kwargs['layer_name']
        events.append(('invert', name))
        # The score carries signed information from the layer that was actually installed.
        return attached[name][0, 0], attached[name][1, -1]

    native_fn = 'coregister' if device == 'squid' else 'opm_headmodel'
    monkeypatch.setattr(laminar, native_fn, setup)
    monkeypatch.setattr(laminar, 'attach_forward_operator', attach)
    monkeypatch.setattr(laminar, 'invert_' + method.lower(), invert)
    f_val, cv_err = laminar.model_comparison(
        fids if device == 'squid' else None, 'data.mat', surfaces, method=method,
        device=device, viz=False, spm_instance=runtime, forward_kwargs=native_kwargs,
        invert_kwargs=inverse_kwargs, forward_operator=operator,
    )

    assert events == [('setup', 'pial'), ('attach', 'pial'), ('invert', 'pial'),
                      ('setup', 'white'), ('attach', 'white'), ('invert', 'white')]
    np.testing.assert_array_equal(attached['pial'], operator.gain[:, :3])
    np.testing.assert_array_equal(attached['white'], operator.gain[:, 3:])
    np.testing.assert_array_equal(f_val, [-1., 5.])
    np.testing.assert_array_equal(cv_err, [-2., -3.])
    assert (native_kwargs, inverse_kwargs) == original_options


def test_invalid_second_layer_fails_before_any_dataset_change(monkeypatch, input_forward):
    """Preflight all layers before touching the first one."""
    surfaces, operator = input_forward
    path = surfaces.get_mesh_path('white')
    mesh = nib.load(path)
    # Only the second layer is wrong. Checking meshes inside the inversion loop
    # would already have overwritten the slot for pial before discovering this.
    mesh.darrays[0].data[1] += [0., 3., 0.]
    nib.save(mesh, path)
    setup = Mock(side_effect=AssertionError('a bad mesh must fail before native setup'))
    attach = Mock(side_effect=AssertionError('a bad mesh must fail before attaching gains'))
    monkeypatch.setattr(laminar, 'coregister', setup)
    monkeypatch.setattr(laminar, 'attach_forward_operator', attach)
    with pytest.raises(ValueError):
        laminar.model_comparison({}, 'data.mat', surfaces, forward_operator=operator)
    setup.assert_not_called()
    attach.assert_not_called()


def test_opm_nonzero_inversion_slot_fails_before_setup(monkeypatch, input_forward):
    """OPM head model setup only supports the default slot."""
    surfaces, operator = input_forward
    setup = Mock(side_effect=AssertionError('OPM only prepares inversion slot zero'))
    monkeypatch.setattr(laminar, 'opm_headmodel', setup)
    with pytest.raises(ValueError):
        laminar.model_comparison(None, 'data.mat', surfaces, device='opm',
                                 invert_kwargs={'inversion_idx': 1}, forward_operator=operator)
    setup.assert_not_called()


def test_conflicting_inversion_slots_fail_before_setup(monkeypatch, input_forward):
    """A gain and inversion must refer to the same SPM slot."""
    surfaces, operator = input_forward
    setup = Mock(side_effect=AssertionError('conflicting slots must fail before setup'))
    monkeypatch.setattr(laminar, 'coregister', setup)
    with pytest.raises(ValueError):
        laminar.model_comparison(
            {}, 'data.mat', surfaces, forward_kwargs={'inversion_idx': 1},
            invert_kwargs={'inversion_idx': 2}, forward_operator=operator,
        )
    setup.assert_not_called()


@pytest.mark.parametrize('prior, message', [(1, 'undefined direction'), (3, 'outside')])
@pytest.mark.parametrize('sliding', [False, True])
def test_seeded_msp_requires_a_defined_column_on_every_layer(
        monkeypatch, input_forward, prior, message, sliding):
    """Reject an invalid explicit patch prior before building any forward model."""
    surfaces, operator = input_forward
    normals = operator.orientations_head.copy()
    # An archived operator can carry a nonzero gain with a missing orientation.
    # Such a column cannot define the sign of a seeded MSP patch.
    normals[4] = 0  # White vertex 1 only.
    operator = replace(operator, orientations_head=normals, allow_undefined_gain=True)
    white = surfaces.get_mesh_path('white')
    mesh = nib.load(white)
    mesh.darrays[2].data[1] = 0
    nib.save(mesh, white)
    setup = Mock(side_effect=AssertionError('invalid prior must fail before setup'))
    monkeypatch.setattr(laminar, 'coregister', setup)
    with pytest.raises(ValueError, match=message):
        if sliding:
            laminar.sliding_window_model_comparison(prior, {}, 'data.mat', surfaces,
                                                    forward_operator=operator)
        else:
            laminar.model_comparison({}, 'data.mat', surfaces, method='MSP',
                                     forward_operator=operator, invert_kwargs={'priors': [prior]})
    setup.assert_not_called()


def test_layer_names_must_match_surfaces(monkeypatch, input_forward):
    """Layer labels must identify exactly the surfaces in the requested comparison."""
    surfaces, operator = input_forward
    setup = Mock(side_effect=AssertionError('mismatched layers must fail before setup'))
    monkeypatch.setattr(laminar, 'coregister', setup)
    with pytest.raises(ValueError, match='layer names'):
        laminar.model_comparison({}, 'data.mat', surfaces, forward_operator=operator.layer('pial'))
    setup.assert_not_called()


@pytest.mark.parametrize('device, slot', [('squid', 2), ('opm', 0)])
def test_sliding_uses_named_gain_once_for_all_windows(monkeypatch, input_forward, device, slot):
    """Each layer gets its matching signed gain, inversion slot and common time grid."""
    surfaces, operator = input_forward
    windows = np.array([[-10., 10.], [10., 30.]])
    events = []
    attached = {}
    # Preflight checks vertex 41 at its compact index 1. The public inverse
    # must still receive 41 so it can map the original anatomical ID itself.
    original_prior = 41
    support = Mock()
    support.map_indices.return_value = np.array([1])
    surfaces.forward_support = support

    def setup(*_args, **kwargs):
        assert kwargs.get('inversion_idx', 0) == slot
        events.append(('setup', kwargs['layer_name']))

    def attach(_dataset, layer, inversion_idx, _spm, mesh_fname):
        name = layer.layer_names[0]
        assert inversion_idx == slot and mesh_fname == surfaces.get_mesh_path(name)
        attached[name] = layer
        events.append(('attach', name))

    def invert(prior, _dataset, _surfaces, layer_name, **kwargs):
        assert prior == original_prior and kwargs['inversion_idx'] == slot
        assert kwargs['n_spatial_modes'] == 'all'
        assert 'forward_operator' not in kwargs
        if layer_name == 'white':
            np.testing.assert_array_equal(kwargs['wois'], windows)
        events.append(('invert', layer_name))
        return attached[layer_name].gain[:, 1], windows

    monkeypatch.setattr(laminar, 'coregister' if device == 'squid' else 'opm_headmodel', setup)
    monkeypatch.setattr(laminar, 'attach_forward_operator', attach)
    monkeypatch.setattr(laminar, 'invert_sliding_window_msp', invert)
    options = {'inversion_idx': slot}
    evidence, actual_windows = laminar.sliding_window_model_comparison(
        original_prior, {}, 'data.mat', surfaces, device=device, invert_kwargs=options,
        forward_operator=operator)
    mapped, call_options = support.map_indices.call_args
    np.testing.assert_array_equal(mapped[0], [original_prior])
    assert call_options == {'multilayer': False}
    assert events == [('setup', 'pial'), ('attach', 'pial'), ('invert', 'pial'),
                      ('setup', 'white'), ('attach', 'white'), ('invert', 'white')]
    # The mock inverse uses signed sensor values as distinct window scores.
    # Local vertex 1 has global gain columns 1 and 4 in the two layer blocks.
    np.testing.assert_array_equal(evidence, operator.gain[:, [1, 4]].T)
    np.testing.assert_array_equal(actual_windows, windows)
    assert options == {'inversion_idx': slot}


def test_sliding_rejects_window_misalignment(monkeypatch):
    """Model evidence from different time intervals cannot share a result column."""
    windows = np.array([[5., 15.], [15., 25.]])
    # Reversed intervals keep the array shape unchanged while assigning the
    # second layer's scores to the wrong times.
    mock = Mock(side_effect=[([1., 2.], windows), ([3., 4.], windows[::-1])])
    monkeypatch.setattr(laminar, 'coregister', Mock())
    monkeypatch.setattr(laminar, 'invert_sliding_window_msp', mock)
    with pytest.raises(ValueError, match="windows differ for layer 'white'"):
        laminar.sliding_window_model_comparison(1, {}, 'data.mat', TwoLayerSurfaces())


@pytest.mark.parametrize('sliding', [False, True])
def test_excluded_original_prior_stops_native_comparison_before_setup(monkeypatch, sliding):
    """The native solver must respect the common anatomical support too."""
    surfaces = TwoLayerSurfaces()
    surfaces.forward_support = Mock()
    # A native single-shell run can reuse surfaces restricted for BEM. Both
    # solvers must reject the same excluded seed before either modifies the slot.
    surfaces.forward_support.map_indices.side_effect = ValueError("Vertex 3 excluded from pial.")
    setup = Mock(side_effect=AssertionError('a missing prior must fail before registration'))
    monkeypatch.setattr(laminar, 'coregister', setup)
    with pytest.raises(ValueError, match='Vertex 3 excluded'):
        if sliding:
            laminar.sliding_window_model_comparison(3, {}, 'data.mat', surfaces)
        else:
            laminar.model_comparison({}, 'data.mat', surfaces, method='MSP',
                                     invert_kwargs={'priors': [3]})
    setup.assert_not_called()
