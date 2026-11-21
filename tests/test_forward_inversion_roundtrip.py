"""End-to-end checks for direct and windowed inversions with precomputed gains."""
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse import issparse

from lameg.forward import ForwardOperator
from lameg.invert import coregister, invert_ebb_layer
from lameg.laminar import sliding_window_model_comparison
from tests.forward_test_utils import separate_dataset


def test_multilayer_ebb_uses_signed_supplied_gain(tmp_path, spm, forward_registration):
    """A global gain sign reversal preserves evidence and reverses the source projector."""
    source, fids, surfaces, _, operator = forward_registration
    dataset = separate_dataset(source, tmp_path / 'multilayer')
    coregister(fids, dataset, surfaces, viz=False, spm_instance=spm)
    opts = {'woi': [100., 200.], 'n_spatial_modes': 'all', 'sum_pair_topk': 1,
            'diff_pair_topk': 1, 'return_mu_matrix': True, 'spm_instance': spm, 'viz': False}
    # return_mu_matrix=True adds a third item to each result.
    # pylint: disable=unbalanced-tuple-unpacking
    native_f, native_cv, native_mu = invert_ebb_layer(dataset, surfaces, **opts)
    flipped = replace(operator, gain=-operator.gain, provenance='EBBlayer polarity test')
    supplied_f, supplied_cv, supplied_mu = invert_ebb_layer(
        dataset, surfaces, forward_operator=flipped, **opts)
    # pylint: enable=unbalanced-tuple-unpacking

    assert np.isfinite(native_f).all() and native_mu.shape == operator.gain.T.shape
    np.testing.assert_allclose(supplied_f, native_f, rtol=0, atol=1e-4)
    np.testing.assert_allclose(supplied_cv, native_cv)
    native_mu = native_mu.toarray() if issparse(native_mu) else native_mu
    supplied_mu = supplied_mu.toarray() if issparse(supplied_mu) else supplied_mu
    assert np.linalg.norm(native_mu) > 0
    np.testing.assert_allclose(supplied_mu, -native_mu, rtol=1e-6, atol=1e-10)
    installed = ForwardOperator.from_spm(
        dataset, surfaces.get_mesh_path(), surfaces.get_layer_names(), allow_undefined_gain=True)
    np.testing.assert_array_equal(installed.gain, flipped.gain)
    assert installed.provenance == flipped.provenance


def test_sliding_model_comparison_roundtrip(tmp_path, spm, forward_registration):
    """Native and supplied gains give matching free energies for each named layer and window."""
    source, fids, surfaces, _, operator = forward_registration
    layers = [operator.layer(name) for name in surfaces.get_layer_names()]
    defined = np.logical_and.reduce([layer.orientation_defined for layer in layers])
    magnitude = np.minimum.reduce([np.linalg.norm(layer.gain, axis=0) for layer in layers])
    prior = int(np.argmax(np.where(defined, magnitude, 0)))
    windows = np.array([[100., 150.], [150., 200.]])
    opts = {'spm_instance': spm, 'viz': False,
            'invert_kwargs': {'wois': windows, 'n_temp_modes': 2}}
    native = separate_dataset(source, tmp_path / 'native')
    supplied = separate_dataset(source, tmp_path / 'supplied')
    native_f, native_windows = sliding_window_model_comparison(
        prior, fids, native, surfaces, **opts)
    supplied_f, supplied_windows = sliding_window_model_comparison(
        prior, fids, supplied, surfaces, forward_operator=operator, **opts)

    assert native_f.shape == (len(layers), len(windows)) and np.isfinite(native_f).all()
    assert np.ptp(native_f, axis=0).max() > 1e-4
    np.testing.assert_array_equal(native_windows, windows)
    np.testing.assert_array_equal(supplied_windows, windows)
    np.testing.assert_allclose(supplied_f, native_f, rtol=0, atol=1e-4)
    assert len(list(Path(supplied).parent.glob('lameg_gain_*.mat'))) == len(layers)
