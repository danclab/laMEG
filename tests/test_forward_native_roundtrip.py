"""End-to-end round trip using the same sub-104 data as the laMEG integration suite."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lameg.laminar import model_comparison, _attach_layer_forward
from tests.forward_test_utils import separate_dataset


def test_native_and_supplied_forward_give_same_layer_evidence(tmp_path, spm, forward_registration):
    """Native and reimported gains yield the same single-patch evidence per layer."""
    source, fids, surfaces, _, operator = forward_registration
    native = separate_dataset(source, tmp_path / 'native')
    supplied = separate_dataset(source, tmp_path / 'supplied')
    layers = [operator.layer(name) for name in surfaces.get_layer_names()]
    valid = np.logical_and.reduce([layer.orientation_defined for layer in layers])
    # A seed is a within-layer vertex ID. Rank sites by the smaller of their two
    # layer-specific sensor norms to choose one observable site shared by both models.
    gain_norm = np.minimum.reduce([np.linalg.norm(layer.gain, axis=0) for layer in layers])
    candidate = np.where(valid, gain_norm, 0.)
    assert candidate.max() > 0
    prior = int(np.argmax(candidate))
    # Use the same trials, seed and temporal modes for the two fits. The native path
    # computes the layer gains in SPM. The supplied path installs the donor's gains.
    kwargs = {'method': 'MSP', 'viz': False, 'spm_instance': spm,
              'invert_kwargs': {'woi': [100., 200.], 'priors': [prior], 'n_temp_modes': 4}}
    native_f, native_cv = model_comparison(fids, native, surfaces, **kwargs)
    supplied_f, supplied_cv = model_comparison(fids, supplied, surfaces,
                                               forward_operator=operator, **kwargs)
    assert native_f.size == supplied_f.size == 2
    assert np.all(np.isfinite(native_f))
    assert abs(native_f.ravel()[1] - native_f.ravel()[0]) > 1e-4
    # Free energies are large and their between-layer differences can be small.
    # An absolute tolerance in log-evidence units tests both levels of the comparison.
    np.testing.assert_allclose(supplied_f, native_f, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.diff(supplied_f, axis=0), np.diff(native_f, axis=0),
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(supplied_cv, native_cv, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize(('problem', 'error'), [
    ('strict', 'coil positions'), ('displaced', 'coil positions'), ('polarity', 'compensation')])
def test_nominal_ctf_attachment_rejects_mismatches(spm, forward_registration, problem, error):
    """Check measured CTF pickups and compensation in the actual standalone attachment path."""
    _, _, _, dataset, operator = forward_registration
    positions = operator.sensor_geometry['coilpos'].copy()
    transfer = operator.sensor_geometry['tra'].copy()
    # CTF reference compensation mixes reference pickups into a primary channel.
    # Pick a fractional transfer weight to test that the signed reference contribution
    # survives the sensor check even if primary pickup coordinates still match.
    row, coil = np.argwhere((np.abs(transfer) > 1e-4) & (np.abs(transfer) < 0.9))[0]
    if problem == 'polarity':
        transfer[row, coil] *= -1
    else:
        # 0.10 mm on one axis is within the nominal limit, 0.12 mm on two axes exceeds it.
        shift = [0.10e-3, 0., 0.] if problem == 'strict' else [0.12e-3, 0.12e-3, 0.]
        # Several coil axes can share a measured centre. Move the complete group so
        # the comparison isolates a change in location from a change in its response.
        colocated = np.linalg.norm(positions - positions[coil], axis=1) < 1e-7
        positions[colocated] += shift
    geometry = dict(operator.sensor_geometry, coilpos=positions, tra=transfer)
    invalid = replace(operator, sensor_geometry=geometry,
                      allow_nominal_ctf_coils=problem != 'strict')
    # Failed validation must leave the active SPM slot usable for subsequent runs.
    # Check the actual header bytes and verify the rejected cache was never published.
    before = Path(dataset).read_bytes()
    with pytest.raises(ValueError, match=error):
        _attach_layer_forward(dataset, invalid, 0, spm)
    assert Path(dataset).read_bytes() == before
    assert not list(Path(dataset).parent.glob('lameg_gain_*.mat'))
