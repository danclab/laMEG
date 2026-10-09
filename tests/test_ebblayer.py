"""Lightweight numerical tests for pure-Python EBBlayer source priors.

The large, subject-specific 2- and 11-layer DANC comparisons remain separate
regression audits and should not be included in the ordinary unit-test suite.
"""

import numpy as np
import pytest
from scipy.sparse import eye

from lameg.ebblayer import build_ebblayer_priors


def _sample(n_layers=3, n_vertices=32, n_modes=6):
    rng = np.random.RandomState(0)
    n = n_layers * n_vertices
    ul = rng.randn(n_modes, n)
    r = rng.randn(n_modes, n_modes)
    ayya = r @ r.T + 0.5 * np.eye(n_modes)
    return ul, ayya, eye(n, format="csc")


def test_three_layer_covariance_structure_and_topk():
    """Source and sensor priors preserve cross-layer covariance structure."""
    ul, ayya, qg = _sample()
    result = build_ebblayer_priors(
        ul, ayya, qg, n_layers=3, sum_pair_topk=2, diff_pair_topk=2)

    assert result["pairs"] == [(0, 1), (0, 2), (1, 2)]
    assert np.all(result["keep_sum"].sum(axis=1) <= 2)
    assert np.all(result["keep_diff"].sum(axis=1) <= 2)
    assert not np.any(result["keep_sum"] & result["endpoint"])
    assert np.count_nonzero(result["keep_sum"]) > 0
    assert np.count_nonzero(result["keep_diff"]) > 0

    qind, qsum, qdiff = result["source_q"]
    assert np.count_nonzero(qind.toarray() - np.diag(qind.diagonal())) == 0

    sum_off = qsum.toarray() - np.diag(qsum.diagonal())
    diff_off = qdiff.toarray() - np.diag(qdiff.diagonal())
    assert np.count_nonzero(sum_off) > 0
    assert np.count_nonzero(diff_off) > 0
    assert np.all(sum_off >= -1e-14)
    assert np.all(diff_off <= 1e-14)

    for source, sensor in zip(result["source_q"], result["sensor_q"]):
        dense_source = source.toarray()
        assert np.allclose(dense_source, dense_source.T, atol=1e-12)
        assert np.linalg.eigvalsh(dense_source).min() >= -1e-10
        assert np.isclose(np.trace(sensor), 1.0, atol=1e-12)
        assert np.allclose(ul @ source @ ul.T, sensor, rtol=1e-11, atol=1e-12)


def test_two_layer_single_pair_and_invalid_topk():
    """Two-layer hypotheses have only one pair, so K must equal one."""
    ul, ayya, qg = _sample(n_layers=2)
    result = build_ebblayer_priors(
        ul, ayya, qg, n_layers=2, sum_pair_topk=1, diff_pair_topk=1)
    assert result["pairs"] == [(0, 1)]
    assert result["keep_sum"].shape == (32, 1)
    assert result["keep_diff"].shape == (32, 1)
    with pytest.raises(ValueError, match="TOP-K"):
        build_ebblayer_priors(ul, ayya, qg, n_layers=2)


def test_input_shape_validation():
    ul, ayya, qg = _sample()
    with pytest.raises(ValueError, match="AYYA shape"):
        build_ebblayer_priors(ul, ayya[:3, :3], qg, n_layers=3)
    with pytest.raises(ValueError, match="smoothing matrix"):
        build_ebblayer_priors(ul, ayya, np.eye(ul.shape[1]), n_layers=3)


def test_diff_recovery_from_catastrophic_cancellation():
    """Recover positive DIFF evidence from near-identical layer leadfields."""
    from lameg.ebblayer import _recover_underfilled_diff, _select_topk

    n_modes = 6
    b = np.zeros((3, 2, n_modes))
    baseline = np.linspace(0.5, 1.5, n_modes)
    b[:, 0, :] = baseline
    b[1, 0, 0] += 1e-8
    b[2, 0, 1] += 2e-8
    invcov = np.eye(n_modes) * 1e-6
    scores = np.zeros((2, 3))
    pairs = [(0, 1), (0, 2), (1, 2)]
    selected = _select_topk(scores, 2)
    scores, selected = _recover_underfilled_diff(
        scores, selected, b, invcov, pairs, 2)

    assert np.all(scores[0] > 0)
    assert np.count_nonzero(selected[0]) == 2
    assert not np.any(selected[1])  # Truly empty patches remain excluded.
