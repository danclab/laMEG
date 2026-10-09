"""Unit tests for the SPM-prepared, Python EBBlayer inversion workflow.

All tests run without MATLAB Runtime or spm-python and work with NumPy/SciPy
versions currently used by laMEG. Large DANC real-data regressions remain
outside the regular test suite.
"""

import numpy as np
import pytest
from scipy.sparse import eye, issparse

from lameg.ebblayer import build_ebblayer_priors
from lameg.ebblayer_inversion import (
    PreparedEBBlayerData, invert_ebb_layer_python)


def _prepared():
    rng = np.random.RandomState(21)
    ul = rng.randn(5, 3 * 12)
    a = rng.randn(5, 5)
    ayya = a @ a.T + 2 * np.eye(5)
    qe = np.eye(5) / 5
    q0 = np.eye(5) * 0.1
    return PreparedEBBlayerData(
        ul=ul, ayya=ayya, qg=eye(36, format="csc"),
        qe=qe, q0=q0, n_samples=20, n_layers=3)


def test_posterior_matches_dense_reference_and_retains_cross_layer_covariance():
    """Confirm both ReML layouts, the full covariance, M and Cq algebra."""
    prepared = _prepared()
    calls = []

    def fake_reml(ayya, components, n_samples, fixed_covariance,
                  runtime_dir=None):
        calls.append((ayya, components, n_samples, fixed_covariance,
                      runtime_dir))
        if len(calls) == 1:
            return dict(hyperparameters=np.array([0.1, 0.9, 1.2, 0.3]),
                        covariance=np.eye(5), free_energy=-100.0)
        return dict(hyperparameters=np.array([0.2, 0.75]),
                    covariance=2.5 * np.eye(5), free_energy=-90.0)

    result = invert_ebb_layer_python(
        prepared, reml_runner=fake_reml, runtime_dir="/fake/runtime",
        sum_pair_topk=2, diff_pair_topk=2, return_priors=True)
    assert len(calls) == 2
    assert all(c[0] is prepared.ayya and c[2] == 20 and
               c[3] is prepared.q0 and c[4] == "/fake/runtime" for c in calls)
    assert np.allclose(calls[0][1][0], prepared.qe)
    assert np.allclose(calls[1][1][0], prepared.q0)
    assert np.allclose(result.source_weights, [0.9, 1.2, 0.3])
    assert result.source_free_energy == -100.0
    assert result.free_energy == -90.0
    assert result.final_source_scale == 0.75
    assert result.operator.shape == (36, 5)
    assert result.posterior_variance.shape == (36,)
    assert issparse(result.source_covariance)
    assert result.priors is not None

    priors = build_ebblayer_priors(prepared.ul, prepared.ayya, prepared.qg, 3)
    qp = sum((w * q.toarray() for w, q in zip(result.source_weights,
                                               priors["source_q"])))
    assert np.any(np.abs(qp - np.diag(np.diag(qp))) > 1e-12)
    assert np.allclose(result.source_covariance.toarray(), qp)
    assert np.allclose(calls[0][1][1], priors["sensor_q"][0])
    assert np.allclose(calls[0][1][2], priors["sensor_q"][1])
    assert np.allclose(calls[0][1][3], priors["sensor_q"][2])
    assert np.allclose(calls[1][1][1], prepared.ul @ qp @ prepared.ul.T)

    cy = 2.5 * np.eye(5)
    lc = 0.75 * prepared.ul @ qp
    m_expected = lc.T @ np.linalg.inv(cy)
    cq_expected = 0.75 * np.diag(qp) - np.einsum("ij,ji->j", lc, m_expected)
    assert np.allclose(result.operator, m_expected)
    assert np.allclose(result.posterior_variance, cq_expected)
    assert result.n_sum_retained == np.count_nonzero(priors["keep_sum"])
    assert result.n_diff_retained == np.count_nonzero(priors["keep_diff"])
    assert result.n_sum_endpoints == np.count_nonzero(priors["endpoint"])


def test_optional_diagnostics_are_not_retained_by_default():
    prepared = _prepared()

    def fake_reml(ayya, components, n_samples, fixed_covariance,
                  runtime_dir=None):
        h = ([1, 1, 1, 1] if len(components) == 4 else [1, 1])
        return dict(hyperparameters=h, covariance=np.eye(5), free_energy=-1)

    result = invert_ebb_layer_python(prepared, reml_runner=fake_reml)
    assert result.priors is None


@pytest.mark.parametrize("field,value,match", [
    ("ul", np.ones((5, 34)), "n_layers"),
    ("ayya", np.eye(4), "ayya must have shape"),
    ("qe", np.eye(4), "qe must have shape"),
    ("q0", np.eye(4), "q0 must have shape"),
    ("n_samples", 0, "n_samples"),
    ("n_layers", 1, "n_layers"),
    ("qg", np.eye(36), "sparse"),
])
def test_prepared_input_validation(field, value, match):
    prepared = _prepared()
    args = dict(prepared.__dict__)
    args[field] = value
    with pytest.raises(ValueError, match=match):
        PreparedEBBlayerData(**args)


def test_invalid_first_stage_reml_result_rejected():
    def invalid_runner(ayya, components, n_samples, fixed_covariance,
                       runtime_dir=None):
        return dict(hyperparameters=[0.5, 1.0], covariance=np.eye(5),
                    free_energy=-1.0)

    with pytest.raises(RuntimeError, match="First-stage"):
        invert_ebb_layer_python(_prepared(), reml_runner=invalid_runner)


def test_singular_second_stage_sensor_covariance_rejected():
    def singular_runner(ayya, components, n_samples, fixed_covariance,
                        runtime_dir=None):
        return dict(hyperparameters=[0.2] + [1.0] * (len(components) - 1),
                    covariance=np.zeros((5, 5)), free_energy=-1.0)

    with pytest.raises(RuntimeError, match="singular"):
        invert_ebb_layer_python(_prepared(), reml_runner=singular_runner)
