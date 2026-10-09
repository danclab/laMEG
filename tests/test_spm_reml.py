"""MAT-file boundary and validation tests for the official SPM ReML adapter.

They do not require MATLAB Runtime or spm-python and run on Python 3.7.
"""
import re

import numpy as np
import pytest
from scipy.io import loadmat, savemat
from scipy.sparse import csc_matrix

from lameg.spm_reml import run_reml


def _fake_matlab(command):
    """Emulate the on-disk MATLAB boundary and inspect covariance ordering."""
    matched = re.search(r"S=load\('([^']+)'\)", command)
    saved = re.search(r"save\('([^']+)'", command)
    assert matched and saved
    data = loadmat(matched.group(1))
    assert np.allclose(data["AYYA"], np.diag([5., 6.]))
    assert np.allclose(data["V"], np.diag([0.01, 0.02]))
    assert np.allclose(data["Q001"], np.eye(2))
    assert np.allclose(data["Q002"], np.array([[1., 0.5], [0.5, 1.]]))
    assert float(data["Nn"].item()) == 10.
    assert "spm_reml_sc(S.AYYA,[],Q,S.Nn,-4,16,S.V)" in command
    # Sparse 'h' tests the MATLAB sparse-output compatibility path.
    savemat(saved.group(1), {
        "C": np.array([[1., 0.1], [0.1, 2.]]),
        "h": csc_matrix([[0.6], [0.4]]),
        "F": -42.25,
    })


def _inputs():
    return (np.diag([5., 6.]),
            [np.eye(2), np.array([[1., 0.5], [0.5, 1.]])],
            10, np.diag([0.01, 0.02]))


def test_reml_serializes_components_in_order_and_reads_sparse_h():
    result = run_reml(*_inputs(), eval_runner=_fake_matlab)
    assert np.allclose(result["hyperparameters"], [0.6, 0.4])
    assert np.allclose(result["covariance"], [[1., 0.1], [0.1, 2.]])
    assert result["free_energy"] == -42.25


@pytest.mark.parametrize("argument,replace,match", [
    ("ayya", np.ones((2, 3)), "ayya must be square"),
    ("components", [], "nonempty"),
    ("components", [np.eye(3)], "component 0"),
    ("n_samples", 0, "n_samples"),
    ("fixed_covariance", np.eye(3), "fixed_covariance"),
])
def test_invalid_inputs_rejected_before_runtime(argument, replace, match):
    names = ["ayya", "components", "n_samples", "fixed_covariance"]
    inputs = dict(zip(names, _inputs()))
    inputs[argument] = replace
    with pytest.raises(ValueError, match=match):
        run_reml(**inputs, eval_runner=lambda _: pytest.fail("runner invoked"))


def test_missing_output_raises_and_temporary_files_are_cleaned():
    with pytest.raises(RuntimeError, match="did not write"):
        run_reml(*_inputs(), eval_runner=lambda _: None)


def test_validated_matrices_may_be_sparse():
    ayya, qs, n, fixed = _inputs()
    result = run_reml(csc_matrix(ayya), [csc_matrix(q) for q in qs],
                      n, csc_matrix(fixed), eval_runner=_fake_matlab)
    assert result["free_energy"] == -42.25
