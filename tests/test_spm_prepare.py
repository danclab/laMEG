"""Offline boundary tests for the SPM -> EBBlayer preparation adapter.

No installed MATLAB, spm-python, or large real-data files needed.
"""

import re

import h5py
import numpy as np
import pytest
from scipy.io import savemat
from scipy.sparse import csc_matrix, eye

from lameg.spm_prepare import prepare_ebblayer_from_spm, _load_cached_kernel


def _matlab_kernel(path, matrix=None):
    if matrix is None:
        matrix = eye(4, format="csc")
    q = matrix.tocsc()
    with h5py.File(str(path), "w") as f:
        g = f.create_group("QG")
        g.create_dataset("data", data=q.data)
        g.create_dataset("ir", data=q.indices.astype(np.int64))
        g.create_dataset("jc", data=q.indptr.astype(np.int64))


def _fixture(tmp_path):
    dataset = tmp_path / "test-data.mat"
    dataset.write_bytes(b"test")
    kernel = tmp_path / "kernel.mat"
    _matlab_kernel(kernel)
    return dataset, kernel


def _fake_runner(captured):
    def callback(code):
        captured.append(code)
        m = re.search(r"save\('([^']+)','UL','AYYA'", code)
        assert m is not None
        savemat(m.group(1), {
            "UL": np.arange(8, dtype=float).reshape(2, 4) * 0.1,
            "AYYA": np.array([[4., 1.], [1., 3.]]),
            "Qe": np.eye(2) / 2.,
            "Q0": np.eye(2) * 0.03,
            "Nn": 24.,
            "A": np.eye(2),
            "S": np.eye(3, 2),
            "Ic": np.array([1, 3]),
            "It": np.array([10, 11, 12]),
            "Ik": np.array([1, 2, 3, 4]),
        })
    return callback


def test_saved_projectors_and_metadata(tmp_path):
    dataset, kernel = _fixture(tmp_path)
    captured = []
    data, meta = prepare_ebblayer_from_spm(
        dataset, kernel, 2, mode="saved", eval_runner=_fake_runner(captured),
        return_metadata=True)
    assert data.ul.shape == (2, 4)
    assert data.qg.nnz == 4
    assert data.n_samples == 24
    assert np.allclose(data.ayya, [[4., 1.], [1., 3.]])
    assert np.allclose(data.qe, np.eye(2) / 2)
    assert meta["mode"] == "saved"
    assert np.array_equal(meta["channels"], [1, 3])
    assert "UL=full(inv.L)" in captured[0]
    assert "[L,D]=spm_eeg_lgainmat(D)" not in captured[0]


def test_compute_projectors_from_coregistered_data(tmp_path):
    dataset, kernel = _fixture(tmp_path)
    captured = []
    result = prepare_ebblayer_from_spm(
        dataset, kernel, 2, mode="compute", n_spatial_modes=2,
        n_temp_modes=2, foi=(0, 48), eval_runner=_fake_runner(captured))
    assert result.ul.shape == (2, 4)
    code = captured[0]
    assert "[L,D]=spm_eeg_lgainmat(D)" in code
    assert "spm_dctmtx" in code
    assert "[Uv,~]=svd(YTY)" in code
    assert "[Usp,~,~]=spm_svd(L*L',1e-12)" in code
    assert "D.inv{val}.mesh.tess_mni.face" in code
    assert "AYYA=AYYA+Y*Y'" in code
    assert "Nn=numel(Ik)*size(S,2)" in code


def test_reuse_spatial_modes_file_in_compute_mode(tmp_path):
    dataset, kernel = _fixture(tmp_path)
    modes = tmp_path / "test-data_testmodes.mat"
    modes.write_bytes(b"fake")
    captured = []
    _, metadata = prepare_ebblayer_from_spm(
        dataset, kernel, 2, n_spatial_modes=2, eval_runner=_fake_runner(captured),
        return_metadata=True)
    assert "sm=load" in captured[0]
    assert "Spatial-mode channel ordering" in captured[0]
    assert metadata["spatial_modes_file"] == str(modes.resolve())
    assert "[Usp,~,~]=spm_svd" not in captured[0]


def test_explicit_missing_modes_file_fails(tmp_path):
    dataset, kernel = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="Spatial modes"):
        prepare_ebblayer_from_spm(
            dataset, kernel, 2,
            spatial_modes_file=tmp_path / "missing.mat",
            eval_runner=lambda _: None)


def test_no_output_raises(tmp_path):
    dataset, kernel = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="did not produce"):
        prepare_ebblayer_from_spm(
            dataset, kernel, 2, mode="saved", eval_runner=lambda _: None)


@pytest.mark.parametrize("keyword,value,pattern", [
    ("mode", "unexpected", "mode"),
    ("n_layers", 1, "n_layers"),
    ("n_layers", 2.5, "n_layers"),
    ("n_spatial_modes", 0, "n_spatial_modes"),
    ("n_temp_modes", 0, "n_temp_modes"),
    ("inversion_idx", -1, "inversion_idx"),
    ("foi", (20, 10), "foi"),
    ("woi", (100, 100), "woi"),
    ("noise_floor", -0.1, "noise_floor"),
])
def test_invalid_options(keyword, value, pattern, tmp_path):
    dataset, kernel = _fixture(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        args = {"n_layers": 2, "eval_runner": lambda _: None}
        args[keyword] = value
        prepare_ebblayer_from_spm(dataset, kernel, **args)


def test_missing_dataset(tmp_path):
    _, kernel = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="dataset"):
        prepare_ebblayer_from_spm(tmp_path / "absent.mat", kernel, 2,
                                   eval_runner=lambda _: None)


def test_incompatible_kernel_count(tmp_path):
    dataset, kernel = _fixture(tmp_path)
    _matlab_kernel(kernel, eye(3, format="csc"))
    with pytest.raises(ValueError, match="divisible"):
        prepare_ebblayer_from_spm(dataset, kernel, 2,
                                   eval_runner=lambda _: None)


def test_kernel_preserves_cross_source_entries(tmp_path):
    kernel = tmp_path / "sparse.mat"
    values = csc_matrix(np.array([[1., .2, 0., 0.],
                                  [.2, 1., 0., 0.],
                                  [0., 0., 1., -.7],
                                  [0., 0., -.7, 1.]]))
    _matlab_kernel(kernel, values)
    actual = _load_cached_kernel(kernel)
    assert actual.nnz == values.nnz
    assert np.array_equal(actual.toarray(), values.toarray())
