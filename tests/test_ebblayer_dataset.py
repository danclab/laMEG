"""No-MATLAB unit tests for dataset-level Python EBBlayer integration."""

import numpy as np
import pytest
from scipy.sparse import eye

import lameg.ebblayer_dataset as dataset_module
from lameg.ebblayer_inversion import PreparedEBBlayerData


def _dummy_prepared(n_layers):
    n_sources = 6 * n_layers
    rng = np.random.RandomState(7)
    x = rng.randn(3, n_sources)
    return PreparedEBBlayerData(
        ul=x, ayya=np.eye(3), qe=np.eye(3) / 3,
        q0=0.01 * np.eye(3), n_samples=16,
        qg=eye(n_sources, format="csc"), n_layers=n_layers)


def test_two_layers_default_topk_and_passthrough(monkeypatch):
    calls = []
    result_object = object()

    def fake_prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return _dummy_prepared(kwargs["n_layers"])

    def fake_invert(prepared, **kwargs):
        calls.append(("invert", kwargs))
        assert isinstance(prepared, PreparedEBBlayerData)
        return result_object

    monkeypatch.setattr(dataset_module, "prepare_ebblayer_from_spm", fake_prepare)
    monkeypatch.setattr(dataset_module, "invert_ebb_layer_python", fake_invert)

    result = dataset_module.invert_ebb_layer_from_spm(
        "data.mat", "QG.mat", 2, foi=(0, 48), n_spatial_modes=40,
        n_temp_modes=4, runtime_dir="/runtime")

    assert result is result_object
    assert calls[0][0] == "prepare"
    assert calls[0][1]["mode"] == "compute"
    assert calls[0][1]["foi"] == (0, 48)
    assert calls[0][1]["n_spatial_modes"] == 40
    assert calls[0][1]["runtime_dir"] == "/runtime"
    assert calls[0][1]["return_metadata"] is False
    assert calls[1][1]["sum_pair_topk"] == 1
    assert calls[1][1]["diff_pair_topk"] == 1
    assert calls[1][1]["runtime_dir"] == "/runtime"


def test_eleven_layers_topk_and_saved_mode_metadata(monkeypatch):
    result_object = object()
    info = {"temporal_projector": np.eye(4), "mode": "saved"}

    def fake_prepare(**kwargs):
        assert kwargs["mode"] == "saved"
        assert kwargs["return_metadata"] is True
        return _dummy_prepared(kwargs["n_layers"]), info

    def fake_invert(prepared, **kwargs):
        assert prepared.n_layers == 11
        assert kwargs["sum_pair_topk"] == 2
        assert kwargs["diff_pair_topk"] == 2
        assert kwargs["return_priors"] is True
        return result_object

    monkeypatch.setattr(dataset_module, "prepare_ebblayer_from_spm", fake_prepare)
    monkeypatch.setattr(dataset_module, "invert_ebb_layer_python", fake_invert)

    result, meta = dataset_module.invert_ebb_layer_from_spm(
        "data.mat", "QG.mat", 11, mode="saved",
        return_metadata=True, return_priors=True)
    assert result is result_object
    assert meta is info


def test_explicit_topk_forwarded(monkeypatch):
    def fake_prepare(**kwargs):
        return _dummy_prepared(kwargs["n_layers"])

    def fake_invert(prepared, **kwargs):
        return (kwargs["sum_pair_topk"], kwargs["diff_pair_topk"])

    monkeypatch.setattr(dataset_module, "prepare_ebblayer_from_spm", fake_prepare)
    monkeypatch.setattr(dataset_module, "invert_ebb_layer_python", fake_invert)
    assert dataset_module.invert_ebb_layer_from_spm(
        "data.mat", "kernel.mat", 4, sum_pair_topk=3,
        diff_pair_topk=6) == (3, 6)


@pytest.mark.parametrize("bad_layers", [0, 1, 2.3, True, None])
def test_invalid_layers_rejected_before_spm(bad_layers, monkeypatch):
    monkeypatch.setattr(dataset_module, "prepare_ebblayer_from_spm",
                        lambda **kw: pytest.fail("Unexpected SPM preparation"))
    with pytest.raises(ValueError, match="n_layers"):
        dataset_module.invert_ebb_layer_from_spm(
            "data.mat", "kernel.mat", bad_layers)


@pytest.mark.parametrize("name,bad", [
    ("sum_pair_topk", 0),
    ("diff_pair_topk", -1),
    ("sum_pair_topk", 2.0),
    ("diff_pair_topk", True),
    ("sum_pair_topk", 2),    # 2-layer mesh has just one pair
    ("diff_pair_topk", 55),
])
def test_invalid_topk_rejected_before_spm(name, bad, monkeypatch):
    monkeypatch.setattr(dataset_module, "prepare_ebblayer_from_spm",
                        lambda **kw: pytest.fail("Unexpected SPM preparation"))
    with pytest.raises(ValueError, match=name):
        dataset_module.invert_ebb_layer_from_spm(
            "data.mat", "kernel.mat", 2, **{name: bad})


def test_no_dataset_mutation_or_default_backend_change(monkeypatch):
    calls = []
    monkeypatch.setattr(dataset_module, "prepare_ebblayer_from_spm",
                        lambda **kw: (_dummy_prepared(2)))
    monkeypatch.setattr(dataset_module, "invert_ebb_layer_python",
                        lambda prepared, **kw: calls.append(kw) or "in-memory")
    assert dataset_module.invert_ebb_layer_from_spm("data.mat", "QG.mat", 2) == "in-memory"
    assert len(calls) == 1


def test_full_dataset_to_operator_without_matlab(tmp_path):
    """Exercise actual preparation -> priors -> both ReML calls -> M."""
    import re
    import h5py
    from scipy.io import savemat

    dataset = tmp_path / "meg.mat"
    dataset.write_bytes(b"SPM dummy; substituted by runner")
    kernel = tmp_path / "kernel.mat"
    q = eye(6, format="csc")
    with h5py.File(str(kernel), "w") as f:
        g = f.create_group("QG")
        g.create_dataset("data", data=q.data)
        g.create_dataset("ir", data=q.indices)
        g.create_dataset("jc", data=q.indptr)

    def fake_spm(code):
        match = re.search(r"save\('([^']+)','UL','AYYA'", code)
        assert match is not None
        savemat(match.group(1), {
            "UL": np.array([[1., 0., .2, 0., -.3, 0.],
                            [0., 1., .2, -.2, 0., .1],
                            [.2, .1, 1., 0., .2, -.2]]),
            "AYYA": np.eye(3) * 8.,
            "Qe": np.eye(3) / 3.,
            "Q0": np.eye(3) * 0.05,
            "Nn": 24.,
            "A": np.eye(3),
            "S": np.eye(4, 2),
            "Ic": [1, 2, 3],
            "It": [1, 2, 3, 4],
            "Ik": [1, 2, 3],
        })

    reml_calls = []

    def fake_reml(ayya, components, n_samples, fixed_covariance,
                  runtime_dir=None):
        reml_calls.append(len(components))
        assert n_samples == 24
        if len(components) == 4:
            return {"hyperparameters": [0.1, 1.0, .5, .25],
                    "covariance": np.eye(3), "free_energy": -15.}
        return {"hyperparameters": [0.1, .75],
                "covariance": np.eye(3) * 2., "free_energy": -14.}

    result, meta = dataset_module.invert_ebb_layer_from_spm(
        dataset, kernel, 2, n_spatial_modes=3, n_temp_modes=2,
        eval_runner=fake_spm, reml_runner=fake_reml,
        return_metadata=True)
    assert reml_calls == [4, 2]
    assert result.operator.shape == (6, 3)
    assert result.posterior_variance.shape == (6,)
    assert np.all(np.isfinite(result.operator))
    assert np.isclose(result.free_energy, -14.)
    assert meta["temporal_projector"].shape == (4, 2)
