"""
Unit tests for source-space HDF5 export in ``lameg.source``.
"""

import types

import h5py
import numpy as np
import pytest

import lameg.source as source_api


def _make_test_surface_set(
    n_layers=2,
    n_columns=3,
    layer_spacing=None,
    subj_id="sub-test",
):
    """Create a minimal LayerSurfaceSet stand-in for export tests."""
    if layer_spacing is None:
        layer_spacing = np.linspace(
            1.0,
            0.0,
            n_layers,
        )

    surf_set = types.SimpleNamespace(
        n_layers=n_layers,
        n_columns=n_columns,
        layer_spacing=np.asarray(
            layer_spacing,
            dtype=float,
        ),
        subj_id=subj_id,
        requested_orientation=None,
        requested_fixed=None,
    )

    def get_vertices_per_layer(
        orientation="link_vector",
        fixed=True,
    ):
        surf_set.requested_orientation = orientation
        surf_set.requested_fixed = fixed
        return surf_set.n_columns

    surf_set.get_vertices_per_layer = get_vertices_per_layer
    return surf_set


def _setup_export_mocks(
    monkeypatch,
    n_sources=6,
    vertices=None,
):
    """Mock a small deterministic sliding-window source inversion."""
    sensor_data = np.array(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            [
                5.0,
                6.0,
                7.0,
                8.0,
            ],
        ],
        dtype=float,
    )

    time_ms = np.array(
        [
            0.0,
            10.0,
            20.0,
            30.0,
        ],
        dtype=float,
    )

    u_matrix = np.eye(
        2,
        dtype=float,
    )

    m_matrix = np.arange(
        n_sources * 2,
        dtype=float,
    ).reshape(
        n_sources,
        2,
    )

    woi = np.array(
        [
            [
                0.0,
                30.0,
            ],
        ],
        dtype=float,
    )

    invc = {
        "U": u_matrix,
        "woi": woi,
        # Only the presence of a sliding-window key
        # matters because the loaders themselves are mocked.
        "M_win": True,
    }

    def load_full(window_idx):
        assert window_idx == 0
        return m_matrix

    def load_rows(
        window_idx,
        source_indices,
    ):
        assert window_idx == 0

        return m_matrix[
            source_indices,
            :,
        ]

    loaders = {
        "n_windows": 1,
        "n_sources": n_sources,
        "load_full": load_full,
        "load_rows": load_rows,
    }

    if vertices is None:
        vertices = np.arange(
            n_sources * 3,
            dtype=float,
        ).reshape(
            n_sources,
            3,
        )

    monkeypatch.setattr(
        "lameg._source_export.load_meg_sensor_data",
        lambda _fname: (
            sensor_data,
            time_ms,
            None,
        ),
    )

    monkeypatch.setattr(
        "lameg._source_export.check_inversion_exists",
        lambda _fname, inversion_idx=0: True,
    )

    monkeypatch.setattr(
        "lameg._source_export._load_inverse_components",
        lambda _fname, inversion_idx=0: invc,
    )

    monkeypatch.setattr(
        "lameg._source_export._make_window_matrix_loaders",
        lambda _invc, _inv_fname, _data_fname: loaders,
    )

    monkeypatch.setattr(
        "lameg._source_export.load_forward_model_vertices",
        lambda _fname, inversion_idx=0: vertices,
    )

    monkeypatch.setattr(
        "lameg._source_export._get_lameg_version",
        lambda: "test-version",
    )

    expected_flat = (
        m_matrix
        @ sensor_data
    )

    return {
        "sensor_data": sensor_data,
        "time": time_ms,
        "m_matrix": m_matrix,
        "vertices": vertices,
        "expected_flat": expected_flat,
    }


def test_export_source_time_series_hdf5(
    tmp_path,
    monkeypatch,
):
    """Export a valid schema-v1 laminar source file."""
    surf_set = _make_test_surface_set(
        n_layers=2,
        n_columns=3,
        layer_spacing=[
            1.0,
            0.0,
        ],
    )

    expected = _setup_export_mocks(
        monkeypatch,
        n_sources=6,
    )

    out_fname = (
        tmp_path
        / "source_export.h5"
    )

    result = (
        source_api.export_source_time_series_hdf5(
            "data.mat",
            out_fname,
            surf_set,
            compression=None,
        )
    )

    assert (
        result
        == str(
            out_fname.resolve()
        )
    )

    assert (
        surf_set.requested_orientation
        == "link_vector"
    )

    assert (
        surf_set.requested_fixed
        is True
    )

    expected_source = (
        expected[
            "expected_flat"
        ].reshape(
            2,
            3,
            4,
        )
    )

    with h5py.File(
        out_fname,
        "r",
    ) as source_h5:
        np.testing.assert_allclose(
            source_h5[
                "source_ts"
            ][()],
            expected_source,
        )

        np.testing.assert_allclose(
            source_h5[
                "time_ms"
            ][()],
            expected[
                "time"
            ],
        )

        np.testing.assert_allclose(
            source_h5[
                "layer_depth"
            ][()],
            np.array(
                [
                    0.0,
                    1.0,
                ]
            ),
        )

        np.testing.assert_allclose(
            source_h5[
                "source_vertices"
            ][()],
            expected[
                "vertices"
            ].reshape(
                2,
                3,
                3,
            ),
        )

        assert (
            source_h5.attrs[
                "lameg_schema_version"
            ]
            == "1.0"
        )

        assert (
            source_h5.attrs[
                "lameg_version"
            ]
            == "test-version"
        )

        assert (
            source_h5.attrs[
                "subject_id"
            ]
            == "sub-test"
        )

        assert (
            source_h5.attrs[
                "orientation_method"
            ]
            == "link_vector"
        )

        assert bool(
            source_h5.attrs[
                "fixed_orientation"
            ]
        )

        assert (
            source_h5.attrs[
                "source_geometry"
            ]
            == "LayerSurfaceSet"
        )

        assert (
            source_h5.attrs[
                "source_order"
            ]
            == "layer-major"
        )

        assert (
            source_h5.attrs[
                "axis_order"
            ]
            == "layer,column,time"
        )

        assert (
            int(
                source_h5.attrs[
                    "n_layers"
                ]
            )
            == 2
        )

        assert (
            int(
                source_h5.attrs[
                    "n_columns"
                ]
            )
            == 3
        )

        assert (
            source_h5[
                "layer_depth"
            ].attrs[
                "depth_convention"
            ]
            == "0=pial,1=white"
        )

    # Integration check: exporter output must immediately
    # satisfy the public source-file reader.
    with source_api.LaminarSourceData(
        out_fname
    ) as source:
        assert (
            source.shape
            == (
                2,
                3,
                4,
            )
        )

        np.testing.assert_allclose(
            source.layer_depth,
            [
                0.0,
                1.0,
            ],
        )

        np.testing.assert_allclose(
            source.layer(),
            expected_source,
        )


def test_export_source_time_series_geometry_mismatch(
    tmp_path,
    monkeypatch,
):
    """Inverse source count must exactly match LayerSurfaceSet geometry."""
    surf_set = _make_test_surface_set(
        n_layers=2,
        n_columns=3,
    )

    _setup_export_mocks(
        monkeypatch,
        n_sources=5,
    )

    out_fname = (
        tmp_path
        / "bad_geometry.h5"
    )

    with pytest.raises(
        ValueError,
        match="Source geometry mismatch",
    ):
        source_api.export_source_time_series_hdf5(
            "data.mat",
            out_fname,
            surf_set,
            compression=None,
        )


def test_export_source_time_series_vertex_geometry_mismatch(
    tmp_path,
    monkeypatch,
):
    """Forward-model vertex count must match LayerSurfaceSet geometry."""
    surf_set = _make_test_surface_set(
        n_layers=2,
        n_columns=3,
    )

    bad_vertices = np.zeros(
        (
            5,
            3,
        ),
        dtype=float,
    )

    _setup_export_mocks(
        monkeypatch,
        n_sources=6,
        vertices=bad_vertices,
    )

    out_fname = (
        tmp_path
        / "bad_vertices.h5"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Forward-model vertex geometry "
            "does not match"
        ),
    ):
        source_api.export_source_time_series_hdf5(
            "data.mat",
            out_fname,
            surf_set,
            compression=None,
        )


@pytest.mark.parametrize(
    "layer_spacing",
    [
        [
            1.0,
            0.5,
            0.5,
        ],
        [
            1.0,
            0.5,
            1.2,
        ],
        [
            0.9,
            0.5,
            0.0,
        ],
        [
            1.0,
            0.5,
            0.1,
        ],
        [
            1.0,
            np.nan,
            0.0,
        ],
    ],
)
def test_export_source_time_series_rejects_invalid_layer_geometry(
    tmp_path,
    monkeypatch,
    layer_spacing,
):
    """Reject invalid LayerSurfaceSet cortical-depth geometry."""
    surf_set = _make_test_surface_set(
        n_layers=3,
        n_columns=2,
        layer_spacing=layer_spacing,
    )

    _setup_export_mocks(
        monkeypatch,
        n_sources=6,
    )

    out_fname = (
        tmp_path
        / "bad_layer_geometry.h5"
    )

    with pytest.raises(
        ValueError
    ):
        source_api.export_source_time_series_hdf5(
            "data.mat",
            out_fname,
            surf_set,
            compression=None,
        )


def test_export_source_time_series_uses_requested_surface_variant(
    tmp_path,
    monkeypatch,
):
    """Orientation and fixed state should define source geometry."""
    surf_set = _make_test_surface_set(
        n_layers=2,
        n_columns=3,
    )

    _setup_export_mocks(
        monkeypatch,
        n_sources=6,
    )

    out_fname = (
        tmp_path
        / "surface_variant.h5"
    )

    source_api.export_source_time_series_hdf5(
        "data.mat",
        out_fname,
        surf_set,
        orientation="ds_surf_norm",
        fixed=False,
        compression=None,
    )

    assert (
        surf_set.requested_orientation
        == "ds_surf_norm"
    )

    assert (
        surf_set.requested_fixed
        is False
    )

    with h5py.File(
        out_fname,
        "r",
    ) as source_h5:
        assert (
            source_h5.attrs[
                "orientation_method"
            ]
            == "ds_surf_norm"
        )

        assert not bool(
            source_h5.attrs[
                "fixed_orientation"
            ]
        )


def test_export_source_time_series_uses_inverse_file_vertices(
    tmp_path,
    monkeypatch,
):
    """Forward-model vertices should come from inv_fname when supplied."""
    surf_set = _make_test_surface_set(
        n_layers=2,
        n_columns=3,
    )

    _setup_export_mocks(
        monkeypatch,
        n_sources=6,
    )

    received = {}

    def fake_load_forward_model_vertices(
        fname,
        inversion_idx=0,
    ):
        received[
            "fname"
        ] = fname

        received[
            "inversion_idx"
        ] = inversion_idx

        return np.arange(
            18,
            dtype=float,
        ).reshape(
            6,
            3,
        )

    monkeypatch.setattr(
        (
            "lameg._source_export."
            "load_forward_model_vertices"
        ),
        fake_load_forward_model_vertices,
    )

    out_fname = (
        tmp_path
        / "separate_inverse.h5"
    )

    source_api.export_source_time_series_hdf5(
        "sensor_data.mat",
        out_fname,
        surf_set,
        inv_fname="inverse.mat",
        inversion_idx=2,
        compression=None,
    )

    assert (
        received[
            "fname"
        ]
        == "inverse.mat"
    )

    assert (
        received[
            "inversion_idx"
        ]
        == 2
    )
