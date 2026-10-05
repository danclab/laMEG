"""
Unit tests for BigBrain mapping and laminar access in ``lameg.source``.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from lameg.source import LaminarSourceData, add_bigbrain_mapping
from tests.source_test_utils import make_source_file


def _make_test_surface_set(
    layer_spacing,
    subj_id="sub-test",
):
    """Create a minimal surface-set stand-in for BigBrain mapping tests."""
    return SimpleNamespace(
        layer_spacing=np.asarray(
            layer_spacing,
            dtype=float,
        ),
        subj_id=subj_id,
    )

def test_add_bigbrain_mapping(
    source_file,
    monkeypatch,
):
    """Test writing layer depths and BigBrain mapping datasets."""
    fname, _ = source_file

    surf_set = _make_test_surface_set(
        [1.0, 0.5, 0.0]
    )

    edges = np.tile(
        np.linspace(
            0.0,
            1.0,
            7,
        ),
        (4, 1),
    )
    weights = np.zeros(
        (4, 6, 3),
        dtype=float,
    )
    weights[:, :, 1] = 1.0

    def fake_compute_bigbrain_laminar_weights(
        received_surf_set,
    ):
        assert received_surf_set is surf_set
        return edges, weights

    monkeypatch.setattr(
        "lameg.laminar.compute_bigbrain_laminar_weights",
        fake_compute_bigbrain_laminar_weights,
    )

    result = add_bigbrain_mapping(
        fname,
        surf_set,
    )

    assert result == str(
        fname.resolve()
    )

    with h5py.File(fname, "r") as source_h5:
        np.testing.assert_array_equal(
            source_h5["layer_depth"][()],
            np.array(
                [0.0, 0.5, 1.0]
            ),
        )
        np.testing.assert_array_equal(
            source_h5["bigbrain"]["edges"][()],
            edges,
        )
        np.testing.assert_array_equal(
            source_h5["bigbrain"]["weights"][()],
            weights,
        )

        labels = [
            value.decode("utf-8")
            for value in source_h5[
                "bigbrain"
            ]["labels"][()]
        ]
        assert labels == [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
        ]

        assert (
            source_h5["layer_depth"].attrs[
                "depth_convention"
            ]
            == "0=pial,1=white"
        )
        assert (
            source_h5["bigbrain"]["weights"].attrs[
                "axis_order"
            ]
            == "column,lamina,layer"
        )
        assert (
            source_h5["bigbrain"].attrs[
                "subject_id"
            ]
            == "sub-test"
        )


def test_add_bigbrain_mapping_requires_overwrite(
    source_file,
    monkeypatch,
):
    """Existing mappings should not be replaced accidentally."""
    fname, _ = source_file
    surf_set = _make_test_surface_set(
        [1.0, 0.5, 0.0]
    )

    edges_1 = np.tile(
        np.linspace(
            0.0,
            1.0,
            7,
        ),
        (4, 1),
    )
    weights_1 = np.zeros(
        (4, 6, 3),
        dtype=float,
    )
    weights_1[:, :, 0] = 1.0

    edges_2 = edges_1.copy()
    weights_2 = np.zeros_like(
        weights_1
    )
    weights_2[:, :, 2] = 1.0

    mapping = {
        "edges": edges_1,
        "weights": weights_1,
    }

    def fake_compute_bigbrain_laminar_weights(
        _surf_set,
    ):
        return (
            mapping["edges"],
            mapping["weights"],
        )

    monkeypatch.setattr(
        "lameg.laminar.compute_bigbrain_laminar_weights",
        fake_compute_bigbrain_laminar_weights,
    )

    add_bigbrain_mapping(
        fname,
        surf_set,
    )

    mapping["edges"] = edges_2
    mapping["weights"] = weights_2

    with pytest.raises(
        FileExistsError,
        match="BigBrain mapping already exists",
    ):
        add_bigbrain_mapping(
            fname,
            surf_set,
        )

    with h5py.File(fname, "r") as source_h5:
        np.testing.assert_array_equal(
            source_h5["bigbrain"]["weights"][()],
            weights_1,
        )

    add_bigbrain_mapping(
        fname,
        surf_set,
        overwrite=True,
    )

    with h5py.File(fname, "r") as source_h5:
        np.testing.assert_array_equal(
            source_h5["bigbrain"]["weights"][()],
            weights_2,
        )


def test_add_bigbrain_mapping_dimension_validation(
    source_file,
    monkeypatch,
):
    """Mapping dimensions must match the exported source geometry."""
    fname, _ = source_file

    bad_layer_surf_set = _make_test_surface_set(
        [1.0, 0.75, 0.5, 0.0]
    )

    with pytest.raises(
        ValueError,
        match="Layer-count mismatch",
    ):
        add_bigbrain_mapping(
            fname,
            bad_layer_surf_set,
        )

    surf_set = _make_test_surface_set(
        [1.0, 0.5, 0.0]
    )

    def bad_compute_bigbrain_laminar_weights(
        _surf_set,
    ):
        return (
            np.zeros(
                (5, 7),
                dtype=float,
            ),
            np.zeros(
                (5, 6, 3),
                dtype=float,
            ),
        )

    monkeypatch.setattr(
        "lameg.laminar.compute_bigbrain_laminar_weights",
        bad_compute_bigbrain_laminar_weights,
    )

    with pytest.raises(
        ValueError,
        match="edge dimensions",
    ):
        add_bigbrain_mapping(
            fname,
            surf_set,
        )

    with h5py.File(fname, "r") as source_h5:
        assert "layer_depth" not in source_h5
        assert "bigbrain" not in source_h5


def _add_test_bigbrain_mapping(fname):
    """Add a deterministic BigBrain mapping directly for reader tests."""
    n_columns = 4
    n_layers = 3
    n_laminae = 6

    layer_depth = np.array(
        [0.0, 0.5, 1.0],
        dtype=float,
    )
    edges = np.tile(
        np.linspace(
            0.0,
            1.0,
            n_laminae + 1,
        ),
        (n_columns, 1),
    )

    weights = np.zeros(
        (
            n_columns,
            n_laminae,
            n_layers,
        ),
        dtype=float,
    )

    for column_idx in range(n_columns):
        alpha = 0.1 * (
            column_idx + 1
        )

        weights[
            column_idx,
            0,
        ] = [1.0, 0.0, 0.0]
        weights[
            column_idx,
            1,
        ] = [0.0, 1.0, 0.0]
        weights[
            column_idx,
            2,
        ] = [0.0, 0.0, 1.0]
        weights[
            column_idx,
            3,
        ] = [0.5, 0.5, 0.0]
        weights[
            column_idx,
            4,
        ] = [0.0, 0.5, 0.5]
        weights[
            column_idx,
            5,
        ] = [
            alpha,
            0.5,
            0.5 - alpha,
        ]

    with h5py.File(fname, "r+") as source_h5:
        source_h5.create_dataset(
            "layer_depth",
            data=layer_depth,
        )

        group = source_h5.create_group(
            "bigbrain"
        )
        group.create_dataset(
            "edges",
            data=edges,
        )
        group.create_dataset(
            "weights",
            data=weights,
        )
        group.create_dataset(
            "labels",
            data=np.asarray(
                [
                    "I",
                    "II",
                    "III",
                    "IV",
                    "V",
                    "VI",
                ],
                dtype="S3",
            ),
        )
        group.attrs["n_laminae"] = 6
        group.attrs[
            "depth_convention"
        ] = "0=pial,1=white"

    return {
        "layer_depth": layer_depth,
        "edges": edges,
        "weights": weights,
        "labels": (
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
        ),
    }


def test_laminar_source_data_bigbrain_metadata(
    source_file,
):
    """Test BigBrain metadata exposed by LaminarSourceData."""
    fname, _ = source_file

    with LaminarSourceData(fname) as source:
        assert not source.has_bigbrain_mapping
        assert source.layer_depth is None
        assert source.laminae is None
        assert source.n_laminae == 0
        assert source.bigbrain_edges is None
        assert source.bigbrain_weights is None

    expected = _add_test_bigbrain_mapping(
        fname
    )

    with LaminarSourceData(fname) as source:
        assert source.has_bigbrain_mapping
        assert source.n_laminae == 6
        assert source.laminae == expected[
            "labels"
        ]

        np.testing.assert_allclose(
            source.layer_depth,
            expected["layer_depth"],
        )
        np.testing.assert_allclose(
            source.bigbrain_edges,
            expected["edges"],
        )
        np.testing.assert_allclose(
            source.bigbrain_weights,
            expected["weights"],
        )


def test_laminar_source_data_lamina_indexing(
    source_file,
):
    """Test on-demand BigBrain mapping for layer x column x time data."""
    fname, expected_source = source_file
    mapping = _add_test_bigbrain_mapping(
        fname
    )

    layer_data = expected_source[
        "source"
    ]
    weights = mapping["weights"]

    expected_all = np.einsum(
        "cal,lct->act",
        weights,
        layer_data,
    )

    with LaminarSourceData(fname) as source:
        np.testing.assert_allclose(
            source.lamina(),
            expected_all,
        )

        np.testing.assert_allclose(
            source.lamina(
                column=2,
            ),
            expected_all[:, 2, :],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina="V",
            ),
            expected_all[4, :, :],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina="v",
                column=2,
            ),
            expected_all[4, 2, :],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina=-1,
                column=-1,
            ),
            expected_all[-1, -1, :],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina=slice(1, 5, 2),
                column=slice(0, 4, 2),
                time=(-50.0, 100.0),
            ),
            expected_all[
                1:5:2,
                0:4:2,
                1:5,
            ],
        )


def test_laminar_source_data_lamina_trial_indexing(
    trial_source_file,
):
    """Test BigBrain mapping for files with a trial dimension."""
    fname, expected_source = (
        trial_source_file
    )
    mapping = _add_test_bigbrain_mapping(
        fname
    )

    layer_data = expected_source[
        "source"
    ]
    weights = mapping["weights"]

    expected_all = np.einsum(
        "cal,lctq->actq",
        weights,
        layer_data,
    )

    with LaminarSourceData(fname) as source:
        np.testing.assert_allclose(
            source.lamina(),
            expected_all,
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina="III",
                column=1,
                time=(0.0, 50.0),
                trial=1,
            ),
            expected_all[
                2,
                1,
                2:4,
                1,
            ],
        )

        np.testing.assert_allclose(
            source.lamina(
                column=1,
                trial=0,
            ),
            expected_all[
                :,
                1,
                :,
                0,
            ],
        )


def test_laminar_source_data_lamina_errors(
    source_file,
):
    """Test missing mapping and invalid lamina selectors."""
    fname, _ = source_file

    with LaminarSourceData(fname) as source:
        with pytest.raises(
            ValueError,
            match="no BigBrain mapping",
        ):
            source.lamina(
                lamina="V",
            )

    _add_test_bigbrain_mapping(
        fname
    )

    with LaminarSourceData(fname) as source:
        with pytest.raises(
            ValueError,
            match="Unknown lamina",
        ):
            source.lamina(
                lamina="VII",
            )

        with pytest.raises(
            IndexError,
            match="lamina index",
        ):
            source.lamina(
                lamina=6,
            )


@pytest.mark.parametrize(
    "failure",
    [
        "partial_mapping",
        "missing_weights",
        "bad_layer_depth",
        "bad_edges",
        "bad_weights",
        "bad_n_laminae",
    ],
)
def test_laminar_source_data_bigbrain_schema_validation(
    tmp_path,
    failure,
):
    """Test rejection of malformed stored BigBrain mappings."""
    fname = tmp_path / (
        f"bigbrain_{failure}.h5"
    )
    make_source_file(
        fname
    )

    if failure == "partial_mapping":
        with h5py.File(
            fname,
            "r+",
        ) as source_h5:
            source_h5.create_dataset(
                "layer_depth",
                data=np.array(
                    [0.0, 0.5, 1.0]
                ),
            )

    else:
        _add_test_bigbrain_mapping(
            fname
        )

        with h5py.File(
            fname,
            "r+",
        ) as source_h5:
            if failure == "missing_weights":
                del source_h5[
                    "bigbrain"
                ]["weights"]

            elif failure == "bad_layer_depth":
                del source_h5[
                    "layer_depth"
                ]
                source_h5.create_dataset(
                    "layer_depth",
                    data=np.zeros(4),
                )

            elif failure == "bad_edges":
                group = source_h5[
                    "bigbrain"
                ]
                del group["edges"]
                group.create_dataset(
                    "edges",
                    data=np.zeros(
                        (5, 7)
                    ),
                )

            elif failure == "bad_weights":
                group = source_h5[
                    "bigbrain"
                ]
                del group["weights"]
                group.create_dataset(
                    "weights",
                    data=np.zeros(
                        (4, 6, 4)
                    ),
                )

            elif failure == "bad_n_laminae":
                source_h5[
                    "bigbrain"
                ].attrs[
                    "n_laminae"
                ] = 5

    with pytest.raises(ValueError):
        LaminarSourceData(
            fname
        )
