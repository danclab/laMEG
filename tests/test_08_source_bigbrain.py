"""
Unit tests for BigBrain mapping and laminar access in ``lameg.source``.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from lameg.source import (
    LaminarSourceData,
    add_bigbrain_mapping,
)
from tests.source_test_utils import make_source_file


LAMINA_LABELS = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
)


def _make_test_surface_set(
    layer_spacing,
    subj_id="sub-test",
):
    """Create a minimal surface-set stand-in for BigBrain tests."""
    return SimpleNamespace(
        layer_spacing=np.asarray(
            layer_spacing,
            dtype=float,
        ),
        subj_id=subj_id,
    )


def _make_uniform_bigbrain_mapping(
    n_columns=4,
    n_layers=3,
    active_layer=1,
):
    """Create a simple deterministic BigBrain mapping."""
    edges = np.tile(
        np.linspace(
            0.0,
            1.0,
            7,
        ),
        (
            n_columns,
            1,
        ),
    )

    weights = np.zeros(
        (
            n_columns,
            6,
            n_layers,
        ),
        dtype=float,
    )

    weights[
        :,
        :,
        active_layer,
    ] = 1.0

    return edges, weights


def test_add_bigbrain_mapping(
    source_file,
    monkeypatch,
):
    """Add BigBrain metadata without modifying core layer geometry."""
    fname, expected_source = source_file

    surf_set = _make_test_surface_set(
        [
            1.0,
            0.5,
            0.0,
        ]
    )

    edges, weights = (
        _make_uniform_bigbrain_mapping()
    )

    def fake_compute_bigbrain_laminar_weights(
        received_surf_set,
    ):
        assert received_surf_set is surf_set
        return edges, weights

    monkeypatch.setattr(
        (
            "lameg.laminar."
            "compute_bigbrain_laminar_weights"
        ),
        fake_compute_bigbrain_laminar_weights,
    )

    result = add_bigbrain_mapping(
        fname,
        surf_set,
    )

    assert result == str(
        fname.resolve()
    )

    with h5py.File(
        fname,
        "r",
    ) as source_h5:
        # Core source geometry must remain unchanged.
        np.testing.assert_allclose(
            source_h5[
                "layer_depth"
            ][()],
            expected_source[
                "layer_depth"
            ],
        )

        group = source_h5[
            "bigbrain"
        ]

        np.testing.assert_array_equal(
            group[
                "edges"
            ][()],
            edges,
        )

        np.testing.assert_array_equal(
            group[
                "weights"
            ][()],
            weights,
        )

        labels = tuple(
            value.decode(
                "utf-8"
            )
            for value in group[
                "labels"
            ][()]
        )

        assert (
            labels
            == LAMINA_LABELS
        )

        np.testing.assert_array_equal(
            group[
                "valid_columns"
            ][()],
            np.ones(
                4,
                dtype=bool,
            ),
        )

        assert (
            source_h5[
                "layer_depth"
            ].attrs[
                "depth_convention"
            ]
            == "0=pial,1=white"
        )

        assert (
            group[
                "weights"
            ].attrs[
                "axis_order"
            ]
            == "column,lamina,layer"
        )

        assert (
            group.attrs[
                "subject_id"
            ]
            == "sub-test"
        )

        assert (
            group.attrs[
                "n_laminae"
            ]
            == 6
        )

    # Also verify that the resulting file passes
    # the public reader/schema validation.
    with LaminarSourceData(
        fname
    ) as source:
        assert (
            source.has_bigbrain_mapping
        )

        np.testing.assert_allclose(
            source.layer_depth,
            expected_source[
                "layer_depth"
            ],
        )


def test_add_bigbrain_mapping_requires_overwrite(
    source_file,
    monkeypatch,
):
    """Existing mappings should not be replaced accidentally."""
    fname, expected_source = source_file

    surf_set = _make_test_surface_set(
        [
            1.0,
            0.5,
            0.0,
        ]
    )

    edges_1, weights_1 = (
        _make_uniform_bigbrain_mapping(
            active_layer=0,
        )
    )

    edges_2, weights_2 = (
        _make_uniform_bigbrain_mapping(
            active_layer=2,
        )
    )

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
        (
            "lameg.laminar."
            "compute_bigbrain_laminar_weights"
        ),
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
        match=(
            "BigBrain mapping "
            "already exists"
        ),
    ):
        add_bigbrain_mapping(
            fname,
            surf_set,
        )

    with h5py.File(
        fname,
        "r",
    ) as source_h5:
        np.testing.assert_array_equal(
            source_h5[
                "bigbrain"
            ][
                "weights"
            ][()],
            weights_1,
        )

    add_bigbrain_mapping(
        fname,
        surf_set,
        overwrite=True,
    )

    with h5py.File(
        fname,
        "r",
    ) as source_h5:
        np.testing.assert_array_equal(
            source_h5[
                "bigbrain"
            ][
                "weights"
            ][()],
            weights_2,
        )

        # Overwriting BigBrain data must not
        # alter the source geometry.
        np.testing.assert_allclose(
            source_h5[
                "layer_depth"
            ][()],
            expected_source[
                "layer_depth"
            ],
        )


def test_add_bigbrain_mapping_dimension_validation(
    source_file,
    monkeypatch,
):
    """Mapping dimensions must match exported source geometry."""
    fname, expected_source = source_file

    bad_layer_surf_set = (
        _make_test_surface_set(
            [
                1.0,
                0.75,
                0.5,
                0.0,
            ]
        )
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
        [
            1.0,
            0.5,
            0.0,
        ]
    )

    def bad_compute_bigbrain_laminar_weights(
        _surf_set,
    ):
        return (
            np.zeros(
                (
                    5,
                    7,
                ),
                dtype=float,
            ),
            np.zeros(
                (
                    5,
                    6,
                    3,
                ),
                dtype=float,
            ),
        )

    monkeypatch.setattr(
        (
            "lameg.laminar."
            "compute_bigbrain_laminar_weights"
        ),
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

    with h5py.File(
        fname,
        "r",
    ) as source_h5:
        # Core geometry remains present.
        np.testing.assert_allclose(
            source_h5[
                "layer_depth"
            ][()],
            expected_source[
                "layer_depth"
            ],
        )

        # Failed mapping must not leave a
        # partial BigBrain group behind.
        assert (
            "bigbrain"
            not in source_h5
        )


def test_add_bigbrain_mapping_rejects_layer_depth_mismatch(
    source_file,
    monkeypatch,
):
    """Surface-set depths must agree with stored source geometry."""
    fname, _ = source_file

    # Stored source depth is [0, 0.5, 1].
    # This surface set implies [0, 0.6, 1].
    surf_set = _make_test_surface_set(
        [
            1.0,
            0.4,
            0.0,
        ]
    )

    edges, weights = (
        _make_uniform_bigbrain_mapping()
    )

    monkeypatch.setattr(
        (
            "lameg.laminar."
            "compute_bigbrain_laminar_weights"
        ),
        lambda _surf_set: (
            edges,
            weights,
        ),
    )

    with pytest.raises(
        ValueError,
        match="layer_depth",
    ):
        add_bigbrain_mapping(
            fname,
            surf_set,
        )

    with h5py.File(
        fname,
        "r",
    ) as source_h5:
        assert (
            "bigbrain"
            not in source_h5
        )


def _add_test_bigbrain_mapping(
    fname,
):
    """Add deterministic BigBrain data directly for reader tests."""
    n_columns = 4
    n_layers = 3
    n_laminae = 6

    layer_depth = np.array(
        [
            0.0,
            0.5,
            1.0,
        ],
        dtype=float,
    )

    edges = np.tile(
        np.linspace(
            0.0,
            1.0,
            n_laminae + 1,
        ),
        (
            n_columns,
            1,
        ),
    )

    weights = np.zeros(
        (
            n_columns,
            n_laminae,
            n_layers,
        ),
        dtype=float,
    )

    for column_idx in range(
        n_columns
    ):
        alpha = 0.1 * (
            column_idx + 1
        )

        weights[
            column_idx,
            0,
        ] = [
            1.0,
            0.0,
            0.0,
        ]

        weights[
            column_idx,
            1,
        ] = [
            0.0,
            1.0,
            0.0,
        ]

        weights[
            column_idx,
            2,
        ] = [
            0.0,
            0.0,
            1.0,
        ]

        weights[
            column_idx,
            3,
        ] = [
            0.5,
            0.5,
            0.0,
        ]

        weights[
            column_idx,
            4,
        ] = [
            0.0,
            0.5,
            0.5,
        ]

        weights[
            column_idx,
            5,
        ] = [
            alpha,
            0.5,
            0.5 - alpha,
        ]

    valid_columns = np.ones(
        n_columns,
        dtype=bool,
    )

    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        # layer_depth is core geometry and
        # must already exist.
        np.testing.assert_allclose(
            source_h5[
                "layer_depth"
            ][()],
            layer_depth,
        )

        group = (
            source_h5.create_group(
                "bigbrain"
            )
        )

        edges_ds = (
            group.create_dataset(
                "edges",
                data=edges,
            )
        )
        edges_ds.attrs[
            "axis_order"
        ] = "column,boundary"

        weights_ds = (
            group.create_dataset(
                "weights",
                data=weights,
            )
        )
        weights_ds.attrs[
            "axis_order"
        ] = "column,lamina,layer"

        group.create_dataset(
            "labels",
            data=np.asarray(
                LAMINA_LABELS,
                dtype="S3",
            ),
        )

        group.create_dataset(
            "valid_columns",
            data=valid_columns,
        )

        group.attrs[
            "n_laminae"
        ] = n_laminae

        group.attrs[
            "depth_convention"
        ] = "0=pial,1=white"

        group.attrs[
            "subject_id"
        ] = "sub-test"

    return {
        "layer_depth": (
            layer_depth
        ),
        "edges": edges,
        "weights": weights,
        "valid_columns": (
            valid_columns
        ),
        "labels": (
            LAMINA_LABELS
        ),
    }


def test_laminar_source_data_bigbrain_metadata(
    source_file,
):
    """Test BigBrain metadata exposed by LaminarSourceData."""
    fname, expected_source = (
        source_file
    )

    # Core depth geometry exists even
    # without a BigBrain mapping.
    with LaminarSourceData(
        fname
    ) as source:
        assert (
            not source.has_bigbrain_mapping
        )

        np.testing.assert_allclose(
            source.layer_depth,
            expected_source[
                "layer_depth"
            ],
        )

        assert (
            source.laminae
            is None
        )

        assert (
            source.n_laminae
            == 0
        )

        assert (
            source.bigbrain_edges
            is None
        )

        assert (
            source.bigbrain_weights
            is None
        )

        assert (
            source.bigbrain_valid_columns
            is None
        )

    expected = (
        _add_test_bigbrain_mapping(
            fname
        )
    )

    with LaminarSourceData(
        fname
    ) as source:
        assert (
            source.has_bigbrain_mapping
        )

        assert (
            source.n_laminae
            == 6
        )

        assert (
            source.laminae
            == expected[
                "labels"
            ]
        )

        np.testing.assert_allclose(
            source.layer_depth,
            expected[
                "layer_depth"
            ],
        )

        np.testing.assert_allclose(
            source.bigbrain_edges,
            expected[
                "edges"
            ],
        )

        np.testing.assert_allclose(
            source.bigbrain_weights,
            expected[
                "weights"
            ],
        )

        np.testing.assert_array_equal(
            source.bigbrain_valid_columns,
            expected[
                "valid_columns"
            ],
        )


def test_laminar_source_data_lamina_indexing(
    source_file,
):
    """Test on-demand mapping for layer x column x time data."""
    fname, expected_source = (
        source_file
    )

    mapping = (
        _add_test_bigbrain_mapping(
            fname
        )
    )

    layer_data = expected_source[
        "source"
    ]

    weights = mapping[
        "weights"
    ]

    expected_all = np.einsum(
        "cal,lct->act",
        weights,
        layer_data,
    )

    with LaminarSourceData(
        fname
    ) as source:
        np.testing.assert_allclose(
            source.lamina(),
            expected_all,
        )

        np.testing.assert_allclose(
            source.lamina(
                column=2,
            ),
            expected_all[
                :,
                2,
                :,
            ],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina="V",
            ),
            expected_all[
                4,
                :,
                :,
            ],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina="v",
                column=2,
            ),
            expected_all[
                4,
                2,
                :,
            ],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina=-1,
                column=-1,
            ),
            expected_all[
                -1,
                -1,
                :,
            ],
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina=slice(
                    1,
                    5,
                    2,
                ),
                column=slice(
                    0,
                    4,
                    2,
                ),
                time=(
                    -50.0,
                    100.0,
                ),
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

    mapping = (
        _add_test_bigbrain_mapping(
            fname
        )
    )

    layer_data = expected_source[
        "source"
    ]

    weights = mapping[
        "weights"
    ]

    expected_all = np.einsum(
        "cal,lctq->actq",
        weights,
        layer_data,
    )

    with LaminarSourceData(
        fname
    ) as source:
        np.testing.assert_allclose(
            source.lamina(),
            expected_all,
        )

        np.testing.assert_allclose(
            source.lamina(
                lamina="III",
                column=1,
                time=(
                    0.0,
                    50.0,
                ),
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

    with LaminarSourceData(
        fname
    ) as source:
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

    with LaminarSourceData(
        fname
    ) as source:
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
        "bad_edges",
        "bad_weights",
        "bad_valid_columns",
        "bad_labels",
        "bad_n_laminae",
    ],
)
def test_laminar_source_data_bigbrain_schema_validation(
    tmp_path,
    failure,
):
    """Reject malformed stored BigBrain mappings."""
    fname = (
        tmp_path
        / f"bigbrain_{failure}.h5"
    )

    make_source_file(
        fname
    )

    if (
        failure
        == "partial_mapping"
    ):
        with h5py.File(
            fname,
            "r+",
        ) as source_h5:
            group = (
                source_h5.create_group(
                    "bigbrain"
                )
            )

            group.create_dataset(
                "edges",
                data=np.zeros(
                    (
                        4,
                        7,
                    )
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
            group = source_h5[
                "bigbrain"
            ]

            if (
                failure
                == "missing_weights"
            ):
                del group[
                    "weights"
                ]

            elif (
                failure
                == "bad_edges"
            ):
                del group[
                    "edges"
                ]

                group.create_dataset(
                    "edges",
                    data=np.zeros(
                        (
                            5,
                            7,
                        )
                    ),
                )

            elif (
                failure
                == "bad_weights"
            ):
                del group[
                    "weights"
                ]

                group.create_dataset(
                    "weights",
                    data=np.zeros(
                        (
                            4,
                            6,
                            4,
                        )
                    ),
                )

            elif (
                failure
                == "bad_valid_columns"
            ):
                del group[
                    "valid_columns"
                ]

                group.create_dataset(
                    "valid_columns",
                    data=np.ones(
                        5,
                        dtype=bool,
                    ),
                )

            elif (
                failure
                == "bad_labels"
            ):
                del group[
                    "labels"
                ]

                group.create_dataset(
                    "labels",
                    data=np.asarray(
                        [
                            [
                                "I",
                                "II",
                                "III",
                                "IV",
                                "V",
                                "VI",
                            ]
                        ],
                        dtype="S3",
                    ),
                )

            elif (
                failure
                == "bad_n_laminae"
            ):
                group.attrs[
                    "n_laminae"
                ] = 5

    with pytest.raises(
        ValueError
    ):
        LaminarSourceData(
            fname
        )