"""
Schema-validation tests for stored BigBrain mappings.
"""

import h5py
import numpy as np
import pytest

import lameg.source as source_api
from tests.source_test_utils import make_source_file


def _add_valid_bigbrain_mapping(fname):
    """Add a minimal valid BigBrain mapping for schema-mutation tests."""
    with h5py.File(
        fname,
        "r+",
    ) as source_h5:
        group = source_h5.create_group(
            "bigbrain"
        )

        group.create_dataset(
            "edges",
            data=np.tile(
                np.linspace(
                    0.0,
                    1.0,
                    7,
                ),
                (4, 1),
            ),
        )

        group.create_dataset(
            "weights",
            data=np.zeros(
                (4, 6, 3),
                dtype=float,
            ),
        )

        group.create_dataset(
            "labels",
            data=np.asarray(
                "I II III IV V VI".split(),
                dtype="S3",
            ),
        )

        group.create_dataset(
            "valid_columns",
            data=np.ones(
                4,
                dtype=bool,
            ),
        )

        group.attrs[
            "n_laminae"
        ] = 6


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
        _add_valid_bigbrain_mapping(
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
                            "I II III IV V VI".split()
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
        source_api.LaminarSourceData(
            fname
        )
