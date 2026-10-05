"""
Internal BigBrain layer-to-lamina mapping utilities.

These helpers implement the numerical mapping from reconstructed cortical
layers to BigBrain-defined histological laminae. Public access is re-exported
from ``lameg.laminar``.
"""

import warnings

import numpy as np


def _get_bigbrain_boundaries(surf_set, columns):
    """Load BigBrain cumulative laminar boundaries for selected columns."""
    if columns is None:
        return np.asarray(
            surf_set.get_bigbrain_layer_boundaries(),
            dtype=float,
        )

    pial = surf_set.load("pial", stage="ds")
    pial_vertices = np.asarray(pial.darrays[0].data)
    column_idx = np.asarray(columns, dtype=int)

    if column_idx.ndim == 0:
        column_idx = column_idx.reshape(1)
    else:
        column_idx = column_idx.ravel()

    if column_idx.size == 0:
        raise ValueError("`columns` must contain at least one column index.")

    if np.any(column_idx < 0) or np.any(column_idx >= len(pial_vertices)):
        raise IndexError(
            "Cortical-column index exceeds the downsampled "
            "pial surface dimensions."
        )

    return np.asarray(
        surf_set.get_bigbrain_layer_boundaries(
            subj_coord=pial_vertices[column_idx]
        ),
        dtype=float,
    )


def _bigbrain_edges(bb_boundaries, source_depth):
    """Return column-wise laminar edges and a validity mask."""
    if bb_boundaries.ndim == 1:
        bb_boundaries = bb_boundaries[:, np.newaxis]

    if bb_boundaries.ndim != 2 or bb_boundaries.shape[0] != 6:
        raise ValueError(
            "Expected BigBrain boundaries with shape "
            f"(6, n_columns), got {bb_boundaries.shape}."
        )

    n_columns = bb_boundaries.shape[1]
    edges = np.empty(
        (n_columns, 7),
        dtype=float,
    )
    edges[:, 0] = 0.0
    edges[:, 1:] = bb_boundaries.T

    tolerance = 1e-8
    valid_columns = np.all(
        np.isfinite(edges),
        axis=1,
    )
    valid_columns &= np.all(
        edges >= source_depth[0] - tolerance,
        axis=1,
    )
    valid_columns &= np.all(
        edges <= source_depth[-1] + tolerance,
        axis=1,
    )

    # Remove only tiny floating-point endpoint excursions for columns that
    # otherwise contain finite, in-range boundaries.
    candidate_idx = np.flatnonzero(
        valid_columns
    )
    if candidate_idx.size:
        edges[candidate_idx] = np.clip(
            edges[candidate_idx],
            source_depth[0],
            source_depth[-1],
        )

    lamina_thickness = np.full(
        (n_columns, 6),
        np.nan,
        dtype=float,
    )
    if candidate_idx.size:
        candidate_thickness = np.diff(
            edges[candidate_idx],
            axis=1,
        )
        monotonic = np.all(
            candidate_thickness > 0,
            axis=1,
        )
        valid_columns[candidate_idx] &= monotonic

        valid_idx = candidate_idx[
            monotonic
        ]
        if valid_idx.size:
            lamina_thickness[valid_idx] = np.diff(
                edges[valid_idx],
                axis=1,
            )

    edges[~valid_columns, :] = np.nan

    return (
        edges,
        lamina_thickness,
        valid_columns,
    )


def compute_bigbrain_laminar_weights(surf_set, columns=None):
    """Implement BigBrain layer-to-lamina weight computation."""
    source_depth = (
        1.0
        - np.asarray(
            surf_set.layer_spacing,
            dtype=float,
        )
    )

    if source_depth.ndim != 1:
        raise ValueError(
            "`surf_set.layer_spacing` must be one-dimensional."
        )

    if source_depth.size < 2:
        raise ValueError(
            "At least two reconstructed layers are required."
        )

    if np.any(np.diff(source_depth) <= 0):
        raise ValueError(
            "Reconstructed layer depths must increase monotonically "
            "from pial (0) to white (1)."
        )

    bb_boundaries = _get_bigbrain_boundaries(
        surf_set,
        columns,
    )
    (
        edges,
        lamina_thickness,
        valid_columns,
    ) = _bigbrain_edges(
        bb_boundaries,
        source_depth,
    )

    n_columns = edges.shape[0]
    n_layers = source_depth.size
    weights = np.full(
        (n_columns, 6, n_layers),
        np.nan,
        dtype=float,
    )

    n_invalid = int(
        np.count_nonzero(
            ~valid_columns
        )
    )
    if n_invalid == n_columns:
        raise ValueError(
            "No cortical columns have valid BigBrain laminar boundaries."
        )
    if n_invalid:
        warnings.warn(
            "BigBrain mapping unavailable for "
            f"{n_invalid} of {n_columns} cortical columns because "
            "their mapped laminar boundaries are non-finite, outside "
            "the reconstructed depth range, or non-increasing. "
            "Their edges and weights are set to NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    valid_edges = edges[
        valid_columns
    ]
    valid_thickness = lamina_thickness[
        valid_columns
    ]
    valid_weights = np.zeros(
        (
            int(np.count_nonzero(valid_columns)),
            6,
            n_layers,
        ),
        dtype=float,
    )

    # Integrate the piecewise-linear depth profile analytically.
    #
    # Between reconstructed depths lower_depth and upper_depth:
    #
    #   f(x) =
    #       ((upper_depth - x) / interval_width) * f0
    #       + ((x - lower_depth) / interval_width) * f1
    #
    # For every BigBrain lamina, determine the overlap with this
    # reconstructed-depth interval and integrate the two basis
    # functions analytically.
    lamina_lower = valid_edges[:, :-1]
    lamina_upper = valid_edges[:, 1:]

    for layer_idx in range(n_layers - 1):
        lower_depth = source_depth[layer_idx]
        upper_depth = source_depth[layer_idx + 1]
        interval_width = upper_depth - lower_depth

        overlap_lower = np.maximum(
            lamina_lower,
            lower_depth,
        )
        overlap_upper = np.minimum(
            lamina_upper,
            upper_depth,
        )

        active = overlap_upper > overlap_lower
        overlap_width = np.where(
            active,
            overlap_upper - overlap_lower,
            0.0,
        )
        squared_difference = np.where(
            active,
            overlap_upper ** 2 - overlap_lower ** 2,
            0.0,
        )

        left_integral = (
            upper_depth * overlap_width
            - 0.5 * squared_difference
        ) / interval_width
        right_integral = (
            0.5 * squared_difference
            - lower_depth * overlap_width
        ) / interval_width

        valid_weights[:, :, layer_idx] += (
            left_integral
            / valid_thickness
        )
        valid_weights[:, :, layer_idx + 1] += (
            right_integral
            / valid_thickness
        )

    weights[
        valid_columns
    ] = valid_weights

    return edges, weights


def surface_to_laminae(layer_data, weights):
    """Apply precomputed layer-to-lamina weights."""
    layer_data = np.asarray(layer_data)
    weights = np.asarray(weights, dtype=float)

    if weights.ndim == 2:
        if layer_data.ndim < 1:
            raise ValueError("`layer_data` must have at least one dimension.")

        if layer_data.shape[0] != weights.shape[1]:
            raise ValueError(
                "Layer dimension mismatch: "
                f"layer_data has {layer_data.shape[0]} layers, "
                f"weights expect {weights.shape[1]}."
            )

        return np.tensordot(weights, layer_data, axes=(1, 0))

    if weights.ndim == 3:
        if layer_data.ndim < 2:
            raise ValueError(
                "For column-specific weights, `layer_data` must "
                "have shape layer x column x ..."
            )

        n_columns, _, n_layers = weights.shape

        if layer_data.shape[0] != n_layers:
            raise ValueError(
                "Layer dimension mismatch: "
                f"layer_data has {layer_data.shape[0]} layers, "
                f"weights expect {n_layers}."
            )

        if layer_data.shape[1] != n_columns:
            raise ValueError(
                "Column dimension mismatch: "
                f"layer_data has {layer_data.shape[1]} columns, "
                f"weights contain {n_columns}."
            )

        return np.einsum("cal,lc...->ac...", weights, layer_data, optimize=True)

    raise ValueError(
        "`weights` must have shape "
        "(lamina, layer) or "
        "(column, lamina, layer)."
    )
