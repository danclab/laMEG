"""Unit tests for BigBrain layer-to-lamina mapping."""

from types import SimpleNamespace

import numpy as np
import pytest

from lameg.laminar import compute_bigbrain_laminar_weights, surface_to_laminae


class _DummySurfaceSet:
    """Minimal surface-set stand-in for BigBrain mapping tests."""

    def __init__(self, boundaries, n_layers=7):
        boundaries = np.asarray(boundaries, dtype=float)
        if boundaries.ndim == 1:
            boundaries = boundaries[:, None]
        self.n_layers = n_layers
        self.layer_spacing = np.linspace(1.0, 0.0, n_layers)
        self._boundaries = boundaries
        n_columns = boundaries.shape[1]
        self._vertices = np.column_stack((np.arange(n_columns), np.zeros((n_columns, 2))))

    def load(self, layer_name, stage="ds"):
        """Return a minimal downsampled pial surface."""
        assert layer_name == "pial"
        assert stage == "ds"
        return SimpleNamespace(darrays=[SimpleNamespace(data=self._vertices)])

    def get_bigbrain_layer_boundaries(self, subj_coord=None):
        """Return all or selected cached BigBrain boundaries."""
        if subj_coord is None:
            return self._boundaries
        column_idx = np.asarray(subj_coord)[:, 0].astype(int)
        return self._boundaries[:, column_idx]


def _uniform_boundaries(n_columns=1):
    """Return six equally spaced cumulative BigBrain boundaries."""
    boundaries = np.linspace(1.0 / 6.0, 1.0, 6)[:, None]
    return np.tile(boundaries, (1, n_columns))


def test_bigbrain_weights_shape_and_normalization():
    """Weights should have the documented shape and average each lamina."""
    surf_set = _DummySurfaceSet(_uniform_boundaries(3), n_layers=11)
    edges, weights = compute_bigbrain_laminar_weights(surf_set)

    assert edges.shape == (3, 7)
    assert weights.shape == (3, 6, 11)
    np.testing.assert_allclose(edges[:, 0], 0.0)
    np.testing.assert_allclose(edges[:, -1], 1.0)
    np.testing.assert_allclose(weights.sum(axis=2), 1.0, atol=1e-12)


def test_bigbrain_weights_exact_for_linear_depth_profile():
    """Analytic weights should exactly average a linear depth profile."""
    surf_set = _DummySurfaceSet(_uniform_boundaries(), n_layers=11)
    edges, weights = compute_bigbrain_laminar_weights(surf_set)
    source_depth = 1.0 - surf_set.layer_spacing
    layer_data = 2.0 + 3.0 * source_depth

    result = surface_to_laminae(layer_data, weights[0])
    midpoint = 0.5 * (edges[0, :-1] + edges[0, 1:])
    expected = 2.0 + 3.0 * midpoint

    assert result.shape == (6,)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_bigbrain_mapping_multiple_columns_and_trailing_dimensions():
    """Column-specific weights should preserve arbitrary trailing data dimensions."""
    boundaries = np.column_stack((
        _uniform_boundaries()[:, 0],
        np.array([0.12, 0.28, 0.43, 0.63, 0.81, 1.00]),
    ))
    surf_set = _DummySurfaceSet(boundaries, n_layers=11)
    edges, weights = compute_bigbrain_laminar_weights(surf_set)
    source_depth = 1.0 - surf_set.layer_spacing

    intercept = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    slope = np.array([[0.5, 1.0, 1.5], [2.0, -0.5, 0.25]])
    layer_data = intercept[None, :, :] + source_depth[:, None, None] * slope[None, :, :]

    result = surface_to_laminae(layer_data, weights)
    midpoint = 0.5 * (edges[:, :-1] + edges[:, 1:])
    expected = intercept[None, :, :] + midpoint.T[:, :, None] * slope[None, :, :]

    assert result.shape == (6, 2, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_bigbrain_invalid_columns_are_retained_as_nan():
    """Invalid BigBrain columns should remain aligned but map to NaN."""
    valid = _uniform_boundaries()[:, 0]
    nonfinite = valid.copy()
    nonfinite[2] = np.nan
    nonincreasing = valid.copy()
    nonincreasing[2] = nonincreasing[1]
    outside = valid.copy()
    outside[4] = 1.05
    boundaries = np.column_stack((valid, nonfinite, nonincreasing, outside))
    surf_set = _DummySurfaceSet(boundaries, n_layers=7)

    with pytest.warns(RuntimeWarning, match="3 of 4 cortical columns"):
        edges, weights = compute_bigbrain_laminar_weights(surf_set)

    assert np.all(np.isfinite(edges[0]))
    assert np.all(np.isfinite(weights[0]))
    assert np.all(np.isnan(edges[1:]))
    assert np.all(np.isnan(weights[1:]))
    np.testing.assert_allclose(weights[0].sum(axis=1), 1.0, atol=1e-12)

    mapped = surface_to_laminae(np.ones((surf_set.n_layers, 4)), weights)
    np.testing.assert_allclose(mapped[:, 0], 1.0)
    assert np.all(np.isnan(mapped[:, 1:]))


def test_bigbrain_all_invalid_columns_raise():
    """Mapping should fail clearly when no valid BigBrain columns remain."""
    boundaries = _uniform_boundaries()
    boundaries[2, 0] = np.nan
    surf_set = _DummySurfaceSet(boundaries)

    with pytest.raises(ValueError, match="No cortical columns have valid BigBrain"):
        compute_bigbrain_laminar_weights(surf_set)


def test_bigbrain_column_selection():
    """Requested cortical columns should be returned in the requested order."""
    boundaries = np.column_stack((
        np.array([0.10, 0.22, 0.40, 0.60, 0.82, 1.00]),
        np.array([0.12, 0.24, 0.42, 0.61, 0.81, 1.00]),
        np.array([0.14, 0.26, 0.44, 0.63, 0.83, 1.00]),
        np.array([0.16, 0.28, 0.46, 0.65, 0.84, 1.00]),
    ))
    surf_set = _DummySurfaceSet(boundaries)
    edges, weights = compute_bigbrain_laminar_weights(surf_set, columns=[3, 1])

    assert edges.shape == (2, 7)
    assert weights.shape == (2, 6, surf_set.n_layers)
    np.testing.assert_allclose(edges[:, 1:], boundaries[:, [3, 1]].T)

    scalar_edges, scalar_weights = compute_bigbrain_laminar_weights(surf_set, columns=2)
    assert scalar_edges.shape == (1, 7)
    assert scalar_weights.shape == (1, 6, surf_set.n_layers)
    np.testing.assert_allclose(scalar_edges[0, 1:], boundaries[:, 2])


def test_bigbrain_column_selection_errors():
    """Empty and out-of-range column selections should be rejected."""
    surf_set = _DummySurfaceSet(_uniform_boundaries(3))

    with pytest.raises(ValueError, match="at least one column"):
        compute_bigbrain_laminar_weights(surf_set, columns=[])
    with pytest.raises(IndexError, match="exceeds"):
        compute_bigbrain_laminar_weights(surf_set, columns=[3])
    with pytest.raises(IndexError, match="exceeds"):
        compute_bigbrain_laminar_weights(surf_set, columns=[-1])


@pytest.mark.parametrize(
    "spacing, message",
    [
        (np.array([[1.0, 0.0]]), "one-dimensional"),
        (np.array([1.0]), "At least two"),
        (np.array([1.0, 0.4, 0.6, 0.0]), "increase monotonically"),
    ],
)
def test_bigbrain_layer_spacing_validation(spacing, message):
    """Invalid reconstructed-depth coordinates should be rejected."""
    surf_set = _DummySurfaceSet(_uniform_boundaries())
    surf_set.layer_spacing = spacing

    with pytest.raises(ValueError, match=message):
        compute_bigbrain_laminar_weights(surf_set)


def test_surface_to_laminae_shape_validation():
    """Surface-to-lamina transformation should reject incompatible dimensions."""
    single_weights = np.zeros((6, 3))
    column_weights = np.zeros((2, 6, 3))

    with pytest.raises(ValueError, match="Layer dimension mismatch"):
        surface_to_laminae(np.zeros((2, 4)), single_weights)
    with pytest.raises(ValueError, match="layer x column"):
        surface_to_laminae(np.zeros(3), column_weights)
    with pytest.raises(ValueError, match="Layer dimension mismatch"):
        surface_to_laminae(np.zeros((2, 2)), column_weights)
    with pytest.raises(ValueError, match="Column dimension mismatch"):
        surface_to_laminae(np.zeros((3, 3)), column_weights)
    with pytest.raises(ValueError, match="weights.*shape"):
        surface_to_laminae(np.zeros((3, 2)), np.zeros((1, 1, 1, 1)))
