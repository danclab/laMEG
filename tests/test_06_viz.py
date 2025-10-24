"""
This module contains the unit tests for the `utils` module from the `lameg` package.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
import pytest

from lameg.surf import LayerSurfaceSet
from lameg.viz import data_to_rgb, rgbtoint, color_map, show_surface, plot_csd


def test_data_to_rgb():
    """
    Unit test for the `data_to_rgb` function to ensure it correctly converts data to RGB values
    using various normalization methods.

    This test verifies:
    1. That the function returns the correct shape and type of the RGB array and colormap object
       when using TwoSlopeNorm normalization.
    2. That the function returns the correct shape of the RGB array when using standard Normalize
       and LogNorm normalizations.
    3. That the function raises a ValueError when an invalid normalization string is provided.

    Methods:
    - The TwoSlopeNorm test uses a linearly spaced dataset, maps it using the viridis colormap, and
      checks the output's shape and type.
    - The Normalize and LogNorm tests check the output shape for appropriate data subsets and range
      settings.
    - The error handling test verifies that the function is robust against incorrect normalization
      parameter inputs.
    """
    # Sample data and colormap setup
    data = np.linspace(-1, 1, 100)
    cmap = cm.viridis # pylint: disable=no-member

    # Test with TwoSlopeNorm
    rgb_values, colormap = data_to_rgb(
        data,
        n_bins=5,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        vcenter=0,
        ret_map=True,
        norm="TS"
    )
    assert rgb_values.shape == (100, 4), "RGB array shape is incorrect"
    assert isinstance(colormap, cm.ScalarMappable), "Colormap object not returned"

    # Test with Normalize
    rgb_values = data_to_rgb(
        data,
        n_bins=5,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        norm="N"
    )
    assert rgb_values.shape == (100, 4), "RGB array shape is incorrect"

    # Test with LogNorm
    rgb_values = data_to_rgb(
        data[data > 0],
        n_bins=5,
        cmap=cmap,
        vmin=0.1,
        vmax=1,
        norm="LOG"
    )
    assert rgb_values.shape == (50, 4), ("RGB array shape is incorrect with logarithmic "
                                         "normalization")

    # Test for ValueError when invalid norm is passed
    with pytest.raises(ValueError):
        _ = data_to_rgb(data, n_bins=5, cmap=cmap, vmin=-1, vmax=1, norm="INVALID")


def test_rgbtoint():
    """
    Test the `rgbtoint` function to ensure it accurately converts an RGB array into a 32-bit
    integer representation.

    This test will:
    1. Verify correct integer conversion of typical RGB color arrays.
    2. Check the function's response to edge cases like maximum and minimum RGB values.
    """
    # Test with typical RGB values
    assert rgbtoint([255, 255, 255]) == 0xFFFFFF, "RGB to int conversion failed for full white"
    assert rgbtoint([0, 0, 0]) == 0x000000, "RGB to int conversion failed for full black"
    assert rgbtoint([255, 0, 0]) == 0xFF0000, "RGB to int conversion failed for full red"
    assert rgbtoint([0, 255, 0]) == 0x00FF00, "RGB to int conversion failed for full green"
    assert rgbtoint([0, 0, 255]) == 0x0000FF, "RGB to int conversion failed for full blue"

    # Test with edge case values
    assert rgbtoint([0, 128, 255]) == 0x0080FF, "RGB to int conversion failed for mixed values"


def test_color_map():
    """
    Tests the `color_map` function to ensure it correctly maps numerical data to a colormap in
    hexadecimal format.

    This test verifies:
    1. Correct mapping of data to hexadecimal colors using various normalization options.
    2. Correct return of the colormap object for use in further visualization.
    3. Functionality across different ranges of data, including edge cases for vmin and vmax.

    Methods:
    - Uses a linspace data array for simplicity and uniform distribution.
    - Tests with different normalization settings ('TS', 'N', 'LOG') to ensure the function handles
      each correctly.
    - Checks the type and format of the output to ensure compatibility with expected visualization
      uses.
    """
    # Create a simple linear data set
    data = np.linspace(-10, 10, 100)
    # pylint: disable=no-member
    cmap = cm.viridis

    # Normalization 'TS' with vcenter at zero
    hex_colors, colormap = color_map(data, cmap, vmin=-10, vmax=10, norm="TS")
    assert isinstance(hex_colors, np.ndarray) and hex_colors.dtype == np.int64, \
        "Color data type or structure incorrect"
    assert isinstance(colormap, cm.ScalarMappable), \
        "Returned colormap is not a ScalarMappable object"

    # Normalization 'N' with no vcenter
    hex_colors, _ = color_map(data, cmap, vmin=0, vmax=10, norm="N")
    assert len(hex_colors) == 100, "Incorrect number of colors returned"

    # Normalization 'LOG' with positive data range
    hex_colors, _ = color_map(np.linspace(1, 100, 100), cmap, vmin=1, vmax=100, norm="LOG")
    # pylint: disable=E1136
    assert not np.all(hex_colors[0] == hex_colors[-1]), \
        ("Logarithmic scaling failed to differentiate extremes")

    # Test for handling of invalid normalization type
    with pytest.raises(ValueError):
        _, _ = color_map(data, cmap, vmin=-10, vmax=10, norm="INVALID")


def test_show_surface_basic():
    """Ensure show_surface runs and produces a k3d plot object."""
    fake_plot = MagicMock()
    fake_mesh = MagicMock()

    surf_set = LayerSurfaceSet('sub-104', 2)

    # Mock k3d and nib freesurfer functions
    with patch('k3d.mesh', return_value=fake_mesh) as mock_mesh, \
         patch('k3d.plot', return_value=fake_plot) as mock_plot:

        _ = show_surface(
            surf_set,
            layer_name='inflated',
            stage='combined',
            vertex_colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
            plot_curvature=True,
            marker_vertices=[0],
            marker_color=[0, 255, 0],
            marker_size=2.0
        )

    # Verify expected calls
    mock_mesh.assert_called_once()
    mock_plot.assert_called_once()
    assert fake_plot.__iadd__.called  # Mesh added to plot

    with patch('k3d.mesh', return_value=fake_mesh) as mock_mesh, \
         patch('k3d.plot', return_value=fake_plot) as mock_plot:

        surf = surf_set.load(layer_name='pial', stage='converted', hemi='lh')
        vertices, _, *_ = surf.agg_data()
        n_verts = vertices.shape[0]
        data = np.random.randint(-1000, 1000, n_verts)

        # Get vertex colors using RdYlBu_r colormap. c_map (matplotlib.colors.Colormap) is
        # returned as well for plotting a colorbar
        v_colors, _ = color_map(
            data,
            "RdYlBu_r",
            data.min(),
            data.max()
        )
        v_colors2 = np.zeros((v_colors.shape[0], 4))
        v_colors2[:, :3] = v_colors
        v_colors2[:10000, -1] = 255
        _ = show_surface(
            surf_set,
            layer_name='pial',
            stage='converted',
            hemi='lh',
            vertex_colors=v_colors2,
            plot_curvature=True,
            marker_vertices=[0],
            marker_coords=[0,0,0],
            marker_size=2.0
        )

    # Verify expected calls
    mock_mesh.assert_called_once()
    mock_plot.assert_called_once()
    assert fake_plot.__iadd__.called  # Mesh added to plot

    with patch('k3d.mesh', return_value=fake_mesh) as mock_mesh, \
         patch('k3d.plot', return_value=fake_plot) as mock_plot:

        _ = show_surface(
            surf_set,
            layer_name='pial',
            stage='converted',
            hemi='lh',
            vertex_colors=np.array([[255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0]]),
            plot_curvature=False,
            marker_vertices=[0],
            marker_coords=[0,0,0],
            marker_size=2.0
        )

    # Verify expected calls
    mock_mesh.assert_called_once()
    mock_plot.assert_called_once()
    assert fake_plot.__iadd__.called  # Mesh added to plot

    with patch('k3d.mesh', return_value=fake_mesh) as mock_mesh, \
         patch('k3d.plot', return_value=fake_plot) as mock_plot:

        cam_view = [335, 9.5, 51,
                    60, 37, 17,
                    0, 0, 1]
        _ = show_surface(
            surf_set,
            layer_name='pial',
            stage='converted',
            hemi='lh',
            plot_curvature=True,
            marker_vertices=[0],
            marker_coords=[0,0,0],
            marker_size=[2.0],
            marker_color=[[255,0,0]],
            camera_view=cam_view,
            info=True
        )

    # Verify expected calls
    mock_mesh.assert_called_once()
    mock_plot.assert_called_once()
    assert fake_plot.__iadd__.called  # Mesh added to plot


@pytest.fixture
def mock_csd_data():
    """
    Generate mock Current Source Density (CSD) data and corresponding time vector for testing.

    This fixture creates reproducible random CSD data (11 layers × 100 time points) scaled to
    microampere units, along with a time vector ranging from -100 to 300 ms.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        csd : array, shape (11, 100)
            Simulated CSD values for 11 cortical layers.
        times : array, shape (100,)
            Corresponding time points in seconds.
    """
    np.random.seed(0)
    csd = np.random.randn(11, 100) * 1e-6
    times = np.linspace(-0.1, 0.3, 100)
    return csd, times


# pylint: disable=W0621
def test_plot_csd_defaults(mock_csd_data):
    """Check that plot_csd runs with default parameters and returns an AxesImage."""
    csd, times = mock_csd_data
    fig, axis = plt.subplots()

    with patch("matplotlib.pyplot.colorbar") as mock_colorbar:
        image = plot_csd(csd, times, axis)

    assert image.__class__.__name__ == "AxesImage"
    assert isinstance(image.norm, TwoSlopeNorm)
    assert image.cmap.name == "RdBu_r"
    mock_colorbar.assert_called_once()
    plt.close(fig)


# pylint: disable=W0621
def test_plot_csd_with_vmin_vmax(mock_csd_data):
    """Check that explicit vmin/vmax uses correct normalization."""
    csd, times = mock_csd_data
    fig, axis = plt.subplots()

    image = plot_csd(csd, times, axis, vmin_vmax=(-1e-6, 1e-6), colorbar=False)
    assert isinstance(image.norm, TwoSlopeNorm)
    assert np.isclose(image.norm.vmin, -1e-6)
    assert np.isclose(image.norm.vmax, 1e-6)
    plt.close(fig)


# pylint: disable=W0621
def test_plot_csd_with_norm_flag(mock_csd_data):
    """Check that 'norm' flag applies Normalize instead of TwoSlopeNorm."""
    csd, times = mock_csd_data
    fig, axis = plt.subplots()

    image = plot_csd(csd, times, axis, vmin_vmax="norm", colorbar=False)
    assert isinstance(image.norm, Normalize)
    plt.close(fig)


# pylint: disable=W0621
def test_plot_csd_with_layer_boundaries(mock_csd_data):
    """Ensure layer boundaries are drawn."""
    csd, times = mock_csd_data
    fig, axis = plt.subplots()

    _ = plot_csd(csd, times, axis, layer_boundaries=[0.2, 0.5, 0.8], colorbar=False)
    # Matplotlib stores drawn lines in ax.lines
    assert len(axis.lines) == 3
    plt.close(fig)
