"""
This module contains the unit tests for the `utils` module from the `lameg` package.
"""

import numpy as np
from matplotlib import cm
import pytest

from lameg.viz import data_to_rgb, rgbtoint, color_map


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
    assert isinstance(hex_colors, np.ndarray) and hex_colors.dtype == np.uint32, \
        "Color data type or structure incorrect"
    assert isinstance(colormap, cm.ScalarMappable), \
        "Returned colormap is not a ScalarMappable object"

    # Normalization 'N' with no vcenter
    hex_colors, _ = color_map(data, cmap, vmin=0, vmax=10, norm="N")
    assert len(hex_colors) == 100, "Incorrect number of colors returned"

    # Normalization 'LOG' with positive data range
    hex_colors, _ = color_map(np.linspace(1, 100, 100), cmap, vmin=1, vmax=100, norm="LOG")
    # pylint: disable=E1136
    assert hex_colors[0] != hex_colors[-1], \
        ("Logarithmic scaling failed to differentiate extremes")

    # Test for handling of invalid normalization type
    with pytest.raises(ValueError):
        _, _ = color_map(data, cmap, vmin=-10, vmax=10, norm="INVALID")
