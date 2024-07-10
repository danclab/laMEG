"""
This module contains the unit tests for the `utils` module from the `lameg` package.
"""
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

import numpy as np

from lameg.util import (check_many, spm_context, big_brain_proportional_layer_boundaries,
                        get_fiducial_coords, get_files, get_directories, make_directory, calc_prop)
from spm import spm_standalone


def test_spm_context():
    """
    Test the spm_context to ensure proper execution and capture stdout
    """
    # Check opening new instance with context manager
    with spm_context() as spm:
        assert spm.name == 'spm_standalone'

        ver = spm.spm(
            "Version",
            nargout=1
        )
        assert ver == 'SPM (dev)'

    # Check that instance is terminated
    terminated = False
    try:
        ver = spm.spm(
            "Version",
            nargout=1
        )
    except RuntimeError:
        terminated = True
    assert terminated

    # Check using existing instance with context manager
    spm_instance = spm_standalone.initialize()
    with spm_context(spm_instance) as spm:
        assert spm.name == 'spm_standalone'

        ver = spm.spm(
            "Version",
            nargout=1
        )
        assert ver == 'SPM (dev)'

    # Check that not terminated
    ver = spm.spm(
        "Version",
        nargout=1
    )
    assert ver == 'SPM (dev)'

    spm_instance.terminate()

    # Check that terminated
    terminated = False
    try:
        ver = spm.spm(
            "Version",
            nargout=1
        )
    except RuntimeError:
        terminated = True
    assert terminated


def test_check_many():
    """
    Test the `check_many` function to verify its response to different scenarios and parameters.

    The function is tested to:
    - Throw a ValueError when `target` contains characters not in `multiple` (when applicable).
    - Correctly return True if any or all elements in `multiple` are in `target` based on the
      `func` parameter.
    - Correctly return False if not all or none of the elements in `multiple` are in `target`,
      based on the `func` parameter.

    Tests include:
    - Single element in `multiple` that is part of `target`.
    - Multiple elements in `multiple` with partial inclusion in `target`.
    - Multiple elements in `multiple` fully included in `target`.
    - No elements from `multiple` included in `target`.

    Args:
        None

    Returns:
        None
    """

    multiple = ['x']
    target = 'xy'
    val_error = False
    try:
        check_many(multiple, target)
    except ValueError:
        val_error = True
    assert val_error

    multiple = ['x', 'y']
    target = 'x'
    assert check_many(multiple, target, func='any')

    multiple = ['x', 'y']
    target = 'z'
    assert not check_many(multiple, target, func='any')

    multiple = ['x', 'x']
    target = 'x'
    assert check_many(multiple, target, func='all')

    multiple = ['x', 'y']
    target = 'x'
    assert not check_many(multiple, target, func='all')

    multiple = ['x', 'y']
    target = 'z'
    assert not check_many(multiple, target, func='all')


def test_get_files():
    """
    Tests the `get_files` function to ensure it correctly identifies files based on specified
    suffixes, prefixes, and content strings, and adheres to depth requirements.
    """
    # Mock files and directories with appropriate properties
    mock_file1 = MagicMock(spec=Path, is_file=MagicMock(return_value=True))
    mock_file2 = MagicMock(spec=Path, is_file=MagicMock(return_value=True))
    mock_dir1 = MagicMock(spec=Path, is_file=MagicMock(return_value=False))
    mock_dir2 = MagicMock(spec=Path, is_file=MagicMock(return_value=False))
    mock_file3 = MagicMock(spec=Path, is_file=MagicMock(return_value=True))

    mock_file1.name = "file1.txt"
    mock_file1.suffix = ".txt"
    mock_file2.name = "file2.txt"
    mock_file2.suffix = ".txt"
    mock_file3.name = "file3.txt"
    mock_file3.suffix = ".txt"

    # Setup the specific methods and properties needed
    mock_path_instance = MagicMock(spec=Path)
    mock_path_instance.rglob.return_value = [mock_file1, mock_file2]
    mock_path_instance.iterdir.return_value = [mock_dir1, mock_dir2, mock_file3]

    # Patch the Path constructor to return our mock instance
    with patch.object(Path, '__new__', return_value=mock_path_instance):
        # Function call for 'all' depth
        files = get_files(
            "test_dir",
            "*.txt",
            strings=["file"],
            prefix="file",
            check="all",
            depth="all"
        )
        assert len(files) == 2, "Expected 2 files to be identified"

        # Function call for 'one' depth
        files = get_files(
            "test_dir",
            "*.txt",
            strings=["file"],
            prefix="file",
            check="all",
            depth="one"
        )
        assert len(files) == 1, "Expected 1 file to be identified in shallow search"


def test_get_directories():
    """
    Tests the `get_directories` function to ensure it accurately finds directories based on
    specific strings and adheres to specified search depths.
    """
    # Create a MagicMock for mocking the glob and iterdir methods
    mock_dir1 = MagicMock(spec=Path, is_dir=MagicMock(return_value=True))
    mock_dir2 = MagicMock(spec=Path, is_dir=MagicMock(return_value=True))
    mock_dir3 = MagicMock(spec=Path, is_dir=MagicMock(return_value=True))

    mock_dir1.__str__.return_value = "test_dir/subdir1"
    mock_dir2.__str__.return_value = "test_dir/subdir2"
    mock_dir3.__str__.return_value = "test_dir/not_included/subdir3"

    with patch.object(Path, 'glob', return_value=[mock_dir1, mock_dir2, mock_dir3]) as mock_glob, \
         patch.object(Path, 'iterdir',
                      return_value=[
                          mock_dir1,
                          mock_dir2,
                          MagicMock(spec=Path, is_dir=MagicMock(return_value=False))
                      ]) as mock_iterdir:

        test_dir = Path("test_dir")
        # Test with depth 'all'
        directories = get_directories(test_dir, strings=["subdir"], check="all", depth="all")
        mock_glob.assert_called_once_with("**/")
        assert len(directories) == 3, "Should return three directories"
        assert all(isinstance(dir, Path) for dir in directories), ("All items should be "
                                                                   "pathlib.Path objects")

        # Test with depth 'one'
        directories = get_directories(test_dir, strings=["subdir"], check="all", depth="one")
        mock_iterdir.assert_called_once()
        assert len(directories) == 2, "Should return two directories"


def test_make_directory():
    """
    Tests the `make_directory` function to ensure it correctly creates directories,
    handles both string and list inputs for directory paths, and properly checks the existence of
    directories.
    """
    # Use the actual Path object for the directory path
    test_root = Path("test_root")
    new_dir = "new_dir"
    dir_list = ["dir1", "dir2"]

    # Patch the mkdir and exists methods of the Path class
    with patch.object(Path, 'mkdir') as mock_mkdir, \
            patch.object(Path, 'exists', return_value=True) as mock_exists:
        # Test directory creation without checking existence
        result = make_directory(test_root, new_dir)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result, "Function should return True if directory is presumed created"

        # Test directory creation with a list of directories
        result = make_directory(test_root, dir_list, check=True)
        assert result, "Function should return True when check=True and the directory exists"

        # Test with check=True and directory does not exist
        mock_exists.return_value = False
        result = make_directory(test_root, new_dir, check=True)
        assert not result, ("Function should return False when check=True and the directory does "
                            "not exist")


def test_calc_prop_zero_sum():
    """
    Tests the `calc_prop` function with a vector where the sum is zero.
    The function should return the original vector unchanged because proportion calculation is not
    possible.
    """
    vec = np.array([0, 0, 0])
    result = calc_prop(vec)
    np.testing.assert_array_equal(
        result,
        vec,
        err_msg="Should return original vector when sum is zero"
    )

def test_calc_prop_non_zero_sum():
    """
    Tests the `calc_prop` function with a vector where the sum is non-zero.
    The function should return a vector of cumulative proportions.
    """
    vec = np.array([1, 2, 3])
    expected_result = np.array([1, 3, 6]) / 6  # cumulative sum divided by total sum
    result = calc_prop(vec)
    np.testing.assert_allclose(
        result,
        expected_result,
        rtol=1e-5,
        err_msg="Cumulative proportions are incorrect"
    )

def test_calc_prop_including_zero_elements():
    """
    Tests the `calc_prop` function with a vector including zero and non-zero elements.
    It should handle zeros correctly in the proportion calculations.
    """
    vec = np.array([0, 1, 2, 0, 3])
    expected_result = np.array([0, 1, 3, 3, 6]) / 6  # cumulative sum taking into account zeros
    result = calc_prop(vec)
    np.testing.assert_allclose(
        result,
        expected_result,
        rtol=1e-5,
        err_msg="Handling of zeros in vector is incorrect"
    )


def test_big_brain_proportional_layer_boundaries():
    """
    Tests the big_brain_proportional_layer_boundaries function to ensure it returns accurate and
    expected layer boundary data for both left hemisphere (lh) and right hemisphere (rh).

    This function performs the following checks:
    - Asserts that the 'lh' and 'rh' keys exist in the returned dictionary.
    - Asserts that the shape of the arrays for 'lh' and 'rh' is correct, verifying the number of
      layers (6) and the expected number of vertices (163842).
    - Checks that the first column of each hemisphere's data closely matches a predefined expected
      array of layer boundary values, with a tolerance for maximum absolute difference set to less
      than 1e-6.

    The function is called twice to verify the consistency of outputs:
    - First with the `overwrite` parameter set to False.
    - Then with the `overwrite` parameter set to True.

    Raises:
        AssertionError: If any of the assertions fail, indicating that the expected data structure
        or values are incorrect or missing.
    """

    bb_data = big_brain_proportional_layer_boundaries(overwrite=False)

    assert 'lh' in bb_data
    assert bb_data['lh'].shape[0] == 6 and bb_data['lh'].shape[1] == 163842
    expected = np.array([0.07864515, 0.13759026, 0.3424378, 0.4091583, 0.64115983, 1])
    assert np.max(np.abs(bb_data['lh'][:, 0] - expected)) < 1e-6

    assert 'rh' in bb_data
    assert bb_data['rh'].shape[0] == 6 and bb_data['rh'].shape[1] == 163842
    expected = np.array([0.07103447, 0.15451714, 0.46817848, 0.53011256, 0.7344828, 1.])
    assert np.max(np.abs(bb_data['rh'][:, 0] - expected)) < 1e-6

    bb_data = big_brain_proportional_layer_boundaries(overwrite=True)

    assert 'lh' in bb_data
    assert bb_data['lh'].shape[0] == 6 and bb_data['lh'].shape[1] == 163842
    expected = np.array([0.07864515, 0.13759026, 0.3424378, 0.4091583, 0.64115983, 1])
    assert np.max(np.abs(bb_data['lh'][:, 0] - expected)) < 1e-6

    assert 'rh' in bb_data
    assert bb_data['rh'].shape[0] == 6 and bb_data['rh'].shape[1] == 163842
    expected = np.array([0.07103447, 0.15451714, 0.46817848, 0.53011256, 0.7344828, 1.])
    assert np.max(np.abs(bb_data['rh'][:, 0] - expected)) < 1e-6


def test_get_fiducial_coords():
    """
    Tests the `get_fiducial_coords` function to ensure it correctly fetches and parses fiducial
    coordinates from a TSV file.

    The function should correctly parse the NAS, LPA, and RPA coordinates as lists of floats for a
    given subject ID.
    """
    # Sample data to mimic what might be found in the TSV file
    tsv_data = "subj_id\tnas\tlpa\trpa\n" \
               "subj1\t1.0,2.0,3.0\t4.0,5.0,6.0\t7.0,8.0,9.0\n" \
               "subj2\t10.0,11.0,12.0\t13.0,14.0,15.0\t16.0,17.0,18.0\n"

    # Use mock_open to simulate file reading operations
    m_file = mock_open(read_data=tsv_data)

    with patch('builtins.open', m_file):
        nas, lpa, rpa = get_fiducial_coords('subj1', 'dummy_filename.tsv')

    # Assert the expected outputs
    assert nas == [1.0, 2.0, 3.0], "NAS coordinates do not match expected values"
    assert lpa == [4.0, 5.0, 6.0], "LPA coordinates do not match expected values"
    assert rpa == [7.0, 8.0, 9.0], "RPA coordinates do not match expected values"

    # Also, you might want to test the case where the subject ID is not found
    with patch('builtins.open', m_file):
        nas, lpa, rpa = get_fiducial_coords('subj3', 'dummy_filename.tsv')

    assert nas is None and lpa is None and rpa is None, ("Should return None for all coordinates if"
                                                         " the subject ID is not found")

    nas, lpa, rpa = get_fiducial_coords('sub-104', './test_data/participants.tsv')
    nas_target = np.array([0.9662503311032098, 108.83514306876269, 1.6712361927090313])
    lpa_target = np.array([-74.28671169006893, 20.15061014698176, -29.849056272705948])
    rpa_target = np.array([76.02110531729883, 18.9467849625573, -25.779407159603114])

    assert np.sum(np.abs(nas - nas_target)) < 1e-6
    assert np.sum(np.abs(lpa - lpa_target)) < 1e-6
    assert np.sum(np.abs(rpa - rpa_target)) < 1e-6
