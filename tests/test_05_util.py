"""
This module contains the unit tests for the `utils` module from the `lameg` package.
"""
import os
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

import numpy as np

import mne
import spm_standalone

from lameg.util import (check_many, spm_context, big_brain_proportional_layer_boundaries,
                        get_fiducial_coords, get_files, get_directories, make_directory,
                        calc_prop, batch, load_meg_sensor_data, get_surface_names,
                        convert_fsaverage_to_native, convert_native_to_fsaverage,
                        ttest_rel_corrected, get_bigbrain_layer_boundaries, ctf_fif_spm_conversion)


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
        _ = spm.spm(
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
        _ = spm.spm(
            "Version",
            nargout=1
        )
    except RuntimeError:
        terminated = True
    assert terminated


def test_batch():
    """
    Test the spm batch functionality
    """
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    mri_fname = os.path.join(
        test_data_path,
        'sub-104/mri/s2023-02-28_13-33-133958-00001-00224-1.nii'
    )

    with spm_context() as spm:
        cfg = {
            "spm": {
                "util": {
                    "checkreg": {
                        "data": np.asarray([f'{mri_fname},1'], dtype="object")
                    }
                }
            }
        }
        batch(cfg, spm_instance=spm)


def test_load_meg_sensor_data():
    """
    Tests the `load_meg_sensor_data` function to ensure it correctly loads and processes MEG
    dataset files.

    This test function handles two cases:
    1. Testing with a .mat file saved in the MATLAB v7.3 format.
    2. Testing with a .mat file saved in a format prior to MATLAB v7.3.

    For each case, it checks:
    - The shape of the loaded data array to ensure it matches expected dimensions.
    - The range and spacing of the time array to confirm correct timing information.
    - The accuracy of the first few data points against a pre-defined target array.

    Assertions are used to verify:
    - Data dimensions to confirm correct data loading and reshaping.
    - Time array values to ensure correct conversion and alignment with expected sample timings.
    - Specific data values to validate data integrity and correct numeric processing.

    Raises:
        AssertionError: If any of the checked conditions fail, indicating an issue with the data
        loading or processing.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')

    # Test file that does not exist
    error_raise = False
    try:
        _, _, _ = load_meg_sensor_data('X.mat')
    except FileNotFoundError:
        error_raise = True
    assert error_raise

    # Test one that is v7.3
    filename = os.path.join(
        test_data_path,
        'sub-104/meg/ses-01/spm/spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    )
    data, time, ch_names = load_meg_sensor_data(filename)
    assert data.shape[0] == 274 and data.shape[1] == 601 and data.shape[2] == 60
    assert (np.min(time) == -0.5 and np.max(time) == 0.5 and
            np.isclose(np.diff(time)[0], 0.001666666667))
    assert ch_names[0] == 'MLC11'
    target = np.array([127.983536, 6.9637985, -12.615336, 76.55547, 13.708231, -137.579, 90.26142,
                       -19.759878, 70.513176, 82.06577])
    assert np.allclose(data[0,0,:10], target)

    # Test one that is prior to v7.3
    filename = os.path.join(test_data_path,
                            'sub-104/meg/ses-01/spm/'
                            'pspm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat')
    data, time, ch_names = load_meg_sensor_data(filename)
    assert data.shape[0] == 274 and data.shape[1] == 121 and data.shape[2] == 60
    assert (np.abs(np.min(time)- -0.09999999999999) < 1e-6 and np.abs(np.max(time)-0.1) < 1e-6 and
            np.isclose(np.diff(time)[0], 0.001666666667))
    assert ch_names[0] == 'MLC11'
    target = np.array([35.541718, 116.940315, -17.433748, 132.34103, 44.1114, 88.78001, -68.96891,
                       8.402161, -80.73266, -18.521292])
    assert np.allclose(data[0, 0, :10], target)


def test_get_surface_names():
    """
   Tests the `get_surface_names` function to ensure it returns correct surface names
   and handles errors appropriately when expected files are not found.

   This function performs two main checks:
   1. It verifies that the `get_surface_names` function correctly returns a list of filenames
      for a given number of surface files in a specified directory with a given file suffix.
   2. It tests the error handling of the `get_surface_names` function by passing a non-existent
      file suffix and expecting a FileNotFoundError to be raised.

   Steps:
   - Retrieves a set of surface names from a directory for a given file type and asserts that
     the number of files returned matches the expected count.
   - Attempts to retrieve surface names using a file suffix that does not exist to ensure that
     the function raises a FileNotFoundError, confirming proper error handling.

   Assertions:
   - Asserts that the number of filenames returned by the first call is exactly 11, matching
     the specified number of surface layers.
   - Asserts that a FileNotFoundError is raised when a non-existent file suffix is used,
     indicating robust error handling.

   Raises:
       AssertionError: If any of the conditions checked by the assertions are not met, indicating
       that there is an issue with the functionality of the `get_surface_names` function.
   """
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    surf_dir = os.path.join(test_data_path, 'sub-104/surf')

    layer_fnames = get_surface_names(
        11,
        surf_dir,
        'link_vector.fixed'
    )

    assert len(layer_fnames) == 11

    error_raise = False
    try:
        _ = get_surface_names(
            11,
            surf_dir,
            'test'
        )
    except FileNotFoundError:
        error_raise = True
    assert error_raise


def test_fif_spm_conversion():
    """
    Tests the `fif_spm_conversion` function to ensure it correctly converts MEG data from FIF
    format used by MNE-Python into a format compatible with SPM, validating both epoch and
    continuous data conversions.

    This test verifies:
    1. Correct conversion of epoch MEG data from FIF format to a format that SPM can use.
    2. Correct conversion of continuous raw MEG data from FIF to SPM format.
    3. The integrity of the converted data by comparing it against the original data to ensure that
       the conversion process preserves the data accurately.

    Steps performed:
    - Convert epoched MEG data from FIF to SPM format and verify the data integrity by comparing
      the converted data with the original data.
    - Convert raw continuous MEG data from FIF to SPM format and similarly verify the data
      integrity.

    Assertions:
    - For epoched data, assert that the data values in the converted SPM file closely match the
      corresponding values in the original FIF file, scaled appropriately.
    - For raw data, perform a similar assertion to ensure the values match, confirming that the
      conversion process does not alter the data inappropriately.

    The test uses file paths and data from a predefined test dataset and involves file operations
    expected to produce outputs in the './output' directory. This test ensures the reliability and
    accuracy of the data conversion tools used in the processing pipeline, essential for subsequent
    analytical tasks in SPM.
    """
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    path = Path(os.path.join(test_data_path, "sub-104/meg/ses-01"))
    exp_name = "mobeta"

    raw_path = path.joinpath("mne/sub-104-ses-01-001-raw.fif")
    epo_path = path.joinpath("mne/autoreject-sub-104-ses-01-001-btn_trial-epo.fif")
    raw_ctf_path = os.path.join(test_data_path, "sub-104/meg/ses-01/raw/")
    subject = 'sub-104'
    res4_files = get_files(raw_ctf_path, "*.res4", strings=[exp_name])
    res4_file = [i for i in res4_files if check_many(i.parts, subject, func="any")][0]

    output_path = './output'

    # Epoched data
    ctf_fif_spm_conversion(
        epo_path, res4_file, output_path,
        True
    )
    epochs = mne.read_epochs(epo_path)
    epo_data = epochs.get_data()
    out_fname = os.path.join(output_path, "spm_autoreject-sub-104-ses-01-001-btn_trial-epo.mat")
    spm_data, time, ch_names = load_meg_sensor_data(out_fname)

    ch_name = epochs.ch_names[100]
    assert ch_name in ch_names
    spm_idx = ch_names.index(ch_name)
    assert np.allclose(epo_data[:, 100, :]*1e15, spm_data[spm_idx,:,:].T)

    assert np.allclose(epochs.times, time)

    # Raw data
    ctf_fif_spm_conversion(
        raw_path, res4_file, output_path,
        False
    )
    raw = mne.io.read_raw_fif(raw_path)
    raw_data = raw.get_data()
    out_fname = os.path.join(output_path, "spm_sub-104-ses-01-001-raw.mat")
    spm_data, time, ch_names = load_meg_sensor_data(out_fname)

    ch_name = raw.ch_names[100]
    assert ch_name in ch_names
    spm_idx = ch_names.index(ch_name)
    assert np.allclose(raw_data[100,:]*1e15, spm_data[spm_idx,:])

    assert np.allclose(raw.times, time)


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
    handles both string and list inputs for directory paths
    """
    # Use the actual Path object for the directory path
    test_root = Path("test_root")
    new_dir = "new_dir"

    # Patch the mkdir and exists methods of the Path class
    with patch.object(Path, 'mkdir') as mock_mkdir:
        # Test directory creation without checking existence
        result = make_directory(test_root, new_dir)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result, "Function should return True if directory is presumed created"


def test_convert_fsaverage_to_native():
    """
    Tests the `convert_fsaverage_to_native` function to ensure it accurately converts
    FreeSurfer average coordinates to native brain space coordinates and handles errors
    appropriately when provided with invalid inputs.

    This test performs the following checks:
    1. Verify that the function returns the correct native vertex index for given valid inputs.
    2. Test the function's error handling by passing an out-of-range vertex index to ensure
       an IndexError is raised.
    3. Test the function's error handling by passing a non-existent subject ID to ensure
       a FileNotFoundError is raised.

    Procedures:
    - Calls `convert_fsaverage_to_native` with a valid subject ID, hemisphere, and vertex index.
      Checks if the returned value matches the expected native vertex index.
    - Attempts to call the function with an invalid vertex index to verify that an IndexError
      is properly triggered.
    - Attempts to call the function with a non-existent subject ID to verify that a
      FileNotFoundError is properly triggered.

    Assertions:
    - Asserts that the native vertex index returned matches the expected value of 166759 for
      known valid inputs.
    - Asserts that an IndexError is raised when a vertex index is used that exceeds the valid range.
    - Asserts that a FileNotFoundError is raised when an invalid subject ID is used.

    Raises:
        AssertionError: If any of the conditions checked by the assertions are not met, indicating
        an issue with the function's accuracy or error handling capabilities.
    """
    native_vtx = convert_fsaverage_to_native('sub-104', 'lh', 1000)
    assert native_vtx == 166759

    native_vtx = convert_fsaverage_to_native('sub-104', 'rh', 1000)
    assert native_vtx == 471282

    error_raise = False
    try:
        convert_fsaverage_to_native('sub-104', 'lh', 100000000)
    except IndexError:
        error_raise = True
    assert error_raise

    error_raise = False
    try:
        convert_fsaverage_to_native('xxx', 'lh', 1000)
    except FileNotFoundError:
        error_raise = True
    assert error_raise


def test_convert_native_to_fsaverage():
    """
    Tests the `convert_native_to_fsaverage` function to ensure it accurately maps native brain
    space coordinates to FreeSurfer average brain coordinates and handles errors appropriately.

    This function performs multiple checks:
    1. Validate that the function correctly returns the hemisphere and fsaverage vertex index
       for a given set of native space coordinates.
    2. Test the function's error handling by passing an invalid subject ID, expecting a
       FileNotFoundError to be raised.
    3. Test the function's error handling by passing an invalid surface path, also expecting a
       FileNotFoundError to be raised.

    Procedures:
    - Calls `convert_native_to_fsaverage` with a valid subject ID and surface path, checking
      if the hemisphere and vertex index returned match expected values.
    - Attempts to invoke the function with a non-existent subject ID to check for proper
      exception handling.
    - Attempts to invoke the function with a non-existent surface path to check for proper
      exception handling.

    Assertions:
    - Asserts that the hemisphere and fsaverage vertex index are as expected for valid inputs,
      confirming the function's accuracy in coordinate transformation.
    - Asserts that a FileNotFoundError is raised for an invalid subject ID and surface path,
      indicating robust error handling.

    Raises:
        AssertionError: If any of the conditions checked by the assertions are not met, indicating
        an issue with the function's accuracy or error handling capabilities.
    """
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    surf_path = os.path.join(test_data_path, 'sub-104/surf')
    hemi, fs_vert = convert_native_to_fsaverage(
        'sub-104',
        surf_path,
        [-5.045391, -58.015587, 28.667336]
    )
    assert hemi == 'lh'
    assert fs_vert == 87729

    hemi, fs_vert = convert_native_to_fsaverage(
        'sub-104',
        surf_path,
        [25.045391, -58.015587, 28.667336]
    )
    assert hemi == 'rh'
    assert fs_vert == 86092

    error_raise = False
    try:
        _, _ = convert_native_to_fsaverage(
            'xxx',
            surf_path,
            [-5.045391, -58.015587, 28.667336]
        )
    except FileNotFoundError:
        error_raise = True
    assert error_raise

    error_raise = False
    try:
        _, _ = convert_native_to_fsaverage(
            'sub-104',
            'xxxx',
            [-5.045391, -58.015587, 28.667336]
        )
    except FileNotFoundError:
        error_raise = True
    assert error_raise


def test_ttest_rel_corrected():
    """
    Tests the `ttest_rel_corrected` function with various configurations to ensure it accurately
    calculates the t-statistic, degrees of freedom, and p-value across different scenarios.

    The function is tested under four main scenarios:
    1. Default settings without any corrections applied.
    2. Applying a variance correction of 0.1 to see its impact on the t-statistic and p-value.
    3. Using a left one-tailed test (tail=-1) to verify the calculation of p-values for this
       scenario.
    4. Using a right one-tailed test (tail=1) to check for correctness in this tail type.

    Each scenario checks:
    - The closeness of computed t-values to the expected values with a tolerance of 0.01.
    - Exact matches for computed degrees of freedom against expected values.
    - The closeness of computed p-values to the expected values with a tolerance of 0.01.

    Assertions are used to ensure that each test accurately reflects expected outcomes, providing
    feedback if any value does not meet the expected criteria.
    """

    data = np.array([[1, 2, np.nan, 4],
                     [1, 2, 3, 4]]).T

    expected_tval = np.array([3.22428901, 4.44456638])
    expected_deg_of_freedom = np.array([2, 3])
    expected_p_val = np.array([0.08421719, 0.02118368])

    tval, deg_of_freedom, p_val = ttest_rel_corrected(data)

    assert np.allclose(tval, expected_tval, atol=1e-2), "T-values do not match expected"
    assert np.array_equal(deg_of_freedom, expected_deg_of_freedom), ("Degrees of freedom do not "
                                                                     "match expected")
    assert np.allclose(p_val, expected_p_val, atol=1e-2), "P-values do not match expected"

    expected_tval = np.array([3.14098262, 4.30331483])
    expected_deg_of_freedom = np.array([2, 3])
    expected_p_val = np.array([0.08816231, 0.02309284])

    tval, deg_of_freedom, p_val = ttest_rel_corrected(data, correction=0.1)

    assert np.allclose(tval, expected_tval, atol=1e-2), "T-values do not match expected"
    assert np.array_equal(deg_of_freedom, expected_deg_of_freedom), ("Degrees of freedom do not "
                                                                     "match expected")
    assert np.allclose(p_val, expected_p_val, atol=1e-2), "P-values do not match expected"

    expected_tval = np.array([3.22428901, 4.44456638])
    expected_deg_of_freedom = np.array([2, 3])
    expected_p_val = np.array([0.9578914, 0.98940816])

    tval, deg_of_freedom, p_val = ttest_rel_corrected(data, tail=-1)

    assert np.allclose(tval, expected_tval, atol=1e-2), "T-values do not match expected"
    assert np.array_equal(deg_of_freedom, expected_deg_of_freedom), ("Degrees of freedom do not "
                                                                     "match expected")
    assert np.allclose(p_val, expected_p_val, atol=1e-2), "P-values do not match expected"

    expected_tval = np.array([3.22428901, 4.44456638])
    expected_deg_of_freedom = np.array([2, 3])
    expected_p_val = np.array([0.9578914, 0.98940816])

    tval, deg_of_freedom, p_val = ttest_rel_corrected(data, tail=1)

    assert np.allclose(tval, expected_tval, atol=1e-2), "T-values do not match expected"
    assert np.array_equal(deg_of_freedom, expected_deg_of_freedom), ("Degrees of freedom do not "
                                                                     "match expected")
    assert np.allclose(p_val, expected_p_val, atol=1e-2), "P-values do not match expected"


def test_calc_prop():
    """
    Tests the `calc_prop` function with a vector where the sum is zero, non-zero, and when it has
    zero- and non-zero elements.
    """
    vec = np.array([0, 0, 0])
    result = calc_prop(vec)
    np.testing.assert_array_equal(
        result,
        vec,
        err_msg="Should return original vector when sum is zero"
    )

    vec = np.array([1, 2, 3])
    expected_result = np.array([1, 3, 6]) / 6  # cumulative sum divided by total sum
    result = calc_prop(vec)
    np.testing.assert_allclose(
        result,
        expected_result,
        rtol=1e-5,
        err_msg="Cumulative proportions are incorrect"
    )

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
    assert np.allclose(bb_data['lh'][:, 0], expected)

    assert 'rh' in bb_data
    assert bb_data['rh'].shape[0] == 6 and bb_data['rh'].shape[1] == 163842
    expected = np.array([0.07103447, 0.15451714, 0.46817848, 0.53011256, 0.7344828, 1.])
    assert np.allclose(bb_data['rh'][:, 0], expected)

    bb_data = big_brain_proportional_layer_boundaries(overwrite=True)

    assert 'lh' in bb_data
    assert bb_data['lh'].shape[0] == 6 and bb_data['lh'].shape[1] == 163842
    expected = np.array([0.07864515, 0.13759026, 0.3424378, 0.4091583, 0.64115983, 1])
    assert np.allclose(bb_data['lh'][:, 0], expected)

    assert 'rh' in bb_data
    assert bb_data['rh'].shape[0] == 6 and bb_data['rh'].shape[1] == 163842
    expected = np.array([0.07103447, 0.15451714, 0.46817848, 0.53011256, 0.7344828, 1.])
    assert np.allclose(bb_data['rh'][:, 0], expected)


def test_get_bigbrain_layer_boundaries():
    """
    Tests the `get_bigbrain_layer_boundaries` function to ensure it accurately retrieves
    the proportional boundaries of brain layers from the BigBrain model.

    This function performs several checks:
    1. Verifies that the function correctly returns the expected array of proportional
       boundaries for a given subject and coordinates using known correct inputs.
    2. Confirms that the function raises a FileNotFoundError when provided with an
       invalid subject identifier, ensuring robust error handling.
    3. Ensures that the function also raises a FileNotFoundError when given an incorrect
       path to the surface files, which tests the function's dependency on file paths.

    The tests use:
    - A path to test data structured in a typical project directory format.
    - Hardcoded coordinates which are representative of typical inputs.
    - Assertions to verify both the data accuracy and the error handling mechanisms.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    surf_path = os.path.join(test_data_path, 'sub-104/surf')

    vert_bb_prop = get_bigbrain_layer_boundaries(
        'sub-104',
        surf_path,
        [-5.045391, -58.015587, 28.667336]
    )
    expected = np.array([0.18065107, 0.2555629 , 0.4672846 , 0.7949229 , 0.90064305, 1.])
    assert np.allclose(vert_bb_prop, expected)

    error_raise = False
    try:
        _ = get_bigbrain_layer_boundaries(
            'xxx',
            surf_path,
            [-5.045391, -58.015587, 28.667336]
        )
    except FileNotFoundError:
        error_raise = True
    assert error_raise

    error_raise = False
    try:
        _ = get_bigbrain_layer_boundaries(
            'sub-104',
            'xxx',
            [-5.045391, -58.015587, 28.667336]
        )
    except FileNotFoundError:
        error_raise = True
    assert error_raise


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
