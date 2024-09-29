"""
This module contains the unit tests for the `invert` module from the `lameg` package.
"""

import os
import shutil
import h5py
import numpy as np
import pytest

from lameg.invert import (coregister, invert_ebb, invert_msp, load_source_time_series,
                          invert_sliding_window)
from lameg.util import get_fiducial_coords, make_directory


@pytest.mark.dependency()
def test_coregister(spm):
    """
    Tests the coregistration process for neuroimaging data, ensuring that the output is properly
    formatted and correctly written to the simulation output files.

    This test is dependent on the successful execution of `test_smoothmesh_multilayer_mm`, as it
    uses data files prepared and potentially modified by that prior test.

    This test performs several key operations:
    1. Retrieves fiducial coordinates necessary for coregistration from a participant's metadata.
    2. Prepares a specific MEG data file for simulation based on test data paths and session
       identifiers.
    3. Copies necessary data files to a designated output directory for processing.
    4. Executes the coregistration function using specified MRI files and mesh data for the forward
       model.
    5. Validates the presence of specific data structures in the output files to verify successful
       coregistration.

    Specific checks include:
    - Verifying that the 'inv' (inverse solution) field does not exist in the original data file's
      structure to ensure it's unprocessed.
    - Confirming that the 'inv' field is present in the new file after coregistration, indicating
      successful processing and data integrity.

    Methods used:
    - get_fiducial_coords to fetch fiducial points.
    - make_directory to create a directory for output data.
    - shutil.copy to duplicate necessary files to the working directory.
    - coregister function to align MEG data with an anatomical MRI using the fiducial points and
      the specified mesh.
    - h5py.File to interact with HDF5 file format for assertions on data structure.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'
    ses_id = 'ses-01'

    # Fiducial coil coordinates
    nas, lpa, rpa = get_fiducial_coords(subj_id, os.path.join(test_data_path, 'participants.tsv'))

    # Data file to base simulations on
    data_file = os.path.join(
        test_data_path,
        subj_id,
        'meg',
        ses_id,
        'spm/spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    )

    data_path, data_file_name = os.path.split(data_file)
    data_base = os.path.splitext(data_file_name)[0]

    # Where to put simulated data
    out_dir = make_directory('./', ['output'])

    # Copy data files to tmp directory
    shutil.copy(
        os.path.join(data_path, f'{data_base}.mat'),
        out_dir.joinpath(f'{data_base}.mat')
    )
    shutil.copy(
        os.path.join(data_path, f'{data_base}.dat'),
        out_dir.joinpath(f'{data_base}.dat')
    )

    # Construct base file name for simulations
    base_fname = os.path.join(out_dir, f'{data_base}.mat')

    # Native space MRI to use for coregistration
    # pylint: disable=duplicate-code
    mri_fname = os.path.join(
        test_data_path,
        subj_id,
        'mri',
        's2023-02-28_13-33-133958-00001-00224-1.nii'
    )

    # Mesh to use for forward model in the simulations
    mesh_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'multilayer.2.ds.link_vector.fixed.gii'
    )

    # Coregister data to pial mesh
    # pylint: disable=duplicate-code
    coregister(
        nas,
        lpa,
        rpa,
        mri_fname,
        mesh_fname,
        base_fname,
        viz=False,
        spm_instance=spm
    )

    with h5py.File(data_file, 'r') as old_file:
        assert 'inv' not in old_file['D']['other']

    with h5py.File(base_fname, 'r') as new_file:
        assert 'inv' in new_file['D']['other']


@pytest.mark.dependency(depends=["test_coregister"])
def test_invert_msp(spm):
    """
    Test the `invert_msp` function to ensure it accurately performs the inversion using specified
    parameters and a predefined mesh, verifying output against expected values.

    This test is dependent on the successful execution of `test_coregister`, as it uses data files
    prepared and potentially modified by that prior test.

    Key Functionalities Tested:
    1. Execution of the `invert_msp` function with a specific pial mesh and a defined number of
       layers, patch size, and number of temporal modes.
    2. Validation of the output against expected free energy values and cross-validation errors to
       ensure accuracy and reliability.

    Steps:
    - Retrieve paths to necessary data and mesh files required for the simulation.
    - Execute the `invert_msp` function using these files along with specified simulation
      parameters.
    - Compare the results (free energy and cross-validation errors) to predefined expected values.

    Assertions:
    - Verify that the computed free energy closely matches the expected theoretical value, ensuring
      the inversion's accuracy.
    - Confirm that the cross-validation errors align with expected results, validating the method's
      effectiveness in realistic usage scenarios.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'

    # Data file to base simulations on
    base_fname = os.path.join(
        './output',
        'spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    )

    # Mesh to use for forward model in the simulations
    mesh_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'multilayer.2.ds.link_vector.fixed.gii'
    )

    patch_size = 5
    n_temp_modes = 4
    n_layers = 2
    [free_energy, cv_err] = invert_msp(
        mesh_fname,
        base_fname,
        n_layers,
        patch_size=patch_size,
        n_temp_modes=n_temp_modes,
        viz=False,
        spm_instance=spm
    )

    target = -414469.3671265507
    assert np.isclose(free_energy[()], target)
    assert np.allclose(cv_err, [1, 0])

    # pylint: disable=unbalanced-tuple-unpacking
    [free_energy, cv_err, mu_matrix] = invert_msp(
        mesh_fname,
        base_fname,
        n_layers,
        priors=[47507],
        patch_size=patch_size,
        n_temp_modes=n_temp_modes,
        return_mu_matrix=True,
        viz=False,
        spm_instance=spm
    )

    target = np.array([ 1.01755888e-04, -1.31537562e-05, -1.46884080e-04, -4.21048807e-04,
                        -7.16753356e-04, -1.23996197e-03, -1.64400621e-03, -2.66500725e-04,
                        -4.32804336e-04, -6.46597353e-04])
    assert np.allclose(mu_matrix[47507,:10], target)
    target = -530908.9460491342
    assert np.isclose(free_energy[()], target)
    assert np.allclose(cv_err, [1, 0])


@pytest.mark.dependency(depends=["test_invert_msp"])
def test_load_source_time_series():
    """
    Tests the `load_source_time_series` function to ensure it correctly loads and processes
    source-level time series data from simulation output files.

    The function is assessed on its ability to:
    1. Load time series data for specified vertices from a given dataset.
    2. Accurately retrieve the associated time points and modulation (mu) matrix related to the
       source time series.

    The test involves:
    - Extracting data from a predefined file path.
    - Verifying that the time series, time vector, and mu matrix data extracted match expected
      predefined values for given vertices.

    Assertions are made to:
    - Confirm that the time series data for the first 10 time points of the first vertex
      closely match the target values, ensuring data integrity and correct processing.
    - Ensure that the extracted time vector for the first 10 time points matches the expected
      sequence, verifying correct time alignment.
    - Verify that the first 10 values of the mu matrix are as expected, demonstrating accurate
      modulation factor retrieval.

    The test uses hardcoded targets for comparison, expecting precise numerical agreement to ensure
    data accuracy and correct functionality.
    """

    # Data file to base simulations on
    base_fname = os.path.join(
        './output',
        'spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    )

    time_series, time, mu_matrix = load_source_time_series(base_fname, vertices=[47507])

    target = np.array([-2.43517717, -2.17243199, 7.50516469, -4.26512417, -0.01384067, -1.77825104,
                       -3.05322203, 0.16328624, -2.27361571, -2.79355311])
    assert np.allclose(time_series[0,:10,0], target)

    target = np.array([[-0.5,        -0.49833333, -0.49666667, -0.495,      -0.49333333,
                        -0.49166667, -0.49,       -0.48833333, -0.48666667, -0.485     ]])
    assert np.allclose(time[:10], target)

    target = np.array([ 1.01755888e-04, -1.31537562e-05, -1.46884080e-04, -4.21048807e-04,
                        -7.16753356e-04, -1.23996197e-03, -1.64400621e-03, -2.66500725e-04,
                        -4.32804336e-04, -6.46597353e-04])
    assert np.allclose(mu_matrix[0,:10], target)


@pytest.mark.dependency(depends=["test_load_source_time_series"])
def test_invert_sliding_window(spm):
    """
    Tests the `invert_sliding_window` function to ensure it accurately performs time-resolved
    source localization by computing free energy and windows of interest (WOIs) for specified
    vertex indices.

    This test focuses on:
    1. Evaluating the function's ability to calculate free energy values across a sliding temporal
       window.
    2. Ensuring the returned WOIs (Windows of Interest) are accurate according to specified
       parameters.
    3. Verifying that the computation results are consistent with expected target values.

    Steps executed in the test:
    - Load essential paths and file names for necessary data and mesh files from a preconfigured
      directory.
    - Execute the `invert_sliding_window` function using a vertex index, a pial surface mesh for
      the forward model,
      and a base filename of preprocessed MEG data.
    - Validate the outputs, both free energy and WOIs, against predefined target arrays.

    Assertions:
    - Confirm that the computed free energy values for the first 10 temporal windows closely match
      the expected results,
      verifying the function's precision in temporal modeling.
    - Ensure that the initial segments of the returned WOIs align with expected values, testing the
      function's accuracy in defining relevant temporal segments.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'
    ses_id = 'ses-01'

    # Fiducial coil coordinates
    nas, lpa, rpa = get_fiducial_coords(subj_id, os.path.join(test_data_path, 'participants.tsv'))

    # Data file to base simulations on
    data_file = os.path.join(
        test_data_path,
        subj_id,
        'meg',
        ses_id,
        'spm/pspm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    )

    data_path, data_file_name = os.path.split(data_file)
    data_base = os.path.splitext(data_file_name)[0]

    # Where to put simulated data
    out_dir = make_directory('./', ['output'])

    # Copy data files to tmp directory
    shutil.copy(
        os.path.join(data_path, f'{data_base}.mat'),
        out_dir.joinpath(f'{data_base}.mat')
    )
    shutil.copy(
        os.path.join(data_path, f'{data_base}.dat'),
        out_dir.joinpath(f'{data_base}.dat')
    )

    # Construct base file name for simulations
    base_fname = os.path.join(out_dir, f'{data_base}.mat')

    # Native space MRI to use for coregistration
    # pylint: disable=duplicate-code
    mri_fname = os.path.join(
        test_data_path,
        subj_id,
        'mri',
        's2023-02-28_13-33-133958-00001-00224-1.nii'
    )

    # Mesh to use for forward model in the simulations
    mesh_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'multilayer.2.ds.link_vector.fixed.gii'
    )

    # Coregister data to pial mesh
    # pylint: disable=duplicate-code
    coregister(
        nas,
        lpa,
        rpa,
        mri_fname,
        mesh_fname,
        base_fname,
        spm_instance=spm
    )

    [free_energy, wois] = invert_sliding_window(
        47507,
        mesh_fname,
        base_fname,
        1,
        win_size=16,
        spm_instance=spm
    )

    target = np.array([-113846.23389608, -115796.21846236, -115796.21846236, -116513.33866804,
                       -117680.12092327, -117680.12092327, -117987.10301609, -117790.06978323,
                       -117067.60766185, -117518.04531009])
    assert np.allclose(free_energy[:10], target)

    target=np.array([-100., -100., -100., -100., -100.,
                     -100., -98.33333333, -96.66666667, -95., -93.33333333])
    assert np.allclose(wois[:10,0], target)

    [free_energy, wois] = invert_sliding_window(
        47507,
        mesh_fname,
        base_fname,
        1,
        win_size=16,
        win_overlap=False,
        spm_instance=spm
    )

    target = np.array([-117680.12092327, -117877.03529818, -117761.38616428, -117456.72473673,
                       -119142.55620933, -118125.46196952, -118169.27464856, -119110.227163,
                       -119041.55656787, -117937.72566738])
    assert np.allclose(free_energy[:10], target)

    target = np.array([-1.00000000e+02, -8.33333333e+01, -6.66666667e+01, -5.00000000e+01,
                       -3.33333333e+01, -1.66666667e+01,  2.84217094e-14,  1.66666667e+01,
                       3.33333333e+01,  5.00000000e+01])
    assert np.allclose(wois[:10, 0], target)


@pytest.mark.dependency(depends=["test_invert_sliding_window"])
def test_invert_ebb(spm):
    """
    Test the `invert_ebb` function to ensure it performs correctly with specified parameters and
    conditions.

    The function `invert_ebb` is tested here under the following conditions:
    1. Utilizing a specific mesh for the forward model, which is essential for the simulation.
    2. Executing with defined parameters for the number of layers, patch size, and temporal modes.
    3. Verifying the output against expected free energy values and cross-validation errors to
       ensure the function's accuracy.

    Steps performed:
    - Load the necessary data and mesh files from specified paths.
    - Execute the `invert_ebb` function with a pial mesh and parameters defining the number of
      layers, patch size, and the number of temporal modes.
    - Check that the computed free energy and cross-validation errors are close to the expected
      values, confirming correct functionality.

    Assertions:
    - The test asserts that the free energy output by `invert_ebb` is close to the expected value,
      ensuring computational accuracy.
    - It also checks that the cross-validation errors meet expected results, which verify the
      function's validity in practical scenarios.
    """
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'

    # Data file to base simulations on
    base_fname = os.path.join(
        './output',
        'spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    )

    # Mesh to use for forward model in the simulations
    mesh_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'multilayer.2.ds.link_vector.fixed.gii'
    )

    patch_size = 5
    n_temp_modes = 4
    n_layers = 2
    # pylint: disable=unbalanced-tuple-unpacking
    [free_energy, cv_err, mu_matrix] = invert_ebb(
        mesh_fname,
        base_fname,
        n_layers,
        patch_size=patch_size,
        n_temp_modes=n_temp_modes,
        return_mu_matrix=True,
        viz=False,
        spm_instance=spm
    )

    target = np.array([5.56419981e-08, -7.92645103e-07, -9.95737798e-07, -2.87239768e-07,
                       4.75528010e-07,  8.49869087e-07, -1.22866823e-06, -1.49523945e-06,
                       2.08794368e-07,  1.48255156e-06])
    assert np.allclose(mu_matrix[0,:10], target)
    target = -413893.00662024197
    assert np.isclose(free_energy[()], target)
    assert np.allclose(cv_err, [1, 0])
