"""
This module contains the unit tests for the `laminar` module from the `lameg` package.
"""
import os

import numpy as np
import pytest
import nibabel as nib

from lameg.laminar import (model_comparison, sliding_window_model_comparison, compute_csd,
                           roi_power_comparison)
from lameg.util import get_fiducial_coords, get_surface_names


@pytest.mark.dependency(depends=["tests/test_03_simulate.py::test_run_current_density_simulation"],
                        scope='session')
def test_roi_power_comparison():
    """
    Tests the `roi_power_comparison` function to evaluate its capability to accurately compute
    statistical measures comparing power in specified regions of interest (ROIs) across different
    cortical layers.

    The unit test ensures that:
    1. The function can effectively use MEG data and surface meshes for ROI analysis.
    2. It calculates t-statistics and p-values correctly for given data sets, reflecting the
       differences in neural activity within specified frequency bands and spatial regions.
    3. It correctly identifies regions of interest based on mesh data and analyzes their
       statistical significance.

    Dependencies:
        This test relies on the `test_run_current_density_simulation`, ensuring that required
        simulation outputs are available for conducting ROI power comparisons.

    Parameters:
        spm (object): An initialized instance of the SPM software

    Key steps in the test:
    - Retrieve fiducial coordinates and use them to coregister simulated MEG data to a native space
      MRI.
    - Extract mu matrices from the source inversion results to use in ROI power comparison.
    - Calculate t-statistics, p-values, degrees of freedom, and ROI indices for designated time and
      frequency ranges.
    - Validate the calculated results against expected values to ensure accuracy and reliability.

    Assertions:
    - Check if the computed t-statistic and p-value are close to their expected values, confirming
      the statistical analysis's correctness.
    - Verify the degrees of freedom and ROI indices to ensure that the function correctly
      identifies and analyzes the specified regions.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'

    # Mesh to use for forward model in the simulations
    multilayer_mesh_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'multilayer.2.ds.link_vector.fixed.gii'
    )

    mesh = nib.load(multilayer_mesh_fname)
    n_layers = 2

    sim_fname = os.path.join('./output/',
                             'sim_24588_current_density_pial_spm-converted_autoreject-'
                             'sub-104-ses-01-001-btn_trial-epo.mat')

    laminar_t_statistic, laminar_p_value, deg_of_freedom, roi_idx = roi_power_comparison(
        sim_fname,
        [0, 500],
        [-500, 0],
        mesh,
        n_layers,
        99.99,
        chunk_size=8000
    )

    target = -0.17077967243430403
    assert np.isclose(laminar_t_statistic, target)
    target = 0.8649813417336001
    assert np.isclose(laminar_p_value, target)
    target = 59
    assert deg_of_freedom == target
    target = np.array([  463,   613,  1028,  1163,  1607,  2775, 18020, 19433, 19636])
    assert np.allclose(roi_idx, target)


@pytest.mark.dependency(depends=["test_roi_power_comparison"])
def test_model_comparison(spm):
    """
    Tests the `model_comparison` function to ensure it correctly compares different source
    reconstruction models using MEG data, MRIs, and a multilayer mesh. The test verifies the
    accuracy of the model output by comparing the free energy computations.

    This test is dependent on the successful completion of `test_run_current_density_simulation`,
    which provides necessary preliminary data and conditions for this test to execute.

    Parameters:
        spm (object): An instance of SPM used for handling brain simulations, required for
                      performing the model comparison.

    Key steps executed in this test:
    - Retrieves fiducial coordinates for the subject to align the MEG data with the MRI.
    - Loads the native space MRI file necessary for the coregistration process.
    - Collects names of the surface meshes for the cortical layers used in source reconstruction.
    - Executes the `model_comparison` function with specific simulation outputs and checks the
      computed free energy values against expected values.

    Assertions:
    - Asserts that the computed free energy values closely match the predefined values, validating
      the efficacy and accuracy of the model comparison process.

    This test ensures that the integration of multiple data sources (MEG, MRI, surface models) and
    their processing through the model comparison function yields consistent and expected results,
    indicative of the function's reliability and correctness in real-world scenarios.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'

    # Fiducial coil coordinates
    nas, lpa, rpa = get_fiducial_coords(subj_id, os.path.join(test_data_path, 'participants.tsv'))

    # Native space MRI to use for coregistration
    # pylint: disable=duplicate-code
    mri_fname = os.path.join(
        test_data_path,
        subj_id,
        'mri',
        's2023-02-28_13-33-133958-00001-00224-1.nii'
    )

    # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used
    # for the sourcereconstruction
    layer_fnames = get_surface_names(
        2,
        os.path.join(test_data_path, subj_id, 'surf'),
        'link_vector.fixed'
    )

    sim_fname = os.path.join('./output/',
                             'sim_24588_current_density_pial_spm-converted_autoreject-'
                             'sub-104-ses-01-001-btn_trial-epo.mat')
    patch_size = 5
    n_temp_modes = 4

    # Test MSP
    [free_energy, _] = model_comparison(
        nas,
        lpa,
        rpa,
        mri_fname,
        [layer_fnames[0], layer_fnames[-1]],
        sim_fname,
        method='MSP',
        viz=False,
        spm_instance=spm,
        invert_kwargs={
            'patch_size': patch_size,
            'n_temp_modes': n_temp_modes,
        }
    )

    target = np.array([-202445.53571645, -200488.54855084])
    assert np.allclose(free_energy, target)

    # Test EBB
    [free_energy, _] = model_comparison(
        nas,
        lpa,
        rpa,
        mri_fname,
        [layer_fnames[0], layer_fnames[-1]],
        sim_fname,
        method='EBB',
        viz=False,
        spm_instance=spm,
        invert_kwargs={
            'patch_size': patch_size,
            'n_temp_modes': n_temp_modes,
        }
    )

    target = np.array([-207555.39271896, -207279.30947456])
    assert np.allclose(free_energy, target)


@pytest.mark.dependency(depends=["test_model_comparison"])
def test_sliding_window_model_comparison(spm):
    """
    Tests the `sliding_window_model_comparison` function to ensure it correctly performs
    time-resolved comparisons of source reconstruction models using sliding window analysis on
    MEG simulation data.

    This test validates:
    1. The function's ability to handle sliding windows of data for comparing different cortical
       layers using specified models.
    2. The correctness of free energy calculations and window of interest (WOI) determinations
       across different time windows and potential overlaps.
    3. The reliability of the function in returning consistent and expected results when analyzing
       temporal variations in simulated brain activity.

    Dependencies:
        This test depends on the successful completion of `test_run_dipole_simulation` which
        prepares necessary simulation outputs used here.

    Parameters:
        spm (object): An instance of SPM, used for handling all computational neuroscience models
                      and simulations.

    Detailed Steps:
    - Retrieve fiducial coordinates for the subject and load the native space MRI for
      coregistration.
    - Extract surface names for the multilayer mesh used in source reconstruction.
    - Run the sliding window model comparison using a simulated MEG dataset.
    - Validate the results for both free energy and windows of interest against pre-defined
      targets.

    Assertions:
    - Compare computed free energy values across the first ten sliding windows with expected
      values, ensuring accuracy in the model's temporal response.
    - Check the computed windows of interest against expected time intervals, verifying correct
      temporal localization and overlap handling.

    The test confirms that the sliding window approach accurately reflects the underlying neural
    dynamics and interactions across different cortical layers, providing insights into the
    temporal aspects of neural activity as simulated.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')

    subj_id = 'sub-104'

    # Fiducial coil coordinates
    nas, lpa, rpa = get_fiducial_coords(subj_id, os.path.join(test_data_path, 'participants.tsv'))

    # Native space MRI to use for coregistration
    # pylint: disable=duplicate-code
    mri_fname = os.path.join(
        test_data_path,
        subj_id,
        'mri',
        's2023-02-28_13-33-133958-00001-00224-1.nii'
    )

    # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used
    # for the sourcereconstruction
    layer_fnames = get_surface_names(
        2,
        os.path.join(test_data_path, subj_id, 'surf'),
        'link_vector.fixed'
    )

    sim_fname = os.path.join('./output/',
                             'sim_24588_dipole_pial_pspm-converted_autoreject-'
                             'sub-104-ses-01-001-btn_trial-epo.mat')
    patch_size = 5
    sliding_n_temp_modes = 1
    # Size of sliding window (in ms)
    win_size = 50
    # Whether or not windows overlap
    win_overlap = True

    # Run sliding time window model comparison between the first layer (pial) and the last layer
    # (white matter)
    [free_energy, wois] = sliding_window_model_comparison(
        24588,
        nas,
        lpa,
        rpa,
        mri_fname,
        [layer_fnames[0], layer_fnames[-1]],
        sim_fname,
        viz=False,
        spm_instance=spm,
        invert_kwargs={
            'patch_size': patch_size,
            'n_temp_modes': sliding_n_temp_modes,
            'win_size': win_size,
            'win_overlap': win_overlap
        }
    )

    target = np.array([[-25742.35576421, -25742.35576421, -25731.55977826, -25825.36418008,
                        -25825.36418008, -25797.52834586, -25740.6772461,  -25740.6772461,
                        -25795.43416278, -25809.27791724],
                       [-25742.35573166, -25742.35573166, -25731.55880407, -25825.36230089,
                        -25825.36230089, -25797.5262519,  -25740.67726161, -25740.67726161,
                        -25795.4352398,  -25809.27771454]])
    assert np.allclose(free_energy[:, :10], target)

    target = np.array([[-100., -75.],
                       [-100., -73.33333333],
                       [-100., -71.66666667],
                       [-100., -70.],
                       [-100., -68.33333333],
                       [-100., -66.66666667],
                       [-100., -65.],
                       [-100., -63.33333333],
                       [-100., -61.66666667],
                       [-100., -60.]])
    assert np.allclose(wois[:10, :], target)


def test_compute_csd():
    """
    Tests the `compute_csd` function to ensure it accurately computes the Current Source Density
    (CSD) from provided sample MEG data and applies a smoothing process to the CSD calculations.

    This unit test verifies:
    1. The correct computation of CSD from sample time series data.
    2. The application of a cubic smoothing process to the computed CSD.
    3. The accuracy of both the computed CSD and the smoothed CSD against predefined expected
       values.

    Steps performed:
    - Load precomputed time series data for specified vertices representing cortical activity.
    - Compute the CSD and a smoothed version of CSD for the loaded time series data.
    - Compare the results to predefined target values to validate the accuracy of the CSD
      computations.

    Assertions:
    - Assert that the values of computed CSD for specified layers closely match the expected target
      values, confirming the model's accuracy.
    - Verify that the smoothed CSD matches its respective expected values, ensuring the
      effectiveness of the smoothing process.
    """

    thickness = 4.6092634
    s_rate = 600

    # Average over trials and compute CSD and smoothed CSD
    mean_layer_ts = np.array([[-2.49494905e-03, 1.00768780e-04, 1.98419396e-03,
                               5.71883386e-04, -6.71066270e-07, 1.20126270e-03,
                               3.98305717e-03, 5.04501537e-04, -9.24579604e-04,
                               -3.59750486e-03],
                              [-2.52018103e-03, -2.35669197e-04, 2.11956574e-03,
                               4.25924778e-04, -1.14843079e-04, 1.68099310e-03,
                               4.17476505e-03, 4.35020257e-04, -9.71325416e-04,
                               -3.94518034e-03],
                              [-2.16320010e-03, -5.38142242e-04, 1.94668338e-03,
                               2.08191629e-04, -2.14643752e-04, 1.92530431e-03,
                               3.73704499e-03, 3.01677715e-04, -8.65114354e-04,
                               -3.70171427e-03],
                              [-1.72057645e-03, -7.61128067e-04, 1.68961052e-03,
                               9.45563482e-06, -2.85542464e-04, 2.02712329e-03,
                               3.13361073e-03, 1.73024677e-04, -7.18358561e-04,
                               -3.26497590e-03],
                              [-1.46091741e-03, -1.01667770e-03, 1.60740279e-03,
                               -1.65295308e-04, -3.70085107e-04, 2.29124340e-03,
                               2.84963073e-03, 7.69352056e-05, -6.43007531e-04,
                               -3.13761261e-03],
                              [-9.83809962e-04, -1.03420231e-03, 1.25902835e-03,
                               -2.74304924e-04, -3.69861793e-04, 2.09755402e-03,
                               2.10592503e-03, -1.03895019e-05, -4.64007740e-04,
                               -2.46570489e-03],
                              [-6.13101302e-04, -9.81605973e-04, 9.66236519e-04,
                               -3.27409225e-04, -3.47215572e-04, 1.85641465e-03,
                               1.50018142e-03, -6.32344645e-05, -3.19116688e-04,
                               -1.88388107e-03],
                              [-3.39723753e-04, -8.83622916e-04, 7.27684625e-04,
                               -3.38297841e-04, -3.10513506e-04, 1.59429703e-03,
                               1.02688698e-03, -8.96038551e-05, -2.07261175e-04,
                               -1.39994643e-03],
                              [-1.59558486e-04, -8.14737789e-04, 5.75912773e-04,
                               -3.42624744e-04, -2.85418715e-04, 1.42506398e-03,
                               7.18130888e-04, -1.04044899e-04, -1.33539139e-04,
                               -1.08198794e-03],
                              [-2.83522366e-05, -7.18369663e-04, 4.44810039e-04,
                               -3.23679082e-04, -2.51649491e-04, 1.23274586e-03,
                               4.70845200e-04, -1.05549327e-04, -7.61171294e-05,
                               -8.06136327e-04],
                              [6.19161819e-05, -6.43688274e-04, 3.54761023e-04,
                               -3.06079933e-04, -2.26176044e-04, 1.09429559e-03,
                               2.99992862e-04, -1.04040227e-04, -3.65785356e-05,
                               -6.12312968e-04]])

    # pylint: disable=W0632
    [csd, smooth_csd] = compute_csd(mean_layer_ts, thickness, s_rate, smoothing='cubic')

    target = np.array([ 0.00986531, -0.07117795,  0.02145686, -0.0356993 , -0.02422255,
                        0.0993618 ,  0.01780521, -0.01642674, -0.00256466, -0.05318114])
    assert np.allclose(csd[5, :10], target)

    target = np.array([ 0.01145994,  0.00207419, -0.00875817, -0.00167645,  0.00080345,
                        -0.00754003, -0.01884328, -0.00144315,  0.00474964,  0.01824842])
    assert np.allclose(smooth_csd[5, :10], target)
