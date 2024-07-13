"""
This module contains the unit tests for the `simulate` and `laminar` modules from the `lameg`
package.
"""
import os

import numpy as np
import pytest
import nibabel as nib

from lameg.invert import coregister, invert_ebb, load_source_time_series
from lameg.laminar import model_comparison, sliding_window_model_comparison, compute_csd
from lameg.simulate import run_dipole_simulation, run_current_density_simulation
from lameg.util import get_fiducial_coords, get_surface_names, load_meg_sensor_data


@pytest.mark.dependency(depends=["tests/test_invert.py::test_invert_sliding_window"],
                        scope='session')
def test_run_dipole_simulation(spm):
    """
    Tests the `run_dipole_simulation` function to verify the accurate generation of simulated MEG
    sensor data based on a specified dipole model in a cortical mesh.

    This test function performs several key operations:
    1. Prepares a Gaussian signal representing dipole activity and uses it along with a specified
       cortical mesh to simulate MEG data.
    2. Ensures that the simulation reflects the input parameters such as dipole moment, signal
       width, and signal-to-noise ratio.
    3. Verifies that the resulting simulated sensor data matches expected target values, testing
       both the amplitude of the sensor signals and the timing of the recordings.
    4. Checks that the channel names are correctly assigned in the simulated data.

    The function relies on the SPM object provided by a session-wide fixture, ensuring that the SPM
    model is initialized only once and reused, optimizing performance across tests.

    Parameters:
        spm (object): The SPM instance used to handle brain simulation and MEG data processing.

    Steps executed in the test:
    - A Gaussian signal is generated to simulate neuronal activity.
    - The `run_dipole_simulation` function is called with the generated signal and parameters
      defining the characteristics of the simulation.
    - The output MEG sensor data is loaded and compared against predefined target values to ensure
      the simulation's accuracy.
    - The first channel name is checked to confirm the correct configuration and data handling.

    Assertions:
    - Asserts that the simulated sensor data closely matches expected values, ensuring the
      simulation's output integrity.
    - Verifies that time vectors are accurate to the specified sampling rate.
    - Checks that the channel name matches expected setup, confirming proper MEG data formatting.

    The test depends on `test_load_source_time_series` being successful, as it ensures the
    necessary data infrastructure and dependencies are correctly set up.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'

    # Strength of simulated activity (nAm)
    dipole_moment = 8
    # Temporal width of the simulated Gaussian
    signal_width = .025  # 25ms

    # Generate 200ms of a Gaussian at a sampling rate of 600Hz (to match the data file)
    time = np.linspace(0, .2, 121)
    zero_time = time[int((len(time) - 1) / 2 + 1)]
    sim_signal = np.exp(-((time - zero_time) ** 2) / (2 * signal_width ** 2)).reshape(1, -1)

    # Mesh to use for forward model in the simulations
    mesh_fname = os.path.join(test_data_path, subj_id, 'surf/pial.ds.link_vector.fixed.gii')

    # Load multilayer mesh and compute the number of vertices per layer
    mesh = nib.load(mesh_fname)

    sim_vertex = 24588
    # Orientation of the simulated dipole
    pial_unit_norm = mesh.darrays[2].data[sim_vertex, :]
    prefix = f'sim_{sim_vertex}_dipole_pial_'

    # Size of simulated patch of activity (mm)
    sim_patch_size = 5
    # SNR of simulated data (dB)
    snr = -10

    # Generate simulated data
    base_fname='./output/pspm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'

    sim_fname = run_dipole_simulation(
        base_fname,
        prefix,
        sim_vertex,
        sim_signal,
        pial_unit_norm,
        dipole_moment,
        sim_patch_size,
        snr,
        spm_instance=spm
    )

    sim_sensor_data, time, ch_names = load_meg_sensor_data(sim_fname)

    target=np.array([9.644654, 22.121809, -20.063894, 3.8661294, 3.8762872,
                     25.618711, -9.541486, 5.927515, -19.01493, 0.867059 ])
    assert np.allclose(sim_sensor_data[0,:10,0], target)

    target=np.array([-0.1, -0.09833333, -0.09666667, -0.095, -0.09333333,
                     -0.09166667, -0.09, -0.08833333, -0.08666667, -0.085])
    assert np.allclose(time[:10], target)

    assert ch_names[0] == 'MLC11'


@pytest.mark.dependency(depends=["tests/test_invert.py::test_invert_sliding_window"],
                        scope='session')
def test_run_current_density_simulation(spm):
    """
    Tests the `run_current_density_simulation` function to ensure it accurately simulates current
    density maps based on specific neural activity models and evaluates the MEG sensor data
    outputs against expected values.

    The function aims to:
    1. Simulate neuronal activity using a Gaussian waveform over a defined time course and verify
       the resulting sensor data.
    2. Ensure that the simulation parameters such as dipole moment, signal width, and
       signal-to-noise ratio (SNR) are correctly applied.
    3. Validate that the simulated data closely aligns with predefined target values, ensuring both
       temporal and spatial accuracy of the simulation.

    Parameters:
        spm (object): An instance of the initialized SPM application, used to handle the simulation
                      processes.

    Key operations in this test include:
    - Generating a Gaussian signal to represent localized brain activity.
    - Running the current density simulation using a cortical mesh and checking the output against
      expected sensor data.
    - Validating the generated sensor data and time series to ensure they match the expected
      patterns and values.
    - Confirming the correct assignment of channel names, verifying that data arrays are properly
      labeled and structured.

    Assertions:
    - Check if the first ten values of the simulated sensor data match the expected array of
      values, validating the accuracy of the simulation.
    - Ensure that the time series for the first ten data points are accurate to the specified
      sampling rate.
    - Verify that the first channel name matches the expected label, ensuring data integrity and
      proper configuration.

    This test is dependent on `test_load_source_time_series` to ensure that required data setups
    and dependencies are appropriately configured beforehand.
    """

    # Strength of simulated activity (nAm)
    dipole_moment = 8
    # Temporal width of the simulated Gaussian
    signal_width = .025  # 25ms

    # Generate 200ms of a Gaussian at a sampling rate of 600Hz (to match the data file)
    time = np.linspace(0, .2, 121)
    zero_time = time[int((len(time) - 1) / 2 + 1)]
    sim_signal = np.exp(-((time - zero_time) ** 2) / (2 * signal_width ** 2)).reshape(1, -1)

    sim_vertex = 24588
    prefix = f'sim_{sim_vertex}_current_density_pial_'

    # Size of simulated patch of activity (mm)
    sim_patch_size = 5
    # SNR of simulated data (dB)
    snr = -10

    # Generate simulated data
    base_fname='./output/pspm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'

    # Generate simulated data
    sim_fname = run_current_density_simulation(
        base_fname,
        prefix,
        sim_vertex,
        sim_signal,
        dipole_moment,
        sim_patch_size,
        snr,
        spm_instance=spm
    )

    sim_sensor_data, time, ch_names = load_meg_sensor_data(sim_fname)

    target=np.array([9.326762, 15.356956, -4.3236766, 26.251503, -9.620896,
                     5.7415423, 9.518472, 17.73018, 12.0625515, -12.886167 ])
    assert np.allclose(sim_sensor_data[0,:10,0], target)

    target=np.array([-0.1, -0.09833333, -0.09666667, -0.095, -0.09333333,
                     -0.09166667, -0.09, -0.08833333, -0.08666667, -0.085])
    assert np.allclose(time[:10], target)

    assert ch_names[0] == 'MLC11'


@pytest.mark.dependency(depends=["test_run_current_density_simulation"])
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
        11,
        os.path.join(test_data_path, subj_id, 'surf'),
        'link_vector.fixed'
    )

    sim_fname=os.path.join('./output/',
                           'sim_24588_current_density_pial_pspm-converted_autoreject-'
                           'sub-104-ses-01-001-btn_trial-epo.mat')
    patch_size = 5
    n_temp_modes = 4
    [free_energy, _] = model_comparison(
        nas,
        lpa,
        rpa,
        mri_fname,
        [layer_fnames[0], layer_fnames[-1]],
        sim_fname,
        patch_size=patch_size,
        n_temp_modes=n_temp_modes,
        spm_instance=spm
    )

    assert np.allclose(free_energy, np.array([-287707.70030694, -293264.38562064]))


@pytest.mark.dependency(depends=["test_run_dipole_simulation"])
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
    mri_fname = os.path.join(
        test_data_path,
        subj_id,
        'mri',
        's2023-02-28_13-33-133958-00001-00224-1.nii'
    )

    # Get name of each mesh that makes up the layers of the multilayer mesh - these will be used
    # for the sourcereconstruction
    layer_fnames = get_surface_names(
        11,
        os.path.join(test_data_path, subj_id, 'surf'),
        'link_vector.fixed'
    )

    sim_fname = os.path.join('./output/',
                             'sim_24588_dipole_pial_pspm-converted_autoreject-'
                             'sub-104-ses-01-001-btn_trial-epo.mat')
    patch_size = 5
    sliding_n_temp_modes = 4
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
        patch_size=patch_size,
        n_temp_modes=sliding_n_temp_modes,
        win_size=win_size,
        win_overlap=win_overlap,
        spm_instance=spm
    )

    target = np.array([[-279296.53033732, -279296.53033732, -279326.98678762,
                        -279312.46292853, -279312.46292853, -279292.45809178,
                        -279214.15994902, -279214.15994902, -279241.90254963,
                        -279253.15129888],
                       [-279296.53050795, -279296.53050795, -279326.98676623,
                        -279312.46298879, -279312.46298879, -279292.45812693,
                        -279214.1600669 , -279214.1600669 , -279241.90265108,
                        -279253.1514743 ]])
    assert np.allclose(free_energy[:,:10], target)

    target = np.array([[-100.        ,  -75.        ],
                       [-100.        ,  -73.33333333],
                       [-100.        ,  -71.66666667],
                       [-100.        ,  -70.        ],
                       [-100.        ,  -68.33333333],
                       [-100.        ,  -66.66666667],
                       [-100.        ,  -65.        ],
                       [-100.        ,  -63.33333333],
                       [-100.        ,  -61.66666667],
                       [-100.        ,  -60.        ]])
    assert np.allclose(wois[:10,:], target)
