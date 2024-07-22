"""
This module contains the unit tests for the `simulate` module from the `lameg` package.
"""
import os

import numpy as np
import pytest
import nibabel as nib

from lameg.simulate import run_dipole_simulation, run_current_density_simulation
from lameg.util import load_meg_sensor_data


@pytest.mark.dependency(depends=["tests/test_02_invert.py::test_invert_sliding_window"],
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

    The test depends on `test_invert_ebb` being successful, as it ensures the
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
    mesh_fname = os.path.join(test_data_path, subj_id, 'surf/multilayer.2.ds.link_vector.fixed.gii')

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
    base_fname = './output/pspm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'

    sim_fname = run_dipole_simulation(
        base_fname,
        prefix,
        sim_vertex,
        sim_signal,
        pial_unit_norm,
        dipole_moment,
        sim_patch_size,
        snr,
        average_trials=True,
        spm_instance=spm
    )

    sim_sensor_data, time, ch_names = load_meg_sensor_data(sim_fname)

    target=np.array([ 0.05684258, -0.12866193,  0.07388893,  0.2516157,   0.09264176,  0.01863054,
                      0.27522185,  0.14102148, -0.21366386, -0.04743002])
    assert np.allclose(sim_sensor_data[0,:10], target)

    target=np.array([-0.1, -0.09833333, -0.09666667, -0.095, -0.09333333,
                     -0.09166667, -0.09, -0.08833333, -0.08666667, -0.085])
    assert np.allclose(time[:10], target)

    assert ch_names[0] == 'MLC11'


@pytest.mark.dependency(depends=["tests/test_02_invert.py::test_invert_ebb"],
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

    This test is dependent on `test_invert_ebb` to ensure that required data setups
    and dependencies are appropriately configured beforehand.
    """

    # Frequency of simulated sinusoid (Hz)
    freq = 20
    # Strength of simulated activity (nAm)
    dipole_moment = 10
    # Sampling rate (must match the data file)
    s_rate = 600

    # Generate 1s of a sine wave at a sampling rate of 600Hz (to match the data file)
    time = np.linspace(0, 1, s_rate + 1)
    sim_signal = np.zeros(time.shape).reshape(1, -1)
    t_idx = np.where(time >= 0.5)[0]
    sim_signal[0, t_idx] = np.sin(time[t_idx] * freq * 2 * np.pi)

    sim_vertex = 24588
    prefix = f'sim_{sim_vertex}_current_density_pial_'

    # Size of simulated patch of activity (mm)
    sim_patch_size = 5
    # SNR of simulated data (dB)
    snr = -10

    # Generate simulated data
    base_fname='./output/spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'

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

    target=np.array([ 2.4205484,  3.9856143, -1.1225368,  6.813098,  -2.4976463,  1.4894879,
                      2.469582,   4.6006646,  3.1293263, -3.3464603])
    assert np.allclose(sim_sensor_data[0,:10,0], target)

    target=np.array([[-0.5, -0.49833333, -0.49666667, -0.495, -0.49333333,
                      -0.49166667, -0.49, -0.48833333, -0.48666667, -0.485]])
    assert np.allclose(time[:10], target)

    assert ch_names[0] == 'MLC11'
