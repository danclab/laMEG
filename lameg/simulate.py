"""
Simulation tools for MEG data generation using SPM.

This module provides functions for generating synthetic MEG datasets based on user-specified
source parameters and an existing SPM forward model. It enables controlled simulations of both
current density fields and discrete dipoles, with configurable spatial, temporal, and noise
properties. These simulations are intended for testing source reconstruction pipelines, validating
laminar inversion methods, and assessing model sensitivity under known ground truth conditions.

Key functionalities
-------------------
- **Current density simulations:** Generate distributed cortical activity from one or more
  vertices with specified dipole moments, patch sizes, and time series, projected through the
  SPM forward model.
- **Dipole simulations:** Create focal dipolar sources with defined orientations, amplitudes,
  and spatial extents, optionally unique across trials.
- **Noise control:** Inject Gaussian sensor noise to achieve a user-defined signal-to-noise ratio
  (SNR).
- **Trial averaging:** Automatically perform trial-level averaging via SPM's `spm_eeg_averaging`.

Notes
-----
- All simulations require a valid SPM M/EEG dataset containing a forward model.
- Both single- and multi-source simulations are supported.
- The generated data can be used to benchmark inversion algorithms (e.g., EBB, MSP),
  estimate laminar sensitivity, or evaluate spatial leakage and depth bias.
- Simulations are executed via SPM's `spm_eeg_simulate` interface, which ensures
  biophysically realistic projection of cortical activity to MEG sensors.
"""

import os
import shutil
import tempfile

import numpy as np
from scipy.io import savemat

from lameg.invert import check_inversion_exists, load_forward_model_vertices
from lameg.util import spm_context
import matlab # pylint: disable=wrong-import-order,import-error


def run_current_density_simulation(data_file, prefix, sim_vertices, sim_signals, dipole_moments,
                                   sim_patch_sizes, snr, sim_woi=None, average_trials=False,
                                   spm_instance=None):
    """
    Simulate laminar current density data using SPM's MEG forward model.

    This function generates synthetic MEG datasets by projecting user-defined source time series
    (signals) from specified cortical vertices through an existing SPM forward model. Each
    simulated source is defined by its vertex location, dipole moment, and patch extent. Gaussian
    noise is added at the sensor level to achieve the desired signal-to-noise ratio (SNR), and the
    simulated data can optionally be averaged across trials.

    Parameters
    ----------
    data_file : str
        Path to an existing SPM M/EEG `.mat` file containing a valid forward model, used as the
        simulation template.
    prefix : str
        Prefix for the output simulated dataset (e.g., 'sim_').
    sim_vertices : int or list of int
        Indices of source vertices (0-based) defining the locations of simulated current sources.
    sim_signals : np.ndarray, shape (n_sources, n_times)
        Time series of the simulated source activity for each vertex.
    dipole_moments : float or list of float
        Dipole moments (in nA·m) associated with each simulated source.
    sim_patch_sizes : float or list of float
        Patch sizes (in mm) around each vertex that define spatial spread of the simulated sources.
    snr : float
        Desired signal-to-noise ratio at the sensor level (in dB).
    sim_woi : list of float, optional
        Window of interest for the simulation, specified as [start, end] in milliseconds.
        Default is [-np.inf, np.inf].
    average_trials : bool, optional
        Whether to average across simulated trials (default: False).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Returns
    -------
    sim_fname : str
        Full path to the generated simulated MEG dataset (`.mat` file).

    Notes
    -----
    - Uses SPM's `spm_eeg_simulate` function to project source-level activity through the forward
      model and add Gaussian noise.
    - Supports both single and multiple simulated sources with different dipole moments or spatial
      extents.
    - The forward model (lead field) must be precomputed and stored within `data_file`.
    - If `average_trials=True`, the simulated data are averaged using SPM's `spm_eeg_averaging`
      routine and saved with prefix `'m'`.
    - This function is useful for validating inversion methods, testing model recovery, and
      quantifying depth sensitivity under controlled SNR conditions.
    """

    if sim_woi is None:
        sim_woi = [-np.inf, np.inf]

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]
    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]
    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    check_inversion_exists(data_file)
    verts = load_forward_model_vertices(data_file)

    sim_coords = np.zeros((len(sim_vertices),3))
    for c_idx,i in enumerate(sim_vertices):
        sim_coords[c_idx,:]=verts[i, :]

    with spm_context(spm_instance) as spm:
        spm.spm_eeg_simulate(
            data_file,
            prefix,
            matlab.double(sim_coords.tolist()),
            matlab.double(sim_signals.tolist()),
            matlab.double([]),
            matlab.double(sim_woi),
            matlab.double([]),
            float(snr),
            matlab.double([]),
            matlab.double([]),
            matlab.double(sim_patch_sizes),
            matlab.double(dipole_moments),
            nargout=0
        )

    data_dir = os.path.dirname(data_file)
    data_fname = os.path.split(os.path.splitext(data_file)[0])[1]
    sim_fname= os.path.join(data_dir, f'{prefix}{data_fname}.mat')

    if average_trials:
        cfg = {
            "spm": {
                "meeg": {
                    "averaging": {
                        "average": {
                            "D": np.asarray([sim_fname], dtype="object"),
                            "userobust": {
                                "standard": 0
                            },
                            "plv": 0,
                            "prefix": 'm'
                        }
                    }
                }
            }
        }

        cfg = {"matlabbatch": [cfg]}
        file, name = tempfile.mkstemp(suffix=".mat")
        savemat(file, cfg)
        spm.spm_standalone(
            "eval",
            f"""
            load('{name}'); 
            spm('defaults', 'EEG'); 
            spm_get_defaults('cmdline',1); 
            spm_jobman('run', matlabbatch);
            """,
            nargout=0
        )
        os.remove(name)
        sim_fname = os.path.join(data_dir, f'm{prefix}{data_fname}.mat')

    return sim_fname


def run_dipole_simulation(data_file, prefix, sim_vertices, sim_signals, dipole_orientations,
                          dipole_moments, sim_patch_sizes, snr, sim_woi=None, average_trials=False,
                          spm_instance=None):
    """
    Simulate dipole-level MEG data using SPM's forward model.

    This function generates synthetic MEG datasets by projecting source-level dipole activity
    through an existing SPM forward model. Each dipole is defined by its cortical vertex location,
    orientation, moment, and patch extent. The resulting MEG data are corrupted with Gaussian
    noise at the sensor level to achieve a specified signal-to-noise ratio (SNR), and can
    optionally be averaged across trials.

    Parameters
    ----------
    data_file : str
        Path to an existing SPM M/EEG `.mat` file containing a valid forward model, used as the
        simulation template.
    prefix : str
        Prefix for the output simulated dataset (e.g., `'sim_'`).
    sim_vertices : int or list of int
        Indices of source vertices (0-based) where dipoles are positioned.
    sim_signals : np.ndarray
        Array of simulated dipole time courses, either shaped (n_dipoles × n_times) or
        (n_dipoles × n_times × n_trials).
    dipole_orientations : np.ndarray, shape (n_dipoles, 3)
        Array specifying the 3D orientation vector for each dipole in head coordinates.
    dipole_moments : float or list of float
        Dipole moments (in nA·m) for each simulated source.
    sim_patch_sizes : float or list of float
        Spatial extent of the simulated dipole patch (in mm).
    snr : float
        Desired signal-to-noise ratio at the sensor level (in dB).
    sim_woi : list of float, optional
        Simulation window of interest, specified as [start, end] in milliseconds.
        Default is [-np.inf, np.inf].
    average_trials : bool, optional
        Whether to average the simulated data over trials (default: False).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Returns
    -------
    sim_fname : str
        Full path to the generated simulated MEG dataset (`.mat` file).

    Notes
    -----
    - Uses SPM's `spm_eeg_simulate` function to generate sensor-level data from specified dipole
      configurations.
    - Supports multiple dipoles with distinct orientations, moments, or spatial extents.
    - The forward model (`lead field`) must already exist in `data_file`.
    - If `average_trials=True`, trial-averaged data are computed via SPM's
      `spm_eeg_averaging` and saved with the `'m'` prefix.
    - This function is primarily used for testing inversion accuracy, source localization
      performance, and evaluating laminar sensitivity under controlled SNR conditions.
    """

    if sim_woi is None:
        sim_woi = [-np.inf, np.inf]

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]

    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]

    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    check_inversion_exists(data_file)
    verts = load_forward_model_vertices(data_file)

    sim_coords = np.zeros((len(sim_vertices), 3))
    for c_idx, i in enumerate(sim_vertices):
        sim_coords[c_idx, :] = verts[i, :]

    with spm_context(spm_instance) as spm:
        spm.spm_eeg_simulate(
            data_file,
            prefix,
            matlab.double(sim_coords.tolist()),
            matlab.double(sim_signals.tolist()),
            matlab.double(dipole_orientations.tolist()),
            matlab.double(sim_woi),
            matlab.double([]),
            float(snr),
            matlab.double([]),
            matlab.double([]),
            matlab.double(sim_patch_sizes),
            matlab.double(dipole_moments),
            nargout=0
        )

        data_dir = os.path.dirname(data_file)
        data_fname = os.path.split(os.path.splitext(data_file)[0])[1]
        sim_fname = os.path.join(data_dir, f'{prefix}{data_fname}.mat')

        if average_trials:
            cfg = {
                "spm": {
                    "meeg": {
                        "averaging": {
                            "average": {
                                "D": np.asarray([sim_fname], dtype="object"),
                                "userobust": {
                                    "standard": 0
                                },
                                "plv": 0,
                                "prefix": 'm'
                            }
                        }
                    }
                }
            }

            cfg = {"matlabbatch": [cfg]}
            file, name = tempfile.mkstemp(suffix=".mat")
            savemat(file, cfg)
            spm.spm_standalone(
                "eval",
                f"""
                load('{name}'); 
                spm('defaults', 'EEG'); 
                spm_get_defaults('cmdline',1); 
                spm_jobman('run', matlabbatch);
                """,
                nargout=0
            )
            os.remove(name)
            sim_fname = os.path.join(data_dir, f'm{prefix}{data_fname}.mat')

    return sim_fname


def setup_opm_simulation(data_file, surf_set, s_rate=1000, wholehead=True, sensor_spacing=35,
                         sensor_offset=6.5, n_samples=1000, n_trials=1, density_checks=40, axes=1,
                         sensor_positions_file=None, iskull_fname=None, oskull_fname=None,
                         scalp_fname=None, spm_instance=None):
    """
    Configure and setup an SPM OPM (Optically Pumped Magnetometer) data file.

    This function wraps the MATLAB routine :func:`spm_opm_sim` to generate synthetic MEG data
    using user-specified anatomical meshes, sensor geometry, and simulation parameters.
    It prepares the required input structure, and executes the SPM simulation within the
    provided SPM standalone context.

    Parameters
    ----------
    data_file : str
        Path to the intended output data file (used to determine simulation name and destination).
    surf_set : LayerSurfaceSet
        Surface set object providing the cortical mesh and MRI reference.
        The pial surface from ``surf_set`` is used as the cortical source model.
    s_rate : int, optional
        Sampling frequency in Hz (default: 1000).
    wholehead : bool, optional
        If True, sensor array covers the entire scalp; otherwise, sensors are restricted
        to the upper hemisphere (default: True).
    sensor_spacing : float, optional
        Inter-sensor spacing in millimeters (default: 35 mm).
    sensor_offset : float, optional
        Distance between scalp and sensor array in millimeters (default: 6.5 mm).
    n_samples : int, optional
        Number of time samples per trial (default: 1000).
    n_trials : int, optional
        Number of trials to simulate (default: 1).
    density_checks : int, optional
        Number of density optimization iterations for sensor packing (default: 40).
    axes : {1, 2, 3}, optional
        Number of orthogonal measurement axes per sensor (default: 1).
    sensor_positions_file : str, optional
        Path to a `.tsv` file containing custom sensor positions and orientations.
        If None, sensors are automatically generated on the scalp surface.
    iskull_fname, oskull_fname, scalp_fname : str, optional
        Optional custom inner skull, outer skull, and scalp surface files (GIFTI format).
        If omitted, default meshes from SPM segmentation are used.
    spm_instance : spm_standalone, optional
        Active SPM standalone context for calling :func:`spm_opm_sim`. If None, a temporary
        instance is created and destroyed automatically.

    Returns
    -------
    str
        Absolute path to the generated SPM MEEG `.mat` file containing the simulated dataset.

    Notes
    -----
    - The simulation generates `.mat` and `.dat` files containing a valid SPM MEEG object.
    - Meshes must be in scanner RAS coordinates and compatible with the subject's MRI.

    Examples
    --------
    >>> sim_file = setup_opm_simulation(
    ...     data_file='/home/user/sim/test_sim.mat',
    ...     surf_set=surf_set,
    ...     srate=1000,
    ...     sensor_spacing=30,
    ...     wholehead=False
    ... )
    """
    data_dir = os.path.dirname(data_file)
    data_fname = os.path.split(os.path.splitext(data_file)[0])[1]

    config = {
        'space': float(sensor_spacing),
        'lead': float(0),
        'wholehead': float(1 if wholehead else 0),
        'offset': float(sensor_offset),
        'axis': float(axes),
        'nTrials': float(n_trials),
        'fs': float(s_rate),
        'nSamples': float(n_samples),
        'cortex': surf_set.get_mesh_path(layer_name='pial', stage='ds'),
        'sMRI': surf_set.mri_file,
        'fname': data_fname,
        'Dens': float(density_checks),
        'data': matlab.double(np.random.randn(1,n_samples,n_trials).tolist())
    }
    if sensor_positions_file is not None:
        config['positions'] = sensor_positions_file
    if scalp_fname is not None:
        config['scalp'] = scalp_fname
    if oskull_fname is not None:
        config['oskull'] = oskull_fname
    if iskull_fname is not None:
        config['iskull'] = iskull_fname

    with spm_context(spm_instance) as spm:
        spm.spm_opm_sim(config, nargout=0)

    shutil.move(
        os.path.join(f'{data_fname}.mat'),
        os.path.join(data_dir, f'{data_fname}.mat'),
    )
    shutil.move(
        os.path.join(f'{data_fname}.dat'),
        os.path.join(data_dir, f'{data_fname}.dat'),
    )
    return os.path.join(data_dir, f'{data_fname}.mat')
