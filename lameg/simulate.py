"""
This module facilitates the simulation of MEG data using the Statistical Parametric Mapping (SPM)
toolbox. It provides functionalities to simulate both current density and dipole-based MEG data
under varying conditions and configurations.

Key Features:

- Current Density Simulation: Allows for the simulation of current density data based on
  user-defined parameters such as vertices, signals, dipole moments, and patch sizes. Includes the
  ability to specify the signal-to-noise ratio and window of interest for the simulations.
- Dipole Simulation: Facilitates the simulation of dipole-based data, providing options to
  configure dipole orientations, moments, and noise levels. It supports the simulation of unique
  signals per trial and can average data across trials if needed.

Examples of use:

- Simulating data with specific noise levels and analyzing the impact of noise on signal processing
  algorithms.
- Generating datasets with known properties to test the efficacy of dipole fitting routines or
  source localization methods.
"""

import os
import tempfile

import numpy as np
import h5py
from scipy.io import savemat

from lameg.util import spm_context
import matlab # pylint: disable=wrong-import-order,import-error


def check_inversion_exists(file):
    """
    Check if inversion data exists in the loaded HDF5 file.

    Parameters
    ----------
    file : h5py.File
        The loaded HDF5 file object. The expected structure is file['D']['other']['inv'].

    Returns
    -------
    bool
        Returns True if inversion data exists in the file.

    Raises
    ------
    ValueError
        If the inversion data is missing or not found in the expected structure.
    KeyError
        If the required keys are missing from the file structure.
    """
    try:
        inv_data = file['D']['other']['inv']
        if inv_data is None or len(inv_data) == 0:
            raise ValueError("Inversion data not found in the file.")
    except KeyError as exc:
        raise KeyError("Inversion data structure not found in the file.") from exc
    return True


def load_vertices(file):
    """
    Load vertices from the HDF5 file based on the file version.

    Parameters
    ----------
    file : h5py.File
        The loaded HDF5 file object. The function expects the structure
        file['D']['other']['inv'][0][0]['mesh']['tess_mni']['vert'].

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the vertices loaded from the file.

    Raises
    ------
    TypeError
        If the file is not saved in a format compatible with MATLAB v7.3 or later.
    KeyError
        If the required keys are missing from the file structure.
    """
    try:
        verts = file[file['D']['other']['inv'][0][0]]['mesh']['tess_mni']['vert'][()].T
    except TypeError as exc:
        raise TypeError("This function requires the .mat file to be saved with MATLAB "
                        "v7.3 or later.") from exc
    return verts


def run_current_density_simulation(data_file, prefix, sim_vertices, sim_signals, dipole_moments,
                                   sim_patch_sizes, snr, sim_woi=None, average_trials=False,
                                   spm_instance=None):
    """
    Simulate current density data based on specified parameters.

    This function generates simulated MEG data based on specified vertices, signals, dipole
    moments, and patch sizes, incorporating a defined signal-to-noise ratio (SNR). White noise is
    added at the sensor level to yield the given SNR.

    Parameters
    ----------
    data_file : str
        Filename or path of the MEG data file used as a template for simulation.
    prefix : str
        Prefix for the output simulated data filename.
    sim_vertices : list or int
        Indices of vertices where simulations are centered. Can be a single integer or a list.
    sim_signals : ndarray
        Array of simulated signals.
    dipole_moments : list or float
        Dipole moments for the simulation. Can be a single float or a list.
    sim_patch_sizes : list or int
        Sizes of patches around each vertex for the simulation. Can be a single integer or a list.
    snr : float
        Signal-to-noise ratio for the simulation.
    sim_woi : list, optional
        Window of interest for the simulation as [start, end]. Default is [-np.inf, np.inf].
    average_trials : bool, optional
        Whether to average the simulated data over trials. Default is False.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.

    Returns
    -------
    sim_fname : str
        Filename of the generated simulated data.

    Notes
    -----
    - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    """

    if sim_woi is None:
        sim_woi = [-np.inf, np.inf]

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]
    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]
    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    with h5py.File(data_file, 'r') as file:
        check_inversion_exists(file)
        verts = load_vertices(file)

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
    Simulate dipole-based MEG data based on specified parameters.

    This function generates simulated MEG data with specific dipole configurations. It creates
    simulations based on specified vertices, signals, dipole orientations, moments, and patch
    sizes, incorporating a defined signal-to-noise ratio (SNR). White noise is added at the sensor
    level to yield the given SNR.

    Parameters
    ----------
    data_file : str
        Filename or path of the MEG data file used as a template for simulation.
    prefix : str
        Prefix for the output simulated data filename.
    sim_vertices : list or int
        Indices of vertices where simulations are centered. Can be a single integer or a list.
    sim_signals : ndarray
        Array of simulated signals. Either dipoles x time (signal will be used for each trial), or
        dipoles x time x trials (unique signal for each trial).
    dipole_orientations : ndarray
        Array of dipole orientations for the simulation.
    dipole_moments : list or float
        Dipole moments for the simulation. Can be a single float or a list.
    sim_patch_sizes : list or int
        Sizes of patches around each vertex for the simulation. Can be a single integer or a list.
    snr : float
        Signal-to-noise ratio for the simulation.
    sim_woi : list, optional
        Window of interest for the simulation as [start, end]. Default is [-np.inf, np.inf].
    average_trials : bool, optional
        Whether to average the simulated data over trials. Default is False.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.

    Returns
    -------
    sim_fname : str
        Filename of the generated simulated data.

    Notes
    -----
    - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    """

    if sim_woi is None:
        sim_woi = [-np.inf, np.inf]

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]

    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]

    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    with h5py.File(data_file, 'r') as file:
        check_inversion_exists(file)
        verts = load_vertices(file)

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
