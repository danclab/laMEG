import matlab.engine
import numpy as np

from lameg.util import get_spm_path, matlab_context


def run_current_density_simulation(data_file, prefix, sim_vertices, sim_signals, dipole_moments, sim_patch_sizes, snr,
                                   sim_woi=None, viz=True, mat_eng=None):
    """
    Simulate current density data based on specified parameters.

    This function interfaces with MATLAB to generate simulated MEG/EEG data. It creates simulations based on specified
    vertices, signals, dipole moments, and patch sizes, incorporating a defined signal-to-noise ratio (SNR). White noise
    is added at the sensor level to yield the given SNR.

    Parameters:
    data_file (str): Filename or path of the MEG/EEG data file used as a template for simulation.
    prefix (str): Prefix for the output simulated data filename.
    sim_vertices (list or int): Indices of vertices where simulations are centered. Can be a single integer or a list.
    sim_signals (ndarray): Array of simulated signals.
    dipole_moments (list or float): Dipole moments for the simulation. Can be a single float or a list.
    sim_patch_sizes (list or int): Sizes of patches around each vertex for the simulation. Can be a single integer or a
                                   list.
    snr (float): Signal-to-noise ratio for the simulation.
    sim_woi (list, optional): Window of interest for the simulation as [start, end]. Default is [-np.inf, np.inf].
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    str: Filename of the generated simulated data.

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - Vertex indices and other parameters are adjusted to align with MATLAB's 1-based indexing and data structures.
    """
    if sim_woi is None:
        sim_woi = [-np.inf, np.inf]

    spm_path = get_spm_path()

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]
    sim_vertices=[x+1 for x in sim_vertices]
    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]
    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    with matlab_context(mat_eng) as eng:
        sim_fname = eng.simulate(
            data_file,
            prefix,
            matlab.double(sim_vertices),
            matlab.double(sim_woi),
            matlab.double(sim_signals.tolist()),
            matlab.double([]),
            matlab.double(dipole_moments),
            matlab.double(sim_patch_sizes),
            float(snr),
            False,
            int(viz),
            spm_path,
            nargout=1
        )

    return sim_fname


def run_dipole_simulation(data_file, prefix, sim_vertices, sim_signals, dipole_orientations, dipole_moments,
                          sim_patch_sizes, snr, sim_woi=None, average_trials=False, viz=False, mat_eng=None):
    """
    Simulate dipole-based MEG/EEG data based on specified parameters.

    This function interfaces with MATLAB to generate simulated MEG/EEG data with specific dipole configurations. It
    creates simulations based on specified vertices, signals, dipole orientations, moments, and patch sizes,
    incorporating a defined signal-to-noise ratio (SNR). White noise is added at the sensor level to yield the given
    SNR.

    Parameters:
    data_file (str): Filename or path of the MEG/EEG data file used as a template for simulation.
    prefix (str): Prefix for the output simulated data filename.
    sim_vertices (list or int): Indices of vertices where simulations are centered. Can be a single integer or a list.
    sim_signals (ndarray): Array of simulated signals. Either dipoles x time (signal will be used for each trial), or
                           dipoles x time x trials (unique signal for each trial)
    dipole_orientations (ndarray): Array of dipole orientations for the simulation.
    dipole_moments (list or float): Dipole moments for the simulation. Can be a single float or a list.
    sim_patch_sizes (list or int): Sizes of patches around each vertex for the simulation. Can be a single integer or a
                                   list.
    snr (float): Signal-to-noise ratio for the simulation.
    sim_woi (list, optional): Window of interest for the simulation as [start, end]. Default is [-np.inf, np.inf].
    average_trials (bool, optional): Whether or not to average the simulated data over trials. Default is False.
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    str: Filename of the generated simulated data.

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - Vertex indices, dipole orientations, and other parameters are adjusted to align with MATLAB's 1-based indexing and
      data structures.
    """

    if sim_woi is None:
        sim_woi = [-np.inf, np.inf]

    spm_path = get_spm_path()

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]
    sim_vertices = [x + 1 for x in sim_vertices]
    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]
    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    with matlab_context(mat_eng) as eng:
        sim_fname=eng.simulate(
            data_file,
            prefix,
            matlab.double(sim_vertices),
            matlab.double(sim_woi),
            matlab.double(sim_signals.tolist()),
            matlab.double(dipole_orientations.tolist()),
            matlab.double(dipole_moments),
            matlab.double(sim_patch_sizes),
            float(snr),
            average_trials,
            int(viz),
            spm_path,
            nargout=1
        )

    return sim_fname