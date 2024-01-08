import matlab.engine
import numpy as np

from util import get_spm_path, matlab_context, load_meg_sensor_data
from surf import smoothmesh_multilayer_mm


def coregister(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, mat_eng=None):
    """
    Run head coregistration.

    This function interfaces with MATLAB to perform head coregistration on MEG/EEG data using an MRI and mesh

    Parameters:
    nas (list): NASion fiducial coordinates.
    lpa (list): Left PreAuricular fiducial coordinates.
    rpa (list): Right PreAuricular fiducial coordinates.
    mri_fname (str): Filename of the MRI data.
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    spm_path = get_spm_path()

    with matlab_context(mat_eng) as eng:
        eng.coregister(
            data_fname,
            mri_fname,
            mesh_fname,
            matlab.double(nas),
            matlab.double(lpa),
            matlab.double(rpa),
            spm_path,
            nargout=0
        )


def compute_gain_matrix(data_fname, out_fname, mat_eng=None):
    """
    Precompute gain matrix for inversion.

    This function interfaces with MATLAB to compute the gain matrix for MEG/EEG data. Coregistration must have already
    been run on the data

    Parameters:
    data_fname (str): Filename of the MEG/EEG data.
    out_fname (str): Filename to save the gain matrix to (.mat).
    data_fname (str): Filename of the MEG/EEG data.
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    spm_path = get_spm_path()

    with matlab_context(mat_eng) as eng:
        eng.compute_lgainmat(
            data_fname,
            out_fname,
            spm_path,
            nargout=0
        )


def invert_ebb(mesh_fname, data_fname, n_layers, patch_size=5, n_temp_modes=4, foi=None, woi=None, n_folds=1,
               ideal_pc_test=0, gain_mat_fname=None, mat_eng=None, return_mu_matrix=False):
    """
    Run the Empirical Bayesian Beamformer (EBB) source reconstruction algorithm.

    This function interfaces with MATLAB to perform EBB source reconstruction on MEG/EEG data. It involves mesh
    smoothing and running the EBB algorithm in MATLAB. The MEG/EEG data must already be coregistered with the given
    mesh.

    Parameters:
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    n_layers (int): Number of layers in the mesh.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 4.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    woi (list, optional): Window of interest as [start, end]. Default is [-np.inf, np.inf].
    n_folds (int): Number of cross validation folds. Must be >1 for cross validation error
    ideal_pc_test (float): Percentage of channels to leave out (ideal because need an integer number of channels)
    gain_mat_fname (str, optional): Filename of the precomputed gain matrix. If None, it will be computed. Default is
                                    None
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    return_mu_matrix (boolean, optional): Whether or not to return the matrix needed to reconstruct source activity.
                                          Default is False

    Returns:
    list: A list containing the free energy, cross validation error (cv_err), and the matrix needed to reconstruct
    source activity (mu_matrix; if return_mu_matrix is True).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    if woi is None:
        woi = [-np.inf, np.inf]
    if foi is None:
        foi = [0, 256]
    if gain_mat_fname is None:
        gain_mat_fname=''

    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    with matlab_context(mat_eng) as eng:

        if return_mu_matrix:
            free_energy, cv_err, mu_matrix = eng.invert_ebb(
                data_fname,
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                gain_mat_fname,
                spm_path,
                nargout=3
            )
            ret_vals = [free_energy, np.array(cv_err), np.array(mu_matrix)]
        else:
            free_energy, cv_err = eng.invert_ebb(
                data_fname,
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                gain_mat_fname,
                spm_path,
                nargout=2
            )
            ret_vals = [free_energy, np.array(cv_err)]

    return ret_vals


def invert_msp(mesh_fname, data_fname, n_layers, priors=None, patch_size=5, n_temp_modes=4, foi=None,
               woi=None, n_folds=1, ideal_pc_test=0, gain_mat_fname=None, mat_eng=None, return_mu_matrix=False):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm.

    This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data. It involves mesh
    smoothing and running the MSP algorithm in MATLAB. The MEG/EEG data must already be coregistered with the given
    mesh.

    Parameters:
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    n_layers (int): Number of layers in the mesh.
    priors (list, optional): Indices of vertices to be used as priors. Default is an empty list.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 4.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    woi (list, optional): Window of interest as [start, end]. Default is [-np.inf, np.inf].
    n_folds (int): Number of cross validation folds. Must be >1 for cross validation error
    ideal_pc_test (float): Percentage of channels to leave out (ideal because need an integer number of channels)
    gain_mat_fname (str, optional): Filename of the precomputed gain matrix. If None, it will be computed. Default is
                                    None
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    return_mu_matrix (boolean, optional): Whether or not to return the matrix needed to reconstruct source activity.
                                          Default is False

    Returns:
    list: A list containing the free energy, cross validation error (cv_err), and the matrix needed to reconstruct
          source activity (mu_matrix; if return_mu_matrix is True).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - Priors are adjusted by adding 1 to each index to align with MATLAB's 1-based indexing.
    """
    if foi is None:
        foi = [0, 256]
    if priors is None:
        priors = []
    if woi is None:
        woi = [-np.inf, np.inf]
    if gain_mat_fname is None:
        gain_mat_fname=''

    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    priors = [x + 1 for x in priors]
    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    with matlab_context(mat_eng) as eng:
        if return_mu_matrix:
            free_energy, cv_err, mu_matrix = eng.invert_msp(
                data_fname,
                matlab.double(priors),
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                gain_mat_fname,
                spm_path,
                nargout=3
            )
            ret_vals = [free_energy, np.array(cv_err), np.array(mu_matrix)]
        else:
            free_energy, cv_err = eng.invert_msp(
                data_fname,
                matlab.double(priors),
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                gain_mat_fname,
                spm_path,
                nargout=2
            )
            ret_vals = [free_energy, np.array(cv_err)]

    return ret_vals


def invert_sliding_window(prior, mesh_fname, data_fname, n_layers, patch_size=5, n_temp_modes=1, win_size=16,
                          win_overlap=True, foi=None, hann=True, gain_mat_fname=None, mat_eng=None):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm in a sliding time window.

    This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data within sliding time
    windows. It involves mesh smoothing and running the MSP algorithm in MATLAB for each time window. The MEG/EEG data
    must already be coregistered with the given mesh.

    Parameters:
    prior (float): Index of the vertex to be used as a prior.
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    n_layers (int): Number of layers in the mesh.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 1.
    win_size (float, optional): Size of the sliding window in ms. Default is 16. If you increase win_size, you may
                                have to increase n_temp_modes.
    win_overlap (bool, optional): Whether the windows should overlap. Default is True.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    hann (bool, optional): Whether or not to use Hann windowing. Default is True
    gain_mat_fname (str, optional): Filename of the precomputed gain matrix. If None, it will be computed. Default is
                                    None
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    list: A list containing the free energy time series (free_energy), and the windows of interest (wois).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
    """
    if foi is None:
        foi = [0, 256]
    if gain_mat_fname is None:
        gain_mat_fname=''

    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    prior = prior + 1.0

    with matlab_context(mat_eng) as eng:
        free_energy, wois = eng.invert_sliding_window(
            float(prior),
            data_fname,
            float(patch_size),
            float(n_temp_modes),
            float(win_size),
            win_overlap,
            matlab.double(foi),
            int(hann),
            gain_mat_fname,
            spm_path,
            nargout=2
        )

    return [np.array(free_energy), np.array(wois)]


def load_source_time_series(data_fname, mu_matrix=None, inv_fname=None, vertices=None, mat_eng=None):
    """
    Load source time series data from specified vertices using precomputed inverse solutions or a lead field matrix.

    This function interfaces with MATLAB to extract time series data from specific vertices, based on precomputed
    inverse solutions, or computes the source time series using a provided lead field matrix.

    Parameters:
    data_fname (str): Filename or path of the MEG/EEG data file.
    mu_matrix (ndarray, optional): Lead field matrix (source x sensor). Default is None.
    inv_fname (str, optional): Filename or path of the file containing the inverse solutions. Default is None.
    vertices (list of int, optional): List of vertex indices from which to extract time series data. Default is None,
                                      which implies all vertices will be used.
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    ndarray: An array containing the extracted source time series data (sources x time x trial).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If 'inv_fname' is not provided, and 'mu_matrix' is None, the inverse solution from the MEG/EEG data file specified
      by 'data_fname' will be used.
    - If 'mu_matrix' is provided, the function will compute the source time series directly using the lead field matrix,
      without the need for precomputed inverse solutions.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - Vertex indices are adjusted by adding 1 to each index to align with MATLAB's 1-based indexing when interfacing with
      MATLAB.
    """
    if vertices is None:
        vertices = []

    spm_path = get_spm_path()

    if mu_matrix is None:
        vertices = [x + 1 for x in vertices]
        if inv_fname is None:
            inv_fname = data_fname

        with matlab_context(mat_eng) as eng:
            source_ts = eng.load_source_time_series(
                data_fname,
                inv_fname,
                matlab.double(vertices),
                spm_path,
                nargout=1
            )
    else:
        sensor_data = load_meg_sensor_data(data_fname, mat_eng=mat_eng)
        v_idx = np.arange(mu_matrix.shape[0])
        if len(vertices):
            v_idx = np.array(vertices)
        source_ts = np.zeros((len(v_idx), sensor_data.shape[1], sensor_data.shape[2]))
        for t in range(sensor_data.shape[2]):
            source_ts[:, :, t] = mu_matrix[v_idx, :] @ sensor_data[:, :, t]

    return source_ts
