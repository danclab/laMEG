import numpy as np

from invert import invert_ebb, invert_msp, invert_sliding_window, coregister
from util import matlab_context


def model_comparison(nas, lpa, rpa, mri_fname, mesh_fnames, data_fname, patch_size=5, n_temp_modes=4, foi=None,
                     woi=None, method='EBB', priors=None, n_folds=1, ideal_pc_test=0,
                     gain_mat_fnames=None, viz=True, mat_eng=None):
    """
    Compare model fits using different meshes by computing the free energy.

    This function runs source reconstruction algorithms (either Empirical Bayesian Beamformer or Multiple Sparse Priors)
    on a set of meshes and compares their model fits using the free energy and cross validation error metrics.

    Parameters:
    nas (list): NASion fiducial coordinates.
    lpa (list): Left PreAuricular fiducial coordinates.
    rpa (list): Right PreAuricular fiducial coordinates.
    mri_fname (str): Filename of the MRI data.
    mesh_fnames (list): List of filenames for different meshes.
    data_fname (str): Filename of the MEG/EEG data.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 4.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    woi (list, optional): Window of interest as [start, end]. Default is [-np.inf, np.inf].
    method (str, optional): Source reconstruction method, either 'EBB' or 'MSP'. Default is 'EBB'.
    priors (list, optional): Indices of vertices to be used as priors (only for MSP). Default is an empty list.
    n_folds (int): Number of cross validation folds. Must be >1 for cross validation error
    ideal_pc_test (float): Percentage of channels to leave out (ideal because need an integer number of channels)
    gain_mat_fnames (list, optional): List of filenames of the precomputed gain matrices, one for each mesh in
                                      mesh_fnames. If None, they will be computed. Default is None
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    list: A list containing the free energy values corresponding to each mesh, and the cross validation error for each
          mesh

    Notes:
    - Free energy is used as a measure of model fit, with higher values indicating better fit.
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    if priors is None:
        priors = []
    if woi is None:
        woi = [-np.inf, np.inf]
    if foi is None:
        foi = [0, 256]
    if gain_mat_fnames is None:
        gain_mat_fnames=[None for _ in mesh_fnames]

    f_vals = []
    cv_errs = []
    with matlab_context(mat_eng) as eng:
        for l_idx, mesh_fname in enumerate(mesh_fnames):
            coregister(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, viz=viz, mat_eng=eng)
            if method == 'EBB':
                [f_val, cv_err] = invert_ebb(mesh_fname, data_fname, 1, patch_size=patch_size,
                                             n_temp_modes=n_temp_modes, foi=foi, woi=woi, n_folds=n_folds,
                                             ideal_pc_test=ideal_pc_test, gain_mat_fname=gain_mat_fnames[l_idx],
                                             viz=viz, mat_eng=eng)
            elif method == 'MSP':
                [f_val, cv_err] = invert_msp(mesh_fname, data_fname, 1, priors=priors, patch_size=patch_size,
                                             n_temp_modes=n_temp_modes, foi=foi, woi=woi, n_folds=n_folds,
                                             ideal_pc_test=ideal_pc_test, gain_mat_fname=gain_mat_fnames[l_idx],
                                             viz=viz, mat_eng=eng)
            f_vals.append(f_val)
            cv_errs.append(cv_err)

    f_vals = np.array(f_vals)
    cv_errs = np.array(cv_errs)

    return f_vals, cv_errs


def sliding_window_model_comparison(prior, nas, lpa, rpa, mri_fname, mesh_fnames, data_fname, patch_size=5,
                                    n_temp_modes=1, foi=None, win_size=16, win_overlap=True, hann=True,
                                    gain_mat_fnames=None, viz=True, mat_eng=None):
    """
    Compare model fits across different meshes using a sliding window approach.

    This function runs source reconstruction using the Multiple Sparse Priors (MSP) method in sliding time windows on a
    set of meshes. It compares the model fits for each mesh by computing the free energy in each window.

    Parameters:
    prior (float): Index of the vertex to be used as a prior.
    nas (list): NASion fiducial coordinates.
    lpa (list): Left PreAuricular fiducial coordinates.
    rpa (list): Right PreAuricular fiducial coordinates.
    mri_fname (str): Filename of the MRI data.
    mesh_fnames (list): List of filenames for different meshes.
    data_fname (str): Filename of the MEG/EEG data.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 1.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    win_size (float, optional): Size of the sliding window in ms. Default is 16. If you increase win_size, you may
                                have to increase n_temp_modes.
    win_overlap (bool, optional): Whether the windows should overlap. Default is True.
    hann (bool, optional): Whether or not to use Hann windowing. Default is True
    gain_mat_fnames (list, optional): List of filenames of the precomputed gain matrices, one for each mesh in
                                      mesh_fnames. If None, they will be computed. Default is None
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    tuple: A tuple containing a list of free energy values for each mesh and the windows of interest (wois).

    Notes:
    - Free energy is used as a measure of model fit, with higher values indicating better fit.
    - The function requires MATLAB and SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
    """
    if foi is None:
        foi = [0, 256]
    if gain_mat_fnames is None:
        gain_mat_fnames=[None for _ in mesh_fnames]

    f_vals = []
    wois = []
    with matlab_context(mat_eng) as eng:
        for l_idx, mesh_fname in enumerate(mesh_fnames):
            coregister(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, viz=viz, mat_eng=eng)

            [mesh_fvals, wois] = invert_sliding_window(prior, mesh_fname, data_fname, 1, patch_size=patch_size,
                                                       n_temp_modes=n_temp_modes, foi=foi, win_size=win_size,
                                                       win_overlap=win_overlap, hann=hann,
                                                       gain_mat_fname=gain_mat_fnames[l_idx], viz=viz, mat_eng=eng)
            f_vals.append(mesh_fvals)

    f_vals=np.array(f_vals)

    return f_vals, wois
