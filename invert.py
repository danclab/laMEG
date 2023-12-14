import json

import os
import matlab.engine
import numpy as np

from util import get_spm_path
from surf import smoothmesh_multilayer_mm


def invert_ebb(out_dir, nas, lpa, rpa, mri_fname, mesh_fname, data_fname, n_layers, patch_size=5,
               n_temp_modes=4, foi=[0, 256], woi=[-np.inf, np.inf], mat_eng=None, return_MU=False):
    """
    Run the Empirical Bayesian Beamformer (EBB) source reconstruction algorithm.

    This function interfaces with MATLAB to perform EBB source reconstruction on MEG/EEG data. It involves mesh
    smoothing, coregistration, and running the EBB algorithm in MATLAB.

    Parameters:
    out_dir (str): Output directory for saving results.
    nas (list): NASion fiducial coordinates.
    lpa (list): Left PreAuricular fiducial coordinates.
    rpa (list): Right PreAuricular fiducial coordinates.
    mri_fname (str): Filename of the MRI data.
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    n_layers (int): Number of layers in the mesh.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 4.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    woi (list, optional): Window of interest as [start, end]. Default is [-np.inf, np.inf].
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    return_MU (boolean, optional): Whether or not to return the matrix needed to reconstruct source activity. Default is
                                   False

    Returns:
    list: A list containing the result filename, the free energy (F), and the matrix needed to reconstruct
    source activity (MU; if return_MU is True).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    close_matlab=False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab=True

    mesh_base=os.path.split(os.path.splitext(mesh_fname)[0])[-1]
    data_base=os.path.split(data_fname)[-1]
    coreg_fname = os.path.join(out_dir, f'{mesh_base}.{data_base}')

    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    if return_MU:
        F,MU=mat_eng.invert_ebb(data_fname, coreg_fname, mri_fname, mesh_fname, matlab.double(nas), matlab.double(lpa),
                                matlab.double(rpa), float(patch_size), float(n_temp_modes), matlab.double(foi),
                                matlab.double(woi), spm_path, nargout=2)
        ret_vals=[coreg_fname, F, MU]
    else:
        F = mat_eng.invert_ebb(data_fname, coreg_fname, mri_fname, mesh_fname, matlab.double(nas),
                               matlab.double(lpa),
                               matlab.double(rpa), float(patch_size), float(n_temp_modes), matlab.double(foi),
                               matlab.double(woi), spm_path, nargout=1)
        ret_vals=[coreg_fname, F]

    if close_matlab:
        mat_eng.close()

    return ret_vals


def invert_msp(out_dir, nas, lpa, rpa, mri_fname, mesh_fname, data_fname, n_layers, priors=[], patch_size=5,
               n_temp_modes=4, foi=[0, 256], woi=[-np.inf, np.inf], mat_eng=None, return_MU=False):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm.

    This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data. It involves mesh
    smoothing, coregistration, and running the MSP algorithm in MATLAB.

    Parameters:
    out_dir (str): Output directory for saving results.
    nas (list): NASion fiducial coordinates.
    lpa (list): Left PreAuricular fiducial coordinates.
    rpa (list): Right PreAuricular fiducial coordinates.
    mri_fname (str): Filename of the MRI data.
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    n_layers (int): Number of layers in the mesh.
    priors (list, optional): Indices of vertices to be used as priors. Default is an empty list.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 4.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    woi (list, optional): Window of interest as [start, end]. Default is [-np.inf, np.inf].
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    return_MU (boolean, optional): Whether or not to return the matrix needed to reconstruct source activity. Default is
                                   False

    Returns:
    list: A list containing the result filename, the free energy (F), and the matrix needed to reconstruct
    source activity (MU; if return_MU is True).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - Priors are adjusted by adding 1 to each index to align with MATLAB's 1-based indexing.
    """
    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    close_matlab=False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab=True

    mesh_base=os.path.split(os.path.splitext(mesh_fname)[0])[-1]
    data_base=os.path.split(data_fname)[-1]
    coreg_fname = os.path.join(out_dir, f'{mesh_base}.{data_base}')

    priors = [x + 1 for x in priors]
    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    if return_MU:
        F,MU=mat_eng.invert_msp(data_fname, coreg_fname, mri_fname, mesh_fname, matlab.double(nas), matlab.double(lpa),
                                matlab.double(rpa), matlab.double(priors), float(patch_size), float(n_temp_modes),
                                matlab.double(foi), matlab.double(woi), spm_path, nargout=2)
        ret_vals = [coreg_fname, F, MU]
    else:
        F = mat_eng.invert_msp(data_fname, coreg_fname, mri_fname, mesh_fname, matlab.double(nas), matlab.double(lpa),
                               matlab.double(rpa), matlab.double(priors), float(patch_size), float(n_temp_modes),
                               matlab.double(foi), matlab.double(woi), spm_path, nargout=1)
        ret_vals = [coreg_fname, F]

    if close_matlab:
        mat_eng.close()

    return ret_vals


def invert_sliding_window(out_dir, prior, nas, lpa, rpa, mri_fname, mesh_fname, data_fname, n_layers, patch_size=5,
                          n_temp_modes=4, win_size=10, win_overlap=True, foi=[0, 256], mat_eng=None):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm in a sliding time window.

    This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data within sliding time
    windows. It involves mesh smoothing, coregistration, and running the MSP algorithm in MATLAB for each time window.

    Parameters:
    out_dir (str): Output directory for saving results.
    prior (float): Index of the vertex to be used as a prior.
    nas (list): NASion fiducial coordinates.
    lpa (list): Left PreAuricular fiducial coordinates.
    rpa (list): Right PreAuricular fiducial coordinates.
    mri_fname (str): Filename of the MRI data.
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    n_layers (int): Number of layers in the mesh.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 4.
    win_size (int, optional): Size of the sliding window in samples. Default is 10.
    win_overlap (bool, optional): Whether the windows should overlap. Default is True.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    list: A list containing the result filename, the free energy time series (F), and the windows of interest (wois).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
    """
    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    close_matlab = False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab = True

    prior = prior+1.0
    mesh_base = os.path.split(os.path.splitext(mesh_fname)[0])[-1]
    data_base = os.path.split(data_fname)[-1]
    coreg_fname = os.path.join(out_dir, f'{mesh_base}.{data_base}')

    F,wois = mat_eng.invert_sliding_window(float(prior), data_fname, coreg_fname, mri_fname, mesh_fname, matlab.double(nas),
                                           matlab.double(lpa), matlab.double(rpa), float(patch_size),
                                           float(n_temp_modes), float(win_size), win_overlap, matlab.double(foi),
                                           spm_path, nargout=2)

    if close_matlab:
        mat_eng.close()

    return [coreg_fname, F, wois]


def load_source_time_series(data_D, inv_D, vertices=[], mat_eng=None):
    """
    Load source time series data from specified vertices using precomputed inverse solutions.

    This function interfaces with MATLAB to extract time series data from specific vertices, based on
    precomputed inverse solutions. It is typically used in the context of MEG/EEG source analysis.

    Parameters:
    data_D (str): Filename or path of the MEG/EEG data file.
    inv_D (str): Filename or path of the file containing the inverse solutions.
    vertices (list, optional): List of vertex indices from which to extract time series data. Default is an empty list,
                            which implies all vertices.
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    ndarray: An array containing the extracted source time series data.

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - Vertex indices are adjusted by adding 1 to each index to align with MATLAB's 1-based indexing.
    """
    spm_path = get_spm_path()

    close_matlab = False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab = True

    vertices = [x + 1 for x in vertices]
    source_ts = mat_eng.load_source_time_series(data_D, inv_D, matlab.double(vertices), spm_path, nargout=1)

    if close_matlab:
        mat_eng.close()

    return source_ts

