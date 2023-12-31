import matlab.engine
import numpy as np

from util import get_spm_path, matlab_context
from surf import smoothmesh_multilayer_mm


def invert_ebb(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, n_layers, patch_size=5, n_temp_modes=4, foi=[0, 256],
               woi=[-np.inf, np.inf], n_folds=1, ideal_pc_test=0, mat_eng=None, return_MU=False):
    """
    Run the Empirical Bayesian Beamformer (EBB) source reconstruction algorithm.

    This function interfaces with MATLAB to perform EBB source reconstruction on MEG/EEG data. It involves mesh
    smoothing, coregistration, and running the EBB algorithm in MATLAB.

    Parameters:
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
    n_folds (int): Number of cross validation folds. Must be >1 for cross validation error
    ideal_pc_test (float): Percentage of channels to leave out (ideal because need an integer number of channels)
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    return_MU (boolean, optional): Whether or not to return the matrix needed to reconstruct source activity. Default is
                                   False

    Returns:
    list: A list containing the free energy (F), cross validation error (CVerr), and the matrix needed to reconstruct
    source activity (MU; if return_MU is True).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    with matlab_context(mat_eng) as eng:

        if return_MU:
            F, CVerr, MU = eng.invert_ebb(
                data_fname,
                mri_fname,
                mesh_fname,
                matlab.double(nas),
                matlab.double(lpa),
                matlab.double(rpa),
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                spm_path,
                nargout=3
            )
            ret_vals = [F, CVerr, MU]
        else:
            F, CVerr = eng.invert_ebb(
                data_fname,
                mri_fname,
                mesh_fname,
                matlab.double(nas),
                matlab.double(lpa),
                matlab.double(rpa),
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                spm_path,
                nargout=2
            )
            ret_vals = [F, CVerr]

    return ret_vals


def invert_msp(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, n_layers, priors=[], patch_size=5, n_temp_modes=4,
               foi=[0, 256], woi=[-np.inf, np.inf], n_folds=1, ideal_pc_test=0, mat_eng=None, return_MU=False):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm.

    This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data. It involves mesh
    smoothing, coregistration, and running the MSP algorithm in MATLAB.

    Parameters:
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
    n_folds (int): Number of cross validation folds. Must be >1 for cross validation error
    ideal_pc_test (float): Percentage of channels to leave out (ideal because need an integer number of channels)
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    return_MU (boolean, optional): Whether or not to return the matrix needed to reconstruct source activity. Default is
                                   False

    Returns:
    list: A list containing the free energy (F), cross validation error (CVerr), and the matrix needed to reconstruct
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

    priors = [x + 1 for x in priors]
    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    with matlab_context(mat_eng) as eng:
        if return_MU:
            F, CVerr, MU = eng.invert_msp(
                data_fname,
                mri_fname,
                mesh_fname,
                matlab.double(nas),
                matlab.double(lpa),
                matlab.double(rpa),
                matlab.double(priors),
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                spm_path,
                nargout=3
            )
            ret_vals = [F, CVerr, MU]
        else:
            F, CVerr = eng.invert_msp(
                data_fname,
                mri_fname,
                mesh_fname,
                matlab.double(nas),
                matlab.double(lpa),
                matlab.double(rpa),
                matlab.double(priors),
                float(patch_size),
                float(n_temp_modes),
                matlab.double(foi),
                matlab.double(woi),
                float(n_folds),
                float(ideal_pc_test),
                spm_path,
                nargout=2
            )
            ret_vals = [F, CVerr]

    return ret_vals


def invert_sliding_window(prior, nas, lpa, rpa, mri_fname, mesh_fname, data_fname, n_layers, patch_size=5,
                          n_temp_modes=1, win_size=10, win_overlap=True, foi=[0, 256], mat_eng=None):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm in a sliding time window.

    This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data within sliding time
    windows. It involves mesh smoothing, coregistration, and running the MSP algorithm in MATLAB for each time window.

    Parameters:
    prior (float): Index of the vertex to be used as a prior.
    nas (list): NASion fiducial coordinates.
    lpa (list): Left PreAuricular fiducial coordinates.
    rpa (list): Right PreAuricular fiducial coordinates.
    mri_fname (str): Filename of the MRI data.
    mesh_fname (str): Filename of the mesh data.
    data_fname (str): Filename of the MEG/EEG data.
    n_layers (int): Number of layers in the mesh.
    patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
    n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 1.
    win_size (int, optional): Size of the sliding window in samples. Default is 10. If you increase win_size, you may
                              have to increase n_temp_modes.
    win_overlap (bool, optional): Whether the windows should overlap. Default is True.
    foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    list: A list containing the free energy time series (F), and the windows of interest (wois).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
    """
    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    prior = prior + 1.0

    with matlab_context(mat_eng) as eng:
        F, wois = eng.invert_sliding_window(
            float(prior),
            data_fname,
            mri_fname,
            mesh_fname,
            matlab.double(nas),
            matlab.double(lpa),
            matlab.double(rpa),
            float(patch_size),
            float(n_temp_modes),
            float(win_size),
            win_overlap,
            matlab.double(foi),
            spm_path,
            nargout=2
        )

    return [F, wois]


def load_source_time_series(data_D, inv_D=None, vertices=[], mat_eng=None):
    """
    Load source time series data from specified vertices using precomputed inverse solutions.

    This function interfaces with MATLAB to extract time series data from specific vertices, based on
    precomputed inverse solutions. It is typically used in the context of MEG/EEG source analysis.

    Parameters:
    data_D (str): Filename or path of the MEG/EEG data file.
    inv_D (str, optional): Filename or path of the file containing the inverse solutions. Default is None.
    vertices (list, optional): List of vertex indices from which to extract time series data. Default is an empty list,
                               which implies all vertices.
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    ndarray: An array containing the extracted source time series data.

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If 'inv_D' is not provided, the inverse solution from the MEG/EEG data file specified by 'data_D' will be used
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - Vertex indices are adjusted by adding 1 to each index to align with MATLAB's 1-based indexing.
    """
    spm_path = get_spm_path()

    vertices = [x + 1 for x in vertices]
    if inv_D is None:
        inv_D = data_D

    with matlab_context(mat_eng) as eng:
        source_ts = eng.load_source_time_series(
            data_D,
            inv_D,
            matlab.double(vertices),
            spm_path,
            nargout=1
        )

    return source_ts
