import h5py
# import matlab.engine
import numpy as np
from scipy.io import savemat
from scipy.sparse import csc_matrix
import tempfile
import os

# from lameg.util import get_spm_path, matlab_context, load_meg_sensor_data
from lameg.util import load_meg_sensor_data
from lameg.surf import smoothmesh_multilayer_mm
import spm_standalone


def coregister(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, viz=True):
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
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
    """
    # Define the structured dtype for 'specification' with nested 'type'
    spec_dtype = np.dtype([('type', 'O')])

    # Define the structured dtype for the top-level 'fiducial'
    fid_dtype = np.dtype([
        ('fidname', 'O'),
        ('specification', spec_dtype)
    ])

    # Construct the structured array correctly for each fiducial entry
    fiducials = np.zeros(3, dtype=fid_dtype)
    fiducials['fidname'] = ['nas', 'lpa', 'rpa']
    fiducials['specification']['type'][0] = np.array(nas,
                                                     dtype=float)
    fiducials['specification']['type'][1] = np.array(lpa,
                                                     dtype=float)
    fiducials['specification']['type'][2] = np.array(rpa,
                                                     dtype=float)

    cfg = {
        "spm": {
            "meeg": {
                "source": {
                    "headmodel": {
                        "D": np.asarray([data_fname],
                                        dtype="object"),
                        "val": 1,
                        "comment": "",
                        "meshing": {
                            "meshes": {
                                "custom": {
                                    "mri": np.asarray(
                                        [f'{mri_fname},1'],
                                        dtype="object"),
                                    "cortex": np.asarray(
                                        [mesh_fname],
                                        dtype="object"),
                                    "iskull": np.asarray([''], dtype="object"),
                                    "oskull": np.asarray([''], dtype="object"),
                                    "scalp": np.asarray([''], dtype="object")
                                }
                            },
                            "meshres": 2
                        },
                        "coregistration": {
                            "coregspecify": {
                                "fiducial": fiducials,
                                "useheadshape": 0
                            }
                        },
                        "forward": {
                            "eeg": 'EEG BEM',
                            "meg": 'Single Shell'
                        }
                    }
                }
            }
        }

    }
    cfg = {"matlabbatch": [cfg]}
    f, name = tempfile.mkstemp(suffix=".mat")
    savemat(f, cfg)
    spm = spm_standalone.initialize()
    spm.spm_standalone("eval",
                       f"load('{name}'); spm('defaults', 'EEG'); spm_get_defaults('cmdline',{int(not viz)}); spm_jobman('run', matlabbatch);",
                       nargout=0)
    os.remove(name)
    spm.terminate()


def invert_ebb(mesh_fname, data_fname, n_layers, patch_size=5, n_temp_modes=4, foi=None, woi=None, n_folds=1,
               ideal_pc_test=0, viz=True, return_mu_matrix=False):
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
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
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

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    # Extract directory name and file name without extension
    data_dir, fname_with_ext = os.path.split(data_fname)
    fname, _ = os.path.splitext(fname_with_ext)

    # Construct new file name with added '_testmodes.mat'
    spatialmodesname = os.path.join(data_dir, f'{fname}_testmodes.mat')
    spm = spm_standalone.initialize()
    spatialmodename,Nmodes,pctest = spm.spm_eeg_inv_prep_modes_xval(data_fname, [], spatialmodesname, n_folds, ideal_pc_test, nargout=3)

    cfg = {
        "spm": {
            "meeg": {
                "source": {
                    "invertiter": {
                        "D": np.asarray([data_fname], dtype="object"),
                        "val": 1,
                        "whatconditions": {
                            "all": 1
                        },
                        "isstandard": {
                            "custom": {
                                "invfunc": 'Classic',
                                "invtype": 'EBB',
                                "woi": np.array(woi, dtype=float),
                                "foi": np.array(foi, dtype=float),
                                "hanning": float(0),
                                "isfixedpatch": {
                                    "randpatch": {
                                        "npatches": float(512),
                                        "niter": float(1)
                                    }
                                },
                                "patchfwhm": -float(patch_size),
                                "mselect": float(0),
                                "nsmodes": float(Nmodes),
                                "umodes": np.asarray([spatialmodename], dtype="object"),
                                "ntmodes": float(n_temp_modes),
                                "priors": {
                                    "priorsmask": np.asarray([''], dtype="object"),
                                    "space": 0
                                },
                                "restrict": {
                                    "locs": np.zeros((0, 3), dtype=float),
                                    "radius": float(32)
                                },
                                "outinv": '',
                            }
                        },
                        "modality": np.asarray(['All'], dtype="object"),
                        "crossval": np.asarray([pctest, n_folds], dtype=float)
                    }
                }
            }
        }
    }

    cfg = {"matlabbatch": [cfg]}
    f, name = tempfile.mkstemp(suffix=".mat")
    savemat(f, cfg)
    spm.spm_standalone("eval",
                       f"load('{name}'); matlabbatch{{1}}.spm.meeg.source.invertiter.isstandard.custom.restrict.locs=zeros(0,3); spm('defaults', 'EEG'); spm_get_defaults('cmdline',{int(not viz)}); spm_jobman('run', matlabbatch);",
                       nargout=0)
    os.remove(name)

    with h5py.File(data_fname, 'r') as file:
        free_energy=file[file['D']['other']['inv'][0][0]]['inverse']['crossF'][()]
        cv_err=file[file['D']['other']['inv'][0][0]]['inverse']['crosserr'][()]

        m_data = file[file['D']['other']['inv'][0][0]]['inverse']['M']['data'][()]
        m_ir = file[file['D']['other']['inv'][0][0]]['inverse']['M']['ir'][()]
        m_jc = file[file['D']['other']['inv'][0][0]]['inverse']['M']['jc'][()]
        U = file[file[file['D']['other']['inv'][0][0]]['inverse']['U'][0][0]][()]

    if not return_mu_matrix:
        return [free_energy, cv_err]
    else:
        # Reconstruct the sparse matrix
        num_rows = int(max(m_ir)) + 1  # Assuming 0-based indexing in Python
        num_cols = len(m_jc) - 1  # The number of columns is one less than the length of jc
        M = csc_matrix((m_data, m_ir, m_jc), shape=(num_rows, num_cols))
        mu_matrix = (M @ U)
        return [free_energy, cv_err, mu_matrix]


#
# def invert_msp(mesh_fname, data_fname, n_layers, priors=None, patch_size=5, n_temp_modes=4, foi=None,
#                woi=None, n_folds=1, ideal_pc_test=0, viz=True, mat_eng=None, return_mu_matrix=False):
#     """
#     Run the Multiple Sparse Priors (MSP) source reconstruction algorithm.
#
#     This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data. It involves mesh
#     smoothing and running the MSP algorithm in MATLAB. The MEG/EEG data must already be coregistered with the given
#     mesh.
#
#     Parameters:
#     mesh_fname (str): Filename of the mesh data.
#     data_fname (str): Filename of the MEG/EEG data.
#     n_layers (int): Number of layers in the mesh.
#     priors (list, optional): Indices of vertices to be used as priors. Default is an empty list.
#     patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
#     n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 4.
#     foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
#     woi (list, optional): Window of interest as [start, end]. Default is [-np.inf, np.inf].
#     n_folds (int): Number of cross validation folds. Must be >1 for cross validation error
#     ideal_pc_test (float): Percentage of channels to leave out (ideal because need an integer number of channels)
#     viz (boolean, optional): Whether or not to show SPM visualization. Default is True
#     mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
#     return_mu_matrix (boolean, optional): Whether or not to return the matrix needed to reconstruct source activity.
#                                           Default is False
#
#     Returns:
#     list: A list containing the free energy, cross validation error (cv_err), and the matrix needed to reconstruct
#           source activity (mu_matrix; if return_mu_matrix is True).
#
#     Notes:
#     - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
#     - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
#     - The function will automatically close the MATLAB engine if it was started within the function.
#     - Priors are adjusted by adding 1 to each index to align with MATLAB's 1-based indexing.
#     """
#     if foi is None:
#         foi = [0, 256]
#     if priors is None:
#         priors = []
#     if woi is None:
#         woi = [-np.inf, np.inf]
#
#     spm_path = get_spm_path()
#
#     print(f'Smoothing {mesh_fname}')
#     _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)
#
#     priors = [x + 1 for x in priors]
#     if isinstance(woi, np.ndarray):
#         woi = woi.tolist()
#
#     with matlab_context(mat_eng) as eng:
#         if return_mu_matrix:
#             free_energy, cv_err, mu_matrix = eng.invert_msp(
#                 data_fname,
#                 matlab.int32(priors),
#                 float(patch_size),
#                 float(n_temp_modes),
#                 matlab.double(foi),
#                 matlab.double(woi),
#                 float(n_folds),
#                 float(ideal_pc_test),
#                 int(viz),
#                 spm_path,
#                 nargout=3
#             )
#             ret_vals = [free_energy, np.array(cv_err), np.array(mu_matrix)]
#         else:
#             free_energy, cv_err = eng.invert_msp(
#                 data_fname,
#                 matlab.int32(priors),
#                 float(patch_size),
#                 float(n_temp_modes),
#                 matlab.double(foi),
#                 matlab.double(woi),
#                 float(n_folds),
#                 float(ideal_pc_test),
#                 int(viz),
#                 spm_path,
#                 nargout=2
#             )
#             ret_vals = [free_energy, np.array(cv_err)]
#
#     return ret_vals
#
#
# def invert_sliding_window(prior, mesh_fname, data_fname, n_layers, patch_size=5, n_temp_modes=1, win_size=16,
#                           win_overlap=True, foi=None, hann=True, viz=True, mat_eng=None):
#     """
#     Run the Multiple Sparse Priors (MSP) source reconstruction algorithm in a sliding time window.
#
#     This function interfaces with MATLAB to perform MSP source reconstruction on MEG/EEG data within sliding time
#     windows. It involves mesh smoothing and running the MSP algorithm in MATLAB for each time window. The MEG/EEG data
#     must already be coregistered with the given mesh.
#
#     Parameters:
#     prior (float): Index of the vertex to be used as a prior.
#     mesh_fname (str): Filename of the mesh data.
#     data_fname (str): Filename of the MEG/EEG data.
#     n_layers (int): Number of layers in the mesh.
#     patch_size (int, optional): Patch size for mesh smoothing. Default is 5.
#     n_temp_modes (int, optional): Number of temporal modes for the beamformer. Default is 1.
#     win_size (float, optional): Size of the sliding window in ms. Default is 16. If you increase win_size, you may
#                                 have to increase n_temp_modes.
#     win_overlap (bool, optional): Whether the windows should overlap. Default is True.
#     foi (list, optional): Frequency of interest range as [low, high]. Default is [0, 256].
#     hann (bool, optional): Whether or not to use Hann windowing. Default is True
#     viz (boolean, optional): Whether or not to show SPM visualization. Default is True
#     mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
#
#     Returns:
#     list: A list containing the free energy time series (free_energy), and the windows of interest (wois).
#
#     Notes:
#     - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
#     - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
#     - The function will automatically close the MATLAB engine if it was started within the function.
#     - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
#     """
#     if foi is None:
#         foi = [0, 256]
#
#     spm_path = get_spm_path()
#
#     print(f'Smoothing {mesh_fname}')
#     _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)
#
#     prior = prior + 1.0
#
#     with matlab_context(mat_eng) as eng:
#         free_energy, wois = eng.invert_sliding_window(
#             float(prior),
#             data_fname,
#             float(patch_size),
#             float(n_temp_modes),
#             float(win_size),
#             win_overlap,
#             matlab.double(foi),
#             int(hann),
#             int(viz),
#             spm_path,
#             nargout=2
#         )
#
#     return [np.array(free_energy), np.array(wois)]


def load_source_time_series(data_fname, mu_matrix=None, inv_fname=None, vertices=None):
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

    Returns:
    ndarray: An array containing the extracted source time series data (sources x time x trial).
    ndarray: An array containing the timestamps

    Notes:
    - If 'inv_fname' is not provided, and 'mu_matrix' is None, the inverse solution from the MEG/EEG data file specified
      by 'data_fname' will be used.
    - If 'mu_matrix' is provided, the function will compute the source time series directly using the lead field matrix,
      without the need for precomputed inverse solutions.
    """

    sensor_data, time, ch_names = load_meg_sensor_data(data_fname)

    if mu_matrix is None:
        if inv_fname is None:
            inv_fname = data_fname

        with h5py.File(inv_fname, 'r') as file:
            if not 'inv' in file['D']['other']:
                print('Error: source inversion has not been run on this dataset')
                return None, None, None

            m_data = file[file['D']['other']['inv'][0][0]]['inverse']['M']['data'][()]
            m_ir = file[file['D']['other']['inv'][0][0]]['inverse']['M']['ir'][()]
            m_jc = file[file['D']['other']['inv'][0][0]]['inverse']['M']['jc'][()]
            U = file[file[file['D']['other']['inv'][0][0]]['inverse']['U'][0][0]][()]
        # Reconstruct the sparse matrix
        num_rows = int(max(m_ir)) + 1  # Assuming 0-based indexing in Python
        num_cols = len(m_jc) - 1  # The number of columns is one less than the length of jc
        M = csc_matrix((m_data, m_ir, m_jc), shape=(num_rows, num_cols))
        if vertices is not None:
            M = M[vertices, :]
        mu_matrix = (M @ U)

    else:
        if vertices is not None:
            mu_matrix = mu_matrix[vertices, :]

    if len(sensor_data.shape) > 2:
        n_trials = sensor_data.shape[2]
        source_ts = np.zeros((mu_matrix.shape[0], len(time), n_trials))
        for trial in range(n_trials):
            source_ts[:, :, trial] = mu_matrix @ sensor_data[:, :, trial]
    else:
        source_ts = mu_matrix @ sensor_data

    return source_ts, time, mu_matrix