"""
This module provides tools for the coregistration, and source reconstruction of MEG data
using SPM (Statistical Parametric Mapping).
Key operations include:

- Coregistration of MRI and surface meshes with MEG data.
- Empirical Bayesian Beamformer (EBB) and Multiple Sparse Priors (MSP) source reconstruction
  algorithms.
- Sliding time window source reconstruction using MSP
- Utility function for loading source data after source reconstruction.
"""

import os
import h5py
import numpy as np
from scipy.io import savemat
from scipy.sparse import csc_matrix

from lameg.util import load_meg_sensor_data, spm_context, batch
import matlab # pylint: disable=wrong-import-order,import-error


def coregister(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, fid_labels=('nas', 'lpa', 'rpa'),
               viz=True, spm_instance=None) -> None:
    """
    Run head coregistration.

    This function performs head coregistration on MEG data using an MRI and mesh, and computes
    a forward model using the Nolte single shell model.

    Parameters
    ----------
    nas : list
        NASion fiducial coordinates.
    lpa : list
        Left PreAuricular fiducial coordinates.
    rpa : list
        Right PreAuricular fiducial coordinates.
    mri_fname : str
        Filename of the MRI data.
    mesh_fname : str
        Filename of the mesh data.
    data_fname : str
        Filename of the MEG data.
    fid_labels : list, optional
        Fiducial coordinate labels. Default is ['nas', 'lpa', 'rpa'].
    viz : bool, optional
        Whether or not to show SPM visualization. Default is True.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.

    Notes
    -----
    If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    The function will automatically close the standalone SPM instance if it was started
    within the function.
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
    fiducials['fidname'] = list(fid_labels)
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
    batch(cfg, viz=viz, spm_instance=spm_instance)


def invert_ebb(mesh_fname, data_fname, n_layers, patch_size=5, n_temp_modes=4, foi=None, woi=None,
               hann_windowing=False, n_folds=1, ideal_pc_test=0, viz=True, return_mu_matrix=False,
               spm_instance=None):
    """
    Run the Empirical Bayesian Beamformer (EBB) source reconstruction algorithm.

    This function performs EBB source reconstruction on MEG data. It involves mesh smoothing
    and running the EBB algorithm. The MEG data must already be coregistered with the given mesh.

    Parameters
    ----------
    mesh_fname : str
        Filename of the mesh data.
    data_fname : str
        Filename of the MEG data.
    n_layers : int
        Number of layers in the mesh.
    patch_size : int, optional
        Patch size for mesh smoothing. Default is 5.
    n_temp_modes : int, optional
        Number of temporal modes for the beamformer. Default is 4.
    foi : list, optional
        Frequency of interest range as [low, high]. Default is [0, 256].
    woi : list, optional
        Window of interest as [start, end]. Default is [-np.inf, np.inf].
    hann_windowing : int, optional
        Whether or not to perform Hann windowing. Default is False.
    n_folds : int
        Number of cross-validation folds. Must be >1 for cross-validation error.
    ideal_pc_test : float
        Percentage of channels to leave out (ideal because it needs an integer number of channels).
    viz : bool, optional
        Whether or not to show SPM visualization. Default is True.
    return_mu_matrix : bool, optional
        Whether or not to return the matrix needed to reconstruct source activity. Default is False.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.

    Returns
    -------
    list
        A list containing the free energy, cross-validation error (cv_err), and the matrix needed
        to reconstruct source activity (mu_matrix; if return_mu_matrix is True).

    Notes
    -----
    - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    """
    if woi is None:
        woi = [-np.inf, np.inf]
    if foi is None:
        foi = [0, 256]

    # Extract directory name and file name without extension
    data_dir, fname_with_ext = os.path.split(data_fname)
    fname, _ = os.path.splitext(fname_with_ext)

    with spm_context(spm_instance) as spm:
        print(f'Smoothing {mesh_fname}')
        _ = spm.spm_eeg_smoothmesh_multilayer_mm(
            mesh_fname,
            float(patch_size),
            float(n_layers),
            nargout=1
        )

        # Construct new file name with added '_testmodes.mat'
        spatialmodesname = os.path.join(data_dir, f'{fname}_testmodes.mat')
        spatialmodename, nmodes, pctest = spm.spm_eeg_inv_prep_modes_xval(
            data_fname,
            matlab.double([]),
            spatialmodesname,
            float(n_folds),
            float(ideal_pc_test),
            nargout=3
        )

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
                                "hanning": float(hann_windowing),
                                "isfixedpatch": {
                                    "randpatch": {
                                        "npatches": float(512),
                                        "niter": float(1)
                                    }
                                },
                                "patchfwhm": -float(patch_size),
                                "mselect": float(0),
                                "nsmodes": float(nmodes),
                                "umodes": np.asarray([spatialmodename], dtype="object"),
                                "ntmodes": float(n_temp_modes),
                                "priors": {
                                    "priorsmask": np.asarray([''], dtype="object"),
                                    "space": 0
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
    batch(cfg, viz=viz, spm_instance=spm_instance)

    with h5py.File(data_fname, 'r') as file:
        inverse_struct = file[file['D']['other']['inv'][0][0]]['inverse']

        free_energy = np.squeeze(inverse_struct['crossF'][()])
        cv_err = np.squeeze(inverse_struct['crosserr'][()])

        m_data = inverse_struct['M']['data'][()]
        m_ir = inverse_struct['M']['ir'][()]
        m_jc = inverse_struct['M']['jc'][()]
        data_reduction_mat = file[inverse_struct['U'][0][0]][()]

    if not return_mu_matrix:
        return [free_energy, cv_err]

    # Reconstruct the sparse matrix
    num_rows = int(max(m_ir)) + 1  # Assuming 0-based indexing in Python
    num_cols = len(m_jc) - 1  # The number of columns is one less than the length of jc
    weighting_mat = csc_matrix((m_data, m_ir, m_jc), shape=(num_rows, num_cols))
    mu_matrix = weighting_mat @ data_reduction_mat
    return [free_energy, cv_err, mu_matrix]


def invert_msp(mesh_fname, data_fname, n_layers, priors=None, patch_size=5, n_temp_modes=4,
               foi=None, woi=None, hann_windowing=False, n_folds=1, ideal_pc_test=0, viz=True,
               return_mu_matrix=False, spm_instance=None):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm.

    This function performs MSP source reconstruction on MEG data. It involves mesh smoothing and
    running the MSP algorithm. The MEG data must already be coregistered with the given mesh.

    Parameters
    ----------
    mesh_fname : str
        Filename of the mesh data.
    data_fname : str
        Filename of the MEG data.
    n_layers : int
        Number of layers in the mesh.
    priors : list, optional
        Indices of vertices to be used as priors. Default is an empty list.
    patch_size : int, optional
        Patch size for mesh smoothing. Default is 5.
    n_temp_modes : int, optional
        Number of temporal modes for the beamformer. Default is 4.
    foi : list, optional
        Frequency of interest range as [low, high]. Default is [0, 256].
    woi : list, optional
        Window of interest as [start, end]. Default is [-np.inf, np.inf].
    hann_windowing : int, optional
        Whether or not to perform Hann windowing. Default is False.
    n_folds : int
        Number of cross-validation folds. Must be >1 for cross-validation error.
    ideal_pc_test : float
        Percentage of channels to leave out (ideal because it needs an integer number of channels).
    viz : bool, optional
        Whether or not to show SPM visualization. Default is True.
    return_mu_matrix : bool, optional
        Whether or not to return the matrix needed to reconstruct source activity. Default is False.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.

    Returns
    -------
    list
        A list containing the free energy, cross-validation error (cv_err), and the matrix
        needed to reconstruct source activity (mu_matrix; if return_mu_matrix is True).

    Notes
    -----
    - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    - Priors are adjusted by adding 1 to each index to align with MATLAB's 1-based indexing.
    """
    if foi is None:
        foi = [0, 256]
    if priors is None:
        priors = []
    if woi is None:
        woi = [-np.inf, np.inf]

    priors = [x + 1 for x in priors]
    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    # Extract directory name and file name without extension
    data_dir, fname_with_ext = os.path.split(data_fname)
    fname, _ = os.path.splitext(fname_with_ext)

    with spm_context(spm_instance) as spm:
        print(f'Smoothing {mesh_fname}')
        _ = spm.spm_eeg_smoothmesh_multilayer_mm(
            mesh_fname,
            float(patch_size),
            float(n_layers),
            nargout=1
        )

        # Construct new file name with added '_testmodes.mat'
        spatialmodesname = os.path.join(data_dir, f'{fname}_testmodes.mat')
        spatialmodename, nmodes, pctest = spm.spm_eeg_inv_prep_modes_xval(
            data_fname,
            matlab.double([]),
            spatialmodesname,
            float(n_folds),
            float(ideal_pc_test),
            nargout=3
        )

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
                                "invtype": 'MSP',
                                "woi": np.array(woi, dtype=float),
                                "foi": np.array(foi, dtype=float),
                                "hanning": float(hann_windowing),
                                "patchfwhm": -float(patch_size),
                                "mselect": float(0),
                                "nsmodes": float(nmodes),
                                "umodes": np.asarray([spatialmodename], dtype="object"),
                                "ntmodes": float(n_temp_modes),
                                "priors": {
                                    "priorsmask": np.asarray([''], dtype="object"),
                                    "space": 0
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
    if len(priors) > 0:
        patchfilename = os.path.join(data_dir, 'patch.mat')
        savemat(patchfilename, {'Ip': priors})

        cfg['spm']['meeg']['source']['invertiter']['isstandard']['custom']['isfixedpatch'] = {
            "fixedpatch": {
                "fixedfile": np.asarray([patchfilename], dtype="object"),
                "fixedrows": np.array([1, np.inf], dtype=float)
            }
        }
    else:
        cfg['spm']['meeg']['source']['invertiter']['isstandard']['custom']['isfixedpatch'] = {
            "randpatch": {
                "npatches": float(512),
                "niter": float(1)
            }
        }

    batch(cfg, viz=viz, spm_instance=spm_instance)

    with h5py.File(data_fname, 'r') as file:
        inverse_struct = file[file['D']['other']['inv'][0][0]]['inverse']

        free_energy = np.squeeze(inverse_struct['crossF'][()])
        cv_err = np.squeeze(inverse_struct['crosserr'][()])

        m_data = inverse_struct['M']['data'][()]
        m_ir = inverse_struct['M']['ir'][()]
        m_jc = inverse_struct['M']['jc'][()]
        data_reduction_mat = file[inverse_struct['U'][0][0]][()]

    if not return_mu_matrix:
        return [free_energy, cv_err]

    # Reconstruct the sparse matrix
    num_rows = int(max(m_ir)) + 1  # Assuming 0-based indexing in Python
    num_cols = len(m_jc) - 1  # The number of columns is one less than the length of jc
    weighting_mat = csc_matrix((m_data, m_ir, m_jc), shape=(num_rows, num_cols))
    mu_matrix = weighting_mat @ data_reduction_mat
    return [free_energy, cv_err, mu_matrix]


def invert_sliding_window(prior, mesh_fname, data_fname, n_layers, patch_size=5, n_temp_modes=1,
                          win_size=50, win_overlap=True, foi=None, hann_windowing=True, viz=True,
                          spm_instance=None):
    """
    Run the Multiple Sparse Priors (MSP) source reconstruction algorithm in a sliding time window.

    This function performs MSP source reconstruction on MEG data within sliding time windows. It
    involves mesh smoothing and running the MSP algorithm. The MEG data must already be
    coregistered with the given mesh.

    Parameters
    ----------
    prior : float
        Index of the vertex to be used as a prior.
    mesh_fname : str
        Filename of the mesh data.
    data_fname : str
        Filename of the MEG data.
    n_layers : int
        Number of layers in the mesh.
    patch_size : int, optional
        Patch size for mesh smoothing. Default is 5.
    n_temp_modes : int, optional
        Number of temporal modes for the beamformer. Default is 1.
    win_size : float, optional
        Size of the sliding window in ms. Default is 50. If you increase `win_size`, you may need
        to increase `n_temp_modes`.
    win_overlap : bool, optional
        Whether the windows should overlap. Default is True.
    foi : list, optional
        Frequency of interest range as [low, high]. Default is [0, 256].
    hann_windowing : bool, optional
        Whether or not to use Hann windowing. Default is True.
    viz : bool, optional
        Whether or not to show SPM visualization. Default is True.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.

    Returns
    -------
    list
        A list containing the free energy time series (free_energy), and the windows of interest
        (wois).

    Notes
    -----
    - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
    """

    if foi is None:
        foi = [0, 256]

    prior = prior + 1.0

    _, time, _ = load_meg_sensor_data(data_fname)

    time = time * 1000  # Convert time to milliseconds
    time_step = time[1] - time[0]  # Compute the difference in time between steps
    win_steps = int(round(win_size / time_step))  # Calculate the number of steps in each window

    wois = []
    if win_overlap:
        for t_idx in range(len(time)):
            win_l = max(0, int(np.ceil(t_idx - win_steps / 2)))
            win_r = min(len(time) - 1, int(np.floor(t_idx + win_steps / 2)))
            woi = [time[win_l], time[win_r]]
            wois.append(woi)
    else:
        time_steps = np.linspace(time[0], time[-1], int((time[-1] - time[0]) / win_size + 1))
        for i in range(1, len(time_steps)):
            wois.append([time_steps[i - 1], time_steps[i]])
    wois = np.array(wois, dtype=float)

    # Extract directory name and file name without extension
    data_dir, fname_with_ext = os.path.split(data_fname)
    fname, _ = os.path.splitext(fname_with_ext)

    with spm_context(spm_instance) as spm:
        print(f'Smoothing {mesh_fname}')
        _ = spm.spm_eeg_smoothmesh_multilayer_mm(
            mesh_fname,
            float(patch_size),
            float(n_layers),
            nargout=1
        )

        # Construct new file name with added '_testmodes.mat'
        spatialmodesname = os.path.join(data_dir, f'{fname}_testmodes.mat')
        spatialmodename, nmodes, _ = spm.spm_eeg_inv_prep_modes_xval(
            data_fname,
            [],
            spatialmodesname,
            1,
            0,
            nargout=3
        )

    patchfilename = os.path.join(data_dir, 'patch.mat')
    savemat(patchfilename, {'Ip': np.array([prior], dtype=float)})

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
                                "invtype": 'MSP',
                                "woi": wois,
                                "foi": np.array(foi, dtype=float),
                                "hanning": float(hann_windowing),
                                "isfixedpatch": {
                                    "fixedpatch": {
                                        "fixedfile": np.asarray([patchfilename], dtype=object),
                                        "fixedrows": np.array([1, np.inf], dtype=float)
                                    }
                                },
                                "patchfwhm": -float(patch_size),
                                "mselect": float(0),
                                "nsmodes": float(nmodes),
                                "umodes": np.asarray([spatialmodename], dtype="object"),
                                "ntmodes": float(n_temp_modes),
                                "priors": {
                                    "priorsmask": np.asarray([''], dtype="object"),
                                    "space": 0
                                },
                                "outinv": '',
                            }
                        },
                        "modality": np.asarray(['All'], dtype="object"),
                        "crossval": np.asarray([0, 1], dtype=float)
                    }
                }
            }
        }
    }

    batch(cfg, viz=viz, spm_instance=spm_instance)

    with h5py.File(data_fname, 'r') as file:
        free_energy = np.squeeze(file[file['D']['other']['inv'][0][0]]['inverse']['crossF'][()])

    return [free_energy, wois]


def load_source_time_series(data_fname, mu_matrix=None, inv_fname=None, vertices=None):
    """
    Load source time series data from specified vertices using precomputed inverse solutions or a
    lead field matrix.

    This function extracts time series data from specific vertices, based on precomputed inverse
    solutions, or computes the source time series using a provided lead field matrix.

    Parameters
    ----------
    data_fname : str
        Filename or path of the MEG data file.
    mu_matrix : ndarray, optional
        Lead field matrix (source x sensor). Default is None.
    inv_fname : str, optional
        Filename or path of the file containing the inverse solutions. Default is None.
    vertices : list of int, optional
        List of vertex indices from which to extract time series data. Default is None, which
        implies all vertices will be used.

    Returns
    -------
    source_ts : np.array
        An array containing the extracted source time series data (sources x time x trial).
    time : np.array
        An array containing the timestamps.
    mu_matrix : np.array
        The matrix needed to reconstruct source activity from sensor signals.

    Notes
    -----
    - If `inv_fname` is not provided, and `mu_matrix` is None, the inverse solution from the
      MEG data file specified by `data_fname` will be used.
    - If `mu_matrix` is provided, the function will compute the source time series directly using
      the lead field matrix, without the need for precomputed inverse solutions.
    """

    sensor_data, time, _ = load_meg_sensor_data(data_fname)

    if mu_matrix is None:
        if inv_fname is None:
            inv_fname = data_fname

        with h5py.File(inv_fname, 'r') as file:
            if 'inv' not in file['D']['other']:
                print('Error: source inversion has not been run on this dataset')
                return None, None, None

            inverse_struct=file[file['D']['other']['inv'][0][0]]['inverse']
            m_data = inverse_struct['M']['data'][()]
            m_ir = inverse_struct['M']['ir'][()]
            m_jc = inverse_struct['M']['jc'][()]
            data_reduction_mat = file[inverse_struct['U'][0][0]][()]

        # Reconstruct the sparse matrix
        num_rows = int(max(m_ir)) + 1  # Assuming 0-based indexing in Python
        num_cols = len(m_jc) - 1  # The number of columns is one less than the length of jc
        weighting_mat = csc_matrix((m_data, m_ir, m_jc), shape=(num_rows, num_cols))
        if vertices is not None:
            weighting_mat = weighting_mat[vertices, :]
        mu_matrix = weighting_mat @ data_reduction_mat

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
