"""
Laminar source inversion and MEG-MRI coregistration tools using SPM.

This module provides a high-level Python interface for performing source reconstruction
and forward modeling of MEG data via the SPM standalone engine. It supports both
Empirical Bayesian Beamformer (EBB) and Multiple Sparse Priors (MSP) algorithms,
as well as utilities for working with inversion outputs.

Main functionalities
--------------------
- **Coregistration**: Aligns MEG sensor data with anatomical MRI and surface meshes,
  and constructs a forward model using the Nolte single-shell approximation.
- **Empirical Bayesian Beamformer (EBB)**: Performs source inversion using empirical Bayesian
  beamforming with cross-validation.
- **Multiple Sparse Priors (MSP)**: Performs sparse Bayesian inversion with optional
  vertex-level priors to constrain source localization.
- **Sliding-window inversion**: Applies MSP within time windows to obtain time-resolved
  estimates of free energy (model evidence).
- **Reconstruction utilities**:
  - Load source-space time series from precomputed inversions or lead field matrices.
  - Verify the existence of inversion structures in SPM data files.
  - Extract cortical mesh vertex coordinates from forward models.

Notes
-----
- All inversion and coregistration routines depend on a valid SPM standalone
  installation accessible via the `spm_context` interface.
- Meshes must be provided through a `LayerSurfaceSet` object, typically containing
  pial, white, and intermediate laminar surfaces.
- This module supports both MATLAB v7.3+ (HDF5) and pre-v7.3 file formats for SPM data.
"""

# pylint: disable=C0302
import os
import h5py
import numpy as np
from scipy.io import savemat, loadmat
from scipy.sparse import csc_matrix, issparse

from lameg.util import load_meg_sensor_data, spm_context, batch
import matlab # pylint: disable=wrong-import-order,import-error


def coregister(fid_coords, data_fname, surf_set, layer_name=None, stage='ds',
               orientation='link_vector', fixed=True, fid_labels=('nas', 'lpa', 'rpa'),
               iskull_fname=None, oskull_fname=None, scalp_fname=None,
               forward_model_type='Single Shell', inversion_idx=0, viz=True, spm_instance=None):
    """
    Perform MEG-MRI coregistration and compute the forward model using the Nolte single-shell
    method.

    This function uses SPM's standalone interface to coregister MEG sensor data to the subject's
    anatomy. It constructs a forward model based on a specified laminar surface mesh (from a
    LayerSurfaceSet) and fiducial-based alignment between headshape and MRI coordinates. The
    laminar surface is selected by layer name, processing stage, and orientation parameters. The
    forward model type can be chosen from several SPM-compatible MEG models.

    Parameters
    ----------
    fid_coords : dict
        Dictionary of fiducial landmark coordinates, e.g.:
        ``{'nas': [x, y, z], 'lpa': [x, y, z], 'rpa': [x, y, z]}``
        Values must be expressed in MEG headspace coordinates (millimeters).
    data_fname : str
        Path to the MEG dataset (SPM-compatible format).
    surf_set : LayerSurfaceSet
        The subject's surface set containing laminar meshes.
    layer_name : str or None, optional
        Layer surface to use for forward modeling (e.g., 'pial', 'white', or fractional layer).
        If None, the multilayer combined surface is used.
    stage : str, optional
        Processing stage of the surface mesh (default: 'ds').
    orientation : str, optional
        Orientation model used for dipole alignment (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    fid_labels : sequence of str, optional
        Labels for fiducial points, typically ('nas', 'lpa', 'rpa').
    iskull_fname : str or None, optional
        Path to the inner-skull surface. If None, an empty string is passed to SPM.
    oskull_fname : str or None, optional
        Path to the outer-skull surface. If None, an empty string is passed to SPM.
    scalp_fname : str or None, optional
        Path to the scalp/outer-skin surface. If None, an empty string is passed to SPM.
    forward_model_type : {'Single Sphere', 'MEG Local Spheres', 'Single Shell',
    'MEG OpenMEEG BEM'}, optional
        MEG forward model to use. Defaults to 'Single Shell'.
    inversion_idx: int, optional
        Index of the forward model to create within the SPM data object (default: 0).
    viz : bool, optional
        Whether to display SPM's coregistration and forward model visualization (default: True).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Notes
    -----
    - The function constructs and executes an SPM batch job using the provided fiducials and
      mesh information.
    - The forward model uses the **Nolte single-shell** MEG head model.
    - The mesh is derived from the `LayerSurfaceSet` and passed to SPM as a custom cortical
      surface.
    """
    mesh_fname = surf_set.get_mesh_path(layer_name=layer_name, stage=stage,
                                        orientation=orientation, fixed=fixed)

    if iskull_fname is None:
        iskull_fname = ''
    if oskull_fname is None:
        oskull_fname = ''
    if scalp_fname is None:
        scalp_fname = ''

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
    fiducials['specification']['type'][0] = np.array(fid_coords['nas'],
                                                     dtype=float)
    fiducials['specification']['type'][1] = np.array(fid_coords['lpa'],
                                                     dtype=float)
    fiducials['specification']['type'][2] = np.array(fid_coords['rpa'],
                                                     dtype=float)

    cfg = {
        "spm": {
            "meeg": {
                "source": {
                    "headmodel": {
                        "D": np.asarray([data_fname],
                                        dtype="object"),
                        "val": (float(inversion_idx) + 1),
                        "comment": "",
                        "meshing": {
                            "meshes": {
                                "custom": {
                                    "mri": np.asarray(
                                        [f'{surf_set.mri_file},1'],
                                        dtype="object"
                                    ),
                                    "cortex": np.asarray(
                                        [mesh_fname],
                                        dtype="object"
                                    ),
                                    "iskull": np.asarray(
                                        [iskull_fname],
                                        dtype="object"
                                    ),
                                    "oskull": np.asarray(
                                        [oskull_fname],
                                        dtype="object"
                                    ),
                                    "scalp": np.asarray(
                                        [scalp_fname],
                                        dtype="object"
                                    )
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
                            "meg": forward_model_type
                        }
                    }
                }
            }
        }

    }
    batch(cfg, viz=viz, spm_instance=spm_instance)


def invert_ebb(data_fname, surf_set, layer_name=None, stage='ds',
               orientation='link_vector', fixed=True, patch_size=5, n_temp_modes=4,
               n_spatial_modes=None, foi=None, woi=None, hann_windowing=False, n_folds=1,
               ideal_pc_test=0, inversion_idx=0, viz=True, return_mu_matrix=False,
               spm_instance=None):
    """
    Perform Empirical Bayesian Beamformer (EBB) source inversion on MEG data.

    This function runs the SPM EBB source reconstruction pipeline on MEG data that has already been
    coregistered with an anatomical mesh. It first smooths the specified cortical or laminar mesh,
    then prepares spatial and temporal modes, and finally computes source inversion using SPM's
    EBB algorithm. Optionally, it returns the full projection matrix used to reconstruct source
    activity.

    Parameters
    ----------
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    surf_set : LayerSurfaceSet
        The subject's surface set containing laminar meshes.
    layer_name : str or None, optional
        Surface layer to use for inversion (e.g., 'pial', 'white', or a fractional layer).
        If None, the full multilayer surface is used.
    stage : str, optional
        Processing stage of the surface mesh (default: 'ds').
    orientation : str, optional
        Orientation model used for dipoles (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    patch_size : float, optional
        Full-width at half-maximum (FWHM) of cortical patch smoothing in millimeters (default: 5).
    n_temp_modes : int, optional
        Number of temporal modes for dimensionality reduction (default: 4).
    n_spatial_modes : int or None, optional
        Number of spatial modes for data reduction. If None, all channels are used.
    foi : list of float, optional
        Frequency range of interest [low, high] in Hz (default: [0, 256]).
    woi : list of float, optional
        Time window of interest [start, end] in ms (default: full epoch).
    hann_windowing : bool, optional
        Whether to apply Hann windowing to data before inversion (default: False).
    n_folds : int, optional
        Number of cross-validation folds for spatial mode testing (default: 1).
    ideal_pc_test : float, optional
        Fraction of channels to leave out during cross-validation (default: 0).
    inversion_idx: int, optional
        Index of the inversion to create within the SPM data object (default: 0).
    viz : bool, optional
        Whether to display SPM's inversion progress and diagnostic plots (default: True).
    return_mu_matrix : bool, optional
        If True, return the full source reconstruction matrix (default: False).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Returns
    -------
    results : list
        A list containing:
        - free_energy : float
            Model evidence (free energy) from the inversion.
        - cv_err : float
            Cross-validation error across data folds.
        - mu_matrix : scipy.sparse.csc_matrix, optional
            Source reconstruction matrix (returned only if `return_mu_matrix=True`).

    Notes
    -----
    - The forward model must be precomputed via `coregister()` before calling this function.
    - Mesh smoothing uses `spm_eeg_smoothmesh_multilayer_mm`.
    - Spatial mode preparation and cross-validation use `spm_eeg_inv_prep_modes_xval`.
    """
    mesh_fname = surf_set.get_mesh_path(layer_name=layer_name, stage=stage,
                                        orientation=orientation, fixed=fixed)
    n_layers = 1
    if layer_name is None:
        n_layers=surf_set.n_layers

    if foi is None:
        foi = [0, 256]

    if woi is None:
        woi = [-np.inf, np.inf]
    elif isinstance(woi, np.ndarray):
        woi = woi.tolist()

    if n_spatial_modes is None:
        n_spatial_modes = matlab.double([])
    else:
        n_spatial_modes = float(n_spatial_modes)

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
            n_spatial_modes,
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
                        "val": (float(inversion_idx) + 1),
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
        inverse_struct = file[file['D']['other']['inv'][inversion_idx][0]]['inverse']

        free_energy = np.squeeze(inverse_struct['crossF'][()])
        cv_err = np.squeeze(inverse_struct['crosserr'][()])

        m_data = inverse_struct['M']['data'][()]
        m_ir = inverse_struct['M']['ir'][()]
        m_jc = inverse_struct['M']['jc'][()]
        try:
            data_reduction_mat = file[inverse_struct['U'][0][0]][()]
        except AttributeError:
            dr_data = file[inverse_struct['U'][0][0]]['data'][()]
            dr_ir = file[inverse_struct['U'][0][0]]['ir'][()]
            dr_jc = file[inverse_struct['U'][0][0]]['jc'][()]
            num_rows = int(max(dr_ir)) + 1
            num_cols = len(dr_jc) - 1
            data_reduction_mat = csc_matrix(
                (dr_data, dr_ir, dr_jc),
                shape=(num_rows, num_cols)
            )

    if not return_mu_matrix:
        return [free_energy, cv_err]

    # Reconstruct the sparse matrix
    num_rows = int(max(m_ir)) + 1  # Assuming 0-based indexing in Python
    num_cols = len(m_jc) - 1  # The number of columns is one less than the length of jc
    weighting_mat = csc_matrix((m_data, m_ir, m_jc), shape=(num_rows, num_cols))
    mu_matrix = weighting_mat @ data_reduction_mat
    return [free_energy, cv_err, mu_matrix]


# pylint: disable=R0915
def invert_msp(data_fname, surf_set, layer_name=None, stage='ds',
               orientation='link_vector', fixed=True, priors=None, patch_size=5, n_temp_modes=4,
               n_spatial_modes=None, foi=None, woi=None, hann_windowing=False, n_folds=1,
               ideal_pc_test=0, inversion_idx=0, viz=True, return_mu_matrix=False,
               spm_instance=None):
    """
    Perform Multiple Sparse Priors (MSP) source inversion on MEG data.

    This function runs SPM's MSP source reconstruction pipeline using a specified cortical or
    laminar surface mesh. The procedure includes mesh smoothing, spatial mode preparation, and
    inversion using MSP. Optional vertex-level priors can be provided to constrain the inversion to
    specific cortical patches. The MEG dataset must already be coregistered with the provided
    surface mesh.

    Parameters
    ----------
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    surf_set : LayerSurfaceSet
        The subject's surface set containing laminar meshes.
    layer_name : str or None, optional
        Surface layer to use for inversion (e.g., 'pial', 'white', or a fractional layer).
        If None, the full multilayer surface is used.
    stage : str, optional
        Processing stage of the surface mesh (default: 'ds').
    orientation : str, optional
        Orientation model used for dipoles (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    priors : list of int, optional
        List of vertex indices to be used as MSP priors (0-based Python indexing).
        If None or empty, random patches are used as priors.
    patch_size : float, optional
        Full-width at half-maximum (FWHM) of cortical patch smoothing in millimeters (default: 5).
    n_temp_modes : int, optional
        Number of temporal modes for dimensionality reduction (default: 4).
    n_spatial_modes : int or None, optional
        Number of spatial modes for data reduction. If None, all channels are used.
    foi : list of float, optional
        Frequency range of interest [low, high] in Hz (default: [0, 256]).
    woi : list of float, optional
        Time window of interest [start, end] in ms (default: full epoch).
    hann_windowing : bool, optional
        Whether to apply Hann windowing to data before inversion (default: False).
    n_folds : int, optional
        Number of cross-validation folds for spatial mode testing (default: 1).
    ideal_pc_test : float, optional
        Fraction of channels to leave out during cross-validation (default: 0).
    inversion_idx: int, optional
        Index of the inversion to create within the SPM data object (default: 0).
    viz : bool, optional
        Whether to display SPM's inversion progress and diagnostic plots (default: True).
    return_mu_matrix : bool, optional
        If True, return the full source reconstruction matrix (default: False).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Returns
    -------
    results : list
        A list containing:
        - free_energy : float
            Model evidence (free energy) from the inversion.
        - cv_err : float
            Cross-validation error across data folds.
        - mu_matrix : scipy.sparse.csc_matrix, optional
            Source reconstruction matrix (returned only if `return_mu_matrix=True`).

    Notes
    -----
    - The forward model must be precomputed via `coregister()` before calling this function.
    - If `priors` are provided, they are converted to 1-based indices to match MATLAB conventions.
    - Mesh smoothing uses `spm_eeg_smoothmesh_multilayer_mm`.
    - Spatial mode preparation and cross-validation use `spm_eeg_inv_prep_modes_xval`.
    """
    mesh_fname = surf_set.get_mesh_path(layer_name=layer_name, stage=stage,
                                        orientation=orientation, fixed=fixed)
    n_layers = 1
    if layer_name is None:
        n_layers = surf_set.n_layers

    if foi is None:
        foi = [0, 256]

    if priors is None:
        priors = []

    if woi is None:
        woi = [-np.inf, np.inf]
    elif isinstance(woi, np.ndarray):
        woi = woi.tolist()

    if n_spatial_modes is None:
        n_spatial_modes = matlab.double([])
    else:
        n_spatial_modes = float(n_spatial_modes)

    priors = [x + 1 for x in priors]

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
            n_spatial_modes,
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
                        "val": (float(inversion_idx) + 1),
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
        inverse_struct = file[file['D']['other']['inv'][inversion_idx][0]]['inverse']

        free_energy = np.squeeze(inverse_struct['crossF'][()])
        cv_err = np.squeeze(inverse_struct['crosserr'][()])

        m_data = inverse_struct['M']['data'][()]
        m_ir = inverse_struct['M']['ir'][()]
        m_jc = inverse_struct['M']['jc'][()]
        try:
            data_reduction_mat = file[inverse_struct['U'][0][0]][()]
        except AttributeError:
            dr_data = file[inverse_struct['U'][0][0]]['data'][()]
            dr_ir = file[inverse_struct['U'][0][0]]['ir'][()]
            dr_jc = file[inverse_struct['U'][0][0]]['jc'][()]
            num_rows = int(max(dr_ir)) + 1
            num_cols = len(dr_jc) - 1
            data_reduction_mat = csc_matrix(
                (dr_data, dr_ir, dr_jc),
                shape=(num_rows, num_cols)
            )

    if not return_mu_matrix:
        return [free_energy, cv_err]

    # Reconstruct the sparse matrix
    num_rows = int(max(m_ir)) + 1  # Assuming 0-based indexing in Python
    num_cols = len(m_jc) - 1  # The number of columns is one less than the length of jc
    weighting_mat = csc_matrix((m_data, m_ir, m_jc), shape=(num_rows, num_cols))
    mu_matrix = weighting_mat @ data_reduction_mat
    return [free_energy, cv_err, mu_matrix]


def invert_sliding_window_msp(prior, data_fname, surf_set, layer_name=None, stage='ds',
                              orientation='link_vector', fixed=True, patch_size=5, n_temp_modes=1,
                              n_spatial_modes=None, wois=None, win_size=50, win_overlap=True, foi=None,
                              hann_windowing=True, inversion_idx=0, viz=True, spm_instance=None):
    """
    Perform Multiple Sparse Priors (MSP) source inversion over sliding time windows.

    This function applies SPM's MSP source reconstruction algorithm within successive overlapping
    or non-overlapping time windows, enabling time-resolved estimation of source model evidence
    (free energy). It smooths the surface mesh, prepares spatial modes, and repeatedly inverts the
    MEG data across windows using a fixed spatial prior centered on a specified vertex. The MEG
    dataset must already be coregistered with the surface mesh.

    Parameters
    ----------
    prior : int
        Index of the vertex to be used as the MSP prior (0-based Python indexing).
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    surf_set : LayerSurfaceSet
        The subject's surface set containing laminar meshes.
    layer_name : str or None, optional
        Surface layer to use for inversion (e.g., 'pial', 'white', or a fractional layer).
        If None, the full multilayer surface is used.
    stage : str, optional
        Processing stage of the surface mesh (default: 'ds').
    orientation : str, optional
        Orientation model used for dipoles (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    patch_size : float, optional
        Full-width at half-maximum (FWHM) of cortical patch smoothing in millimeters (default: 5).
    n_temp_modes : int, optional
        Number of temporal modes for dimensionality reduction (default: 1).
    n_spatial_modes : int or None, optional
        Number of spatial modes for data reduction. If None, all channels are used.
    wois : list of float, optional
        List of time windows of interest [start, end] pairs in ms (default: None). 
        If None, wois are generated on the full epoch, based on win_size and win_overlap
        (parameters ignored otherwise).
    win_size : float, optional
        Duration of each sliding window in milliseconds (default: 50).
    win_overlap : bool, optional
        Whether consecutive windows overlap (default: True).
    foi : list of float, optional
        Frequency range of interest [low, high] in Hz (default: [0, 256]).
    hann_windowing : bool, optional
        Whether to apply Hann windowing to each window before inversion (default: True).
    inversion_idx: int, optional
        Index of the inversion to create within the SPM data object (default: 0).
    viz : bool, optional
        Whether to display SPM's inversion progress and diagnostic plots (default: True).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Returns
    -------
    results : list
        A list containing:
        - free_energy : ndarray
            Array of model evidence (free energy) values across time windows.
        - wois : ndarray, shape (n_windows, 2)
            Time windows of interest in milliseconds.

    Notes
    -----
    - The forward model must be precomputed via `coregister()` before calling this function.
    - The `prior` index is internally converted to 1-based indexing for MATLAB compatibility.
    - Mesh smoothing uses `spm_eeg_smoothmesh_multilayer_mm`.
    - Spatial mode preparation uses `spm_eeg_inv_prep_modes_xval`.
    - Each windowed inversion uses the **Multiple Sparse Priors (MSP)** algorithm in SPM.
    """

    mesh_fname = surf_set.get_mesh_path(layer_name=layer_name, stage=stage,
                                        orientation=orientation, fixed=fixed)
    n_layers = 1
    if layer_name is None:
        n_layers = surf_set.n_layers

    if foi is None:
        foi = [0, 256]

    if n_spatial_modes is None:
        n_spatial_modes = matlab.double([])
    else:
        n_spatial_modes = float(n_spatial_modes)

    prior = prior + 1.0

    if wois is None:
        _, time, _ = load_meg_sensor_data(data_fname)

        time_step = time[1] - time[0]  # Compute the difference in time between steps
        sampling_rate = 1000.0 / time_step
        win_steps = int(round(win_size / time_step))  # Calculate the number of steps in each window

        if (win_steps / n_temp_modes) < 2:
            raise ValueError(
                f"win_size={win_size} ms yields only {win_steps} samples "
                f"({sampling_rate:.2f} Hz sampling). With n_temp_modes={n_temp_modes}, "
                f"the ratio win_samples / n_temp_modes = {win_steps / n_temp_modes:.2f} < 2. "
                "Increase win_size or reduce n_temp_modes."
            )

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
            n_spatial_modes,
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
                        "val": (float(inversion_idx) + 1),
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
        inv_struct = file[file['D']['other']['inv'][inversion_idx][0]]
        free_energy = np.squeeze(inv_struct['inverse']['crossF'][()])

    return [free_energy, wois]


def invert_sliding_window_ebb(data_fname, surf_set, layer_name=None, stage='ds',
                              orientation='link_vector', fixed=True, patch_size=5, n_temp_modes=1,
                              n_spatial_modes=None, wois=None, win_size=50, win_overlap=True,
                              foi=None, hann_windowing=True, inversion_idx=0,
                              viz=True, spm_instance=None):
    """
    Perform Empirical Bayesian Beamformer (EBB) source inversion over sliding time windows.

    This function applies SPM's EBB source reconstruction algorithm within successive overlapping
    or non-overlapping time windows, enabling time-resolved estimation of source model evidence
    (free energy). It smooths the surface mesh, prepares spatial modes, and repeatedly inverts the
    MEG data across windows. The MEG dataset must already be coregistered with the surface mesh.

    Parameters
    ----------
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    surf_set : LayerSurfaceSet
        The subject's surface set containing laminar meshes.
    layer_name : str or None, optional
        Surface layer to use for inversion (e.g., 'pial', 'white', or a fractional layer).
        If None, the full multilayer surface is used.
    stage : str, optional
        Processing stage of the surface mesh (default: 'ds').
    orientation : str, optional
        Orientation model used for dipoles (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    patch_size : float, optional
        Full-width at half-maximum (FWHM) of cortical patch smoothing in millimeters (default: 5).
    n_temp_modes : int, optional
        Number of temporal modes for dimensionality reduction (default: 1).
    n_spatial_modes : int or None, optional
        Number of spatial modes for data reduction. If None, all channels are used.
    wois : list of float, optional
        List of time windows of interest [start, end] pairs in ms (default: None).
        If None, wois are generated on the full epoch, based on win_size and win_overlap
        (parameters ignored otherwise).
    win_size : float, optional
        Duration of each sliding window in milliseconds (default: 50).
    win_overlap : bool, optional
        Whether consecutive windows overlap (default: True).
    foi : list of float, optional
        Frequency range of interest [low, high] in Hz (default: [0, 256]).
    hann_windowing : bool, optional
        Whether to apply Hann windowing to each window before inversion (default: True).
    inversion_idx: int, optional
        Index of the inversion to create within the SPM data object (default: 0).
    viz : bool, optional
        Whether to display SPM's inversion progress and diagnostic plots (default: True).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Returns
    -------
    results : list
        A list containing:
        - free_energy : ndarray
            Array of model evidence (free energy) values across time windows.
        - wois : ndarray, shape (n_windows, 2)
            Time windows of interest in milliseconds.
    Notes
    -----
    - The forward model must be precomputed via `coregister()` before calling this function.
    - Mesh smoothing uses `spm_eeg_smoothmesh_multilayer_mm`.
    - Spatial mode preparation uses `spm_eeg_inv_prep_modes_xval`.
    - Each windowed inversion uses the **Empirical Bayesian Beamformer (EBB)** algorithm in SPM.
    """

    mesh_fname = surf_set.get_mesh_path(layer_name=layer_name, stage=stage,
                                        orientation=orientation, fixed=fixed)
    n_layers = 1
    if layer_name is None:
        n_layers = surf_set.n_layers

    if foi is None:
        foi = [0, 256]

    if n_spatial_modes is None:
        n_spatial_modes = matlab.double([])
    else:
        n_spatial_modes = float(n_spatial_modes)

    if wois is None:
        _, time, _ = load_meg_sensor_data(data_fname)

        time_step = time[1] - time[0]  # Compute the difference in time between steps
        sampling_rate = 1000.0 / time_step
        win_steps = int(round(win_size / time_step))  # Calculate the number of steps in each window

        if (win_steps / n_temp_modes) < 2:
            raise ValueError(
                f"win_size={win_size} ms yields only {win_steps} samples "
                f"({sampling_rate:.2f} Hz sampling). With n_temp_modes={n_temp_modes}, "
                f"the ratio win_samples / n_temp_modes = {win_steps / n_temp_modes:.2f} < 2. "
                "Increase win_size or reduce n_temp_modes."
            )

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
            n_spatial_modes,
            spatialmodesname,
            1,
            0,
            nargout=3
        )

    cfg = {
        "spm": {
            "meeg": {
                "source": {
                    "invertiter": {
                        "D": np.asarray([data_fname], dtype="object"),
                        "val": (float(inversion_idx) + 1),
                        "whatconditions": {
                            "all": 1
                        },
                        "isstandard": {
                            "custom": {
                                "invfunc": 'Classic',
                                "invtype": 'EBB',
                                "woi": wois,
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
                        "crossval": np.asarray([0, 1], dtype=float)
                    }
                }
            }
        }
    }

    batch(cfg, viz=viz, spm_instance=spm_instance)

    with h5py.File(data_fname, 'r') as file:
        inv_struct = file[file['D']['other']['inv'][inversion_idx][0]]
        free_energy = np.squeeze(inv_struct['inverse']['crossF'][()])

    return [free_energy, wois]


# pylint: disable=R0912
def load_source_time_series(data_fname, mu_matrix=None, inv_fname=None, vertices=None,
                            inversion_idx=0):
    """
    Load or compute source-space time series from MEG data.

    This function reconstructs source-level activity either from precomputed inverse solutions
    (stored in the MEG data file or an external inversion file) or directly from a provided
    forward/inverse matrix (`mu_matrix`). Optionally, the reconstruction can be restricted to a
    subset of vertices.

    Parameters
    ----------
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    mu_matrix : np.ndarray or scipy.sparse matrix, optional
        Precomputed source reconstruction matrix (sources × sensors). If provided, this matrix is
        used directly to compute source time series. Default is None.
    inv_fname : str, optional
        Path to a file containing the precomputed inverse solution. If None, the inversion stored
        within `data_fname` is used. Default is None.
    vertices : list of int, optional
        List of vertex indices to extract source time series from. If None, all vertices are used.
    inversion_idx: int, optional
        Index of the inversion to use within the SPM data object (default: 0).

    Returns
    -------
    source_ts : np.ndarray
        Source-space time series (sources × time × trial).
    time : np.ndarray
        Time vector in milliseconds.
    mu_matrix : np.ndarray or scipy.sparse matrix
        Matrix used to reconstruct source activity from sensor data.

    Notes
    -----
    - If both `mu_matrix` and `inv_fname` are None, the function attempts to load the inverse
      solution embedded in `data_fname`.
    - When `vertices` is specified, only the corresponding subset of the inverse matrix is used.
    - Supports both single-trial and multi-trial MEG data structures.
    """

    sensor_data, time, _ = load_meg_sensor_data(data_fname)

    if mu_matrix is None:
        if inv_fname is None:
            inv_fname = data_fname
        check_inversion_exists(inv_fname, inversion_idx=inversion_idx)

        try:
            # HDF5 case
            with h5py.File(inv_fname, "r") as file:
                inv = file[file["D"]["other"]["inv"][inversion_idx][0]]["inverse"]

                # Load M (MAP projector)
                m_data = inv["M"]["data"][()]
                m_ir = inv["M"]["ir"][()]
                m_jc = inv["M"]["jc"][()]
                m_mat = csc_matrix((m_data, m_ir, m_jc),
                               shape=(int(m_ir.max()) + 1, len(m_jc) - 1))

                # Load U (spatial projector, may be dense or sparse)
                u_ref = inv["U"][0][0]
                if isinstance(file[u_ref], h5py.Group):  # sparse storage
                    dr_group = file[u_ref]
                    dr_data = dr_group["data"][()]
                    dr_ir = dr_group["ir"][()]
                    dr_jc = dr_group["jc"][()]
                    n_rows = int(max(dr_ir)) + 1
                    n_cols = len(dr_jc) - 1
                    u_mat = csc_matrix((dr_data, dr_ir, dr_jc),
                                   shape=(n_rows, n_cols))
                else:  # dense storage
                    u_mat = np.array(file[u_ref])

                # Temporal projector T
                temporal_projector = inv["T"][()].T
                tt_mat = temporal_projector @ temporal_projector.T

        except OSError:
            # MATLAB .mat case
            mat = loadmat(inv_fname, simplify_cells=True)
            inv = mat["D"]["other"]['inv'][inversion_idx]
            m_mat = csc_matrix(inv["M"]) if not issparse(inv["M"]) else inv["M"]

            u_mat_ = inv["U"][0] if isinstance(inv["U"], (list, tuple, np.ndarray)) else inv["U"]
            u_mat = csc_matrix(u_mat_) if not issparse(u_mat_) else u_mat_

            temporal_projector = inv["T"].T
            tt_mat = temporal_projector @ temporal_projector.T

        # Select vertices
        if vertices is not None:
            m_mat = m_mat[vertices, :]

        # Compose mu_matrix = M @ U
        mu_matrix = (m_mat @ u_mat) if issparse(u_mat) or issparse(m_mat) else m_mat @ u_mat

    else:
        if vertices is not None:
            mu_matrix = mu_matrix[vertices, :]
        tt_mat = np.eye(sensor_data.shape[1])

    # Keep track of original time length
    orig_n_time = sensor_data.shape[1]

    # Match sensor_data to tt_mat (for projection only)
    if tt_mat.shape[0] != orig_n_time:
        diff = tt_mat.shape[0] - orig_n_time
        if abs(diff) == 1:
            if diff > 0:
                # tt_mat expects one more time sample - pad zeros temporarily
                sensor_data = np.pad(sensor_data, ((0, 0), (0, diff)), mode='constant')
            else:
                # tt_mat expects one fewer - trim one time bin
                sensor_data = sensor_data[:, :tt_mat.shape[0]]
        else:
            raise ValueError(
                f"Temporal projector ({tt_mat.shape}) and sensor data ({sensor_data.shape}) "
                f"differ by more than one sample."
            )

    # Reconstruct source time series
    if sensor_data.ndim == 3:
        n_trials = sensor_data.shape[2]
        n_sources = mu_matrix.shape[0]
        source_ts = np.zeros((n_sources, orig_n_time, n_trials))
        for trial_idx in range(n_trials):
            y_mat = sensor_data[:, :, trial_idx] @ tt_mat  # apply temporal projection
            yv_mat = mu_matrix @ y_mat
            yv_full_mat = np.asarray(yv_mat)
            # Restore to original time length (pad/trim if needed)
            if yv_full_mat.shape[1] < orig_n_time:
                yv_full_mat = np.pad(
                    yv_full_mat,
                    ((0, 0), (0, orig_n_time - yv_full_mat.shape[1])),
                    mode='constant'
                )
            elif yv_full_mat.shape[1] > orig_n_time:
                yv_full_mat = yv_full_mat[:, :orig_n_time]
            source_ts[:, :, trial_idx] = yv_full_mat
    else:
        y_mat = sensor_data @ tt_mat
        yv_full_mat = np.asarray(mu_matrix @ y_mat)
        if yv_full_mat.shape[1] < orig_n_time:
            yv_full_mat = np.pad(
                yv_full_mat,
                ((0, 0), (0, orig_n_time - yv_full_mat.shape[1])),
                mode='constant'
            )
        elif yv_full_mat.shape[1] > orig_n_time:
            yv_full_mat = yv_full_mat[:, :orig_n_time]
        source_ts = yv_full_mat

    return source_ts, time, mu_matrix


def check_inversion_exists(data_file, inversion_idx=0):
    """
    Verify the presence of source inversion data within an SPM M/EEG dataset file.

    This function checks whether an SPM-compatible M/EEG `.mat` file (either v7.3+ HDF5 or
    pre-v7.3 MATLAB format) contains the required inversion structure (`D.other.inv`), indicating
    that a source reconstruction has been performed.

    Parameters
    ----------
    data_file : str
        Path to the SPM M/EEG `.mat` file.
    inversion_idx: int, optional
        Index of the inversion to use within the SPM data object (default: 0).

    Returns
    -------
    bool
        True if an inversion structure is present in the file.

    Raises
    ------
    KeyError
        If the inversion structure is missing, indicating that source inversion has not been run
        on the dataset.

    Notes
    -----
    - Supports both MATLAB v7.3+ (HDF5) and older pre-v7.3 formats.
    - Used internally by reconstruction utilities such as `load_source_time_series()`.
    """
    try:
        with h5py.File(data_file, 'r') as file:
            if 'inv' not in file['D']['other'] or len(file['D']['other']['inv'])<=inversion_idx:
                raise KeyError('Error: source inversion has not been run on this dataset')
    except OSError:
        mat_contents = loadmat(data_file)
        if 'inv' not in [x[0] for x in mat_contents['D'][0][0]['other'][0][0].dtype.descr] or \
                len(mat_contents['D'][0][0]['other'][0][0]['inv'])<=inversion_idx:
            raise KeyError('Error: source inversion has not been run on this dataset') # pylint: disable=raise-missing-from
    return True


def load_forward_model_vertices(data_file, inversion_idx=0):
    """
    Load the vertex coordinates from the forward model stored in an SPM M/EEG dataset.

    This function extracts the cortical mesh vertices used in the forward model from an
    SPM-compatible `.mat` file, supporting both MATLAB v7.3+ (HDF5) and legacy pre-v7.3 formats.
    The returned array contains the vertex coordinates in MNI space.

    Parameters
    ----------
    data_file : str
        Path to the SPM M/EEG `.mat` file containing the forward model.
    inversion_idx: int, optional
        Index of the forward model to load within the SPM data object (default: 0).

    Returns
    -------
    numpy.ndarray
        Array of vertex coordinates (n_vertices × 3) in MNI space.

    Raises
    ------
    TypeError
        If the file cannot be read as a valid MATLAB file.
    KeyError
        If the forward model or vertex data is missing from the file.

    Notes
    -----
    - For MATLAB v7.3+ files, the function reads directly from the HDF5 hierarchy:
      `D/other/inv/mesh/tess_mni/vert`.
    - For older MATLAB formats, it accesses the corresponding nested struct fields.
    - Typically used after source inversion to recover the 3D cortical geometry associated
      with the lead field model.
    """
    try:
        with h5py.File(data_file, 'r') as file:
            mesh_path = file[file['D']['other']['inv'][inversion_idx][0]]
            verts = mesh_path['mesh']['tess_mni']['vert'][()].T
    except (OSError, KeyError, TypeError):
        try:
            mat_contents = loadmat(data_file, struct_as_record=False, squeeze_me=True)
            verts = mat_contents['D'].other.inv[inversion_idx].mesh.tess_mni.vert
            verts = np.asarray(verts)
        except Exception as exc:
            raise KeyError("Could not load vertex data from the file.") from exc
    return verts


def get_lead_field_rms_diff(data_file, surf_set, inversion_idx=0):
    """
    Compute vertex-wise RMS difference of lead field vectors across cortical depths
    relative to the superficial (pial) layer.

    This function loads the forward model (gain matrix) generated by SPM and
    quantifies how the lead field topographies change with cortical depth.
    For each vertex, it computes the root-mean-square (RMS) difference between the
    superficial (layer 1) lead field and the corresponding lead fields at deeper
    layers, returning the RMS difference for the deepest layer.

    Parameters
    ----------
    data_file : str or pathlib.Path
        Path to the SPM-compatible MEG dataset (.mat) for which the inversion was run.
        The corresponding gain matrix file (e.g., ``SPMgainmatrix_<data_base>_1.mat``)
        is expected in the same directory.
    surf_set : LayerSurfaceSet
        The subject's laminar surface set used for forward modeling. Provides the
        number of layers and vertices per layer.
    inversion_idx: int, optional
        Index of the forward model to load within the SPM data object (default: 0).

    Returns
    -------
    rms_dff : numpy.ndarray, shape (n_vertices,)
        Vertex-wise RMS difference between the deepest and superficial layer lead fields.
        Values reflect how much lead field topography changes with depth.

    Notes
    -----
    - The gain matrix ('G') is assumed to be depth-ordered, with all vertices from
      layer 1 first, followed by subsequent layers.
    - RMSE is computed as the Euclidean distance between lead field vectors, averaged
      across sensors.
    - The function requires a valid SPM inversion; otherwise, it will not execute.
    - Useful for assessing layer-wise forward model sensitivity and laminar
      discriminability.
    """
    check_inversion_exists(data_file, inversion_idx=inversion_idx)

    data_path, data_file_name = os.path.split(data_file)
    data_base = os.path.splitext(data_file_name)[0]

    gainmat_fname = os.path.join(data_path, f'SPMgainmatrix_{data_base}_{inversion_idx+1}.mat')
    with h5py.File(gainmat_fname, 'r') as file:
        lf_mat = np.array(file['G'][()])

    verts_per_layer = surf_set.get_vertices_per_layer()
    layer_lf_mat = np.zeros((verts_per_layer, surf_set.n_layers, lf_mat.shape[1]))
    diff_layer_lf_mat = np.zeros((verts_per_layer, surf_set.n_layers))
    for i in range(surf_set.n_layers):
        layer_lf_mat[:, i, :] = lf_mat[i * verts_per_layer:(i + 1) * verts_per_layer, :]
    for j in range(verts_per_layer):
        for i in range(surf_set.n_layers):
            lf_diff = layer_lf_mat[j, i, :] - layer_lf_mat[j, 0, :]
            diff_layer_lf_mat[j, i] = np.sqrt(np.mean((lf_diff) ** 2))
    return diff_layer_lf_mat[:, -1]
