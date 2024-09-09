"""
This module provides tools for performing laminar analysis of MEG signals.
It integrates functionalities for performing source reconstruction using different methods
(Empirical Bayesian Beamformer and Multiple Sparse Priors), and assessing model fit via free
energy and cross validation error. It also includes capabilities for analyzing laminar current
source density and comparing regional brain activity across different conditions or layers.

Key functionalities include:
- Source reconstruction with options to choose between different methods and evaluate model fits.
- Calculation of current source density from laminar neural signals.
- Comparison of power changes across different regions and layers to identify areas of interest.
"""

import numpy as np
from scipy.interpolate import interp1d
import elephant
import neo
import quantities as pq

from lameg.invert import (invert_ebb, coregister, load_source_time_series, invert_msp,
                          invert_sliding_window)
from lameg.util import ttest_rel_corrected


def model_comparison(nas, lpa, rpa, mri_fname, mesh_fnames, data_fname, method='EBB', viz=True,
                     spm_instance=None, coregister_kwargs=None, invert_kwargs=None):
    """
    Compare model fits using different meshes by computing the free energy.

    This function runs source reconstruction algorithms (either Empirical Bayesian Beamformer or
    Multiple Sparse Priors) on a set of meshes and compares their model fits using the free energy
    and cross-validation error metrics.

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
    mesh_fnames : list
        List of filenames for different meshes.
    data_fname : str
        Filename of the MEG data.
    method : str, optional
        Source reconstruction method, either 'EBB' or 'MSP'. Default is 'EBB'.
    viz : bool, optional
        Whether to display SPM visualizations. Default is True.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM software. Default is None.
    coregister_kwargs : dict, optional
        Keyword arguments specifically for the `coregister` function.
    invert_kwargs : dict, optional
        Keyword arguments specifically for the `invert` function.

    Returns
    -------
    f_vals : np.array
        Free energy values for each mesh.
    cv_errs : np.array
        The cross-validation error for each mesh.

    Notes
    -----
    - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    - Free energy is used as a measure of model fit, with higher values indicating better fit.
    """

    if coregister_kwargs is None:
        coregister_kwargs = {}
    if invert_kwargs is None:
        invert_kwargs = {}

    f_vals = []
    cv_errs = []
    for mesh_fname in mesh_fnames:
        coregister(
            nas,
            lpa,
            rpa,
            mri_fname,
            mesh_fname,
            data_fname,
            viz=viz,
            spm_instance=spm_instance,
            **coregister_kwargs
        )

        f_val = np.nan
        cv_err = np.nan
        if method == 'EBB':
            [f_val, cv_err] = invert_ebb(
                mesh_fname,
                data_fname,
                1,
                viz=viz,
                spm_instance=spm_instance,
                **invert_kwargs
            )
        elif method == 'MSP':
            [f_val, cv_err] = invert_msp(
                mesh_fname,
                data_fname,
                1,
                viz=viz,
                spm_instance=spm_instance,
                **invert_kwargs
            )
        f_vals.append(f_val)
        cv_errs.append(cv_err)

    f_vals = np.array(f_vals)
    cv_errs = np.array(cv_errs)

    return f_vals, cv_errs


def sliding_window_model_comparison(prior, nas, lpa, rpa, mri_fname, mesh_fnames, data_fname,
                                    viz=True, spm_instance=None, coregister_kwargs=None,
                                    invert_kwargs=None):
    """
    Compare model fits across different meshes using a sliding window approach.

    This function runs source reconstruction using the Multiple Sparse Priors (MSP) method in
    sliding time windows on a set of meshes. It compares the model fits for each mesh by computing
    the free energy in each window.

    Parameters
    ----------
    prior : float
        Index of the vertex to be used as a prior.
    nas : list
        NASion fiducial coordinates.
    lpa : list
        Left PreAuricular fiducial coordinates.
    rpa : list
        Right PreAuricular fiducial coordinates.
    mri_fname : str
        Filename of the MRI data.
    mesh_fnames : list
        List of filenames for different meshes.
    data_fname : str
        Filename of the MEG data.
    viz : bool, optional
        Whether or not to show SPM visualization. Default is True.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.
    coregister_kwargs : dict, optional
        Keyword arguments specifically for the `coregister` function.
    invert_kwargs : dict, optional
        Keyword arguments specifically for the `invert_sliding_window` function.

    Returns
    -------
    f_vals : list
        Free energy values for each mesh.
    wois : list
        Windows of interest.

    Notes
    -----
    - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    - Free energy is used as a measure of model fit, with higher values indicating better fit.
    - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
    """

    if coregister_kwargs is None:
        coregister_kwargs = {}
    if invert_kwargs is None:
        invert_kwargs = {}

    f_vals = []
    wois = []
    for mesh_fname in mesh_fnames:
        coregister(
            nas,
            lpa,
            rpa,
            mri_fname,
            mesh_fname,
            data_fname,
            viz=viz,
            spm_instance=spm_instance,
            **coregister_kwargs
        )

        [mesh_fvals, wois] = invert_sliding_window(
            prior,
            mesh_fname,
            data_fname,
            1,
            viz=viz,
            spm_instance=spm_instance,
            **invert_kwargs
        )
        f_vals.append(mesh_fvals)

    f_vals = np.vstack(f_vals)

    return f_vals, wois


def compute_csd(signal, thickness, sfreq, smoothing=None):
    """
    Compute the laminar Current Source Density (CSD) from a given signal.

    This function calculates CSD using the Standard CSD method. It takes a multi-layered neural
    signal, and computes the CSD. An optional smoothing step can be applied to the CSD output.

    Parameters
    ----------
    signal : numpy.ndarray
        The neural signal matrix, where rows correspond to different layers and columns to time
        points.
    thickness : float
        The laminar thickness of the cortex from which the signal was recorded, in millimeters.
    sfreq : float
        Sampling frequency of the signal in Hertz.
    smoothing : str, optional
        Specifies the kind of smoothing to apply to the CSD. Acceptable values are those compatible
        with `scipy.interpolate.interp1d`, such as 'linear', 'nearest', 'zero', 'slinear',
        'quadratic', 'cubic', etc. If None, no smoothing is applied. Default is None.

    Returns
    -------
    ret_vals : list
        A list containing the CSD matrix as the first element. If smoothing is applied, the second
        element is the smoothed CSD matrix. The CSD matrix dimensions are layers x time points.

    Notes
    -----
    - The function requires the 'neo', 'quantities' (pq), 'numpy' (np), 'elephant', and
      'scipy.interpolate.interp1d' libraries.
    - The CSD is calculated using the Standard CSD method provided by the 'elephant' package.
    - Smoothing is applied across layers and is independent for each time point.
    - The returned CSD matrix has the same number of time points as the input signal but can have a
      different number of layers if smoothing is applied.
    """

    signal = neo.core.AnalogSignal(
        signal.T, units="T", sampling_rate=sfreq*pq.Hz
    )
    coords = pq.Quantity(np.linspace(0, thickness, num=signal.shape[1]).reshape(-1,1)) * pq.mm

    csd = elephant.current_source_density.estimate_csd(
        signal, coords,
        method = "StandardCSD"
    ).as_array().T
    csd[:2,:]=0
    csd[-2:,:]=0

    ret_vals=[csd]

    if smoothing is not None:
        layers, time = csd.shape
        smoothed = []
        layer_x = np.linspace(0, 1, num=layers)
        smooth_x = np.linspace(0, 1, num=500)
        for t_idx in range(time):
            interp_function = interp1d(layer_x, csd[:, t_idx], kind=smoothing)
            interp_y = interp_function(smooth_x)
            smoothed.append(interp_y)
        smoothed = np.array(smoothed).T
        ret_vals.append(smoothed)

    return ret_vals


def roi_power_comparison(data_fname, woi, baseline_woi, mesh, n_layers, perc_thresh,
                         mu_matrix=None, chunk_size=None):
    """
    Computes and compares power changes in pial and white matter layers to define
    regions of interest (ROI) based on significant power shifts.

    This function calculates power changes in the pial and white matter layers during a specified
    window of interest (WOI) and a baseline window. It identifies ROIs by comparing these changes
    against a percentile threshold and performs a relative comparison of power changes between
    the layers to assess laminar differences.

    Parameters
    ----------
    data_fname : str
        Filename of the data file containing source time series.
    woi : tuple
        Window of interest, specified as a start and end time (in milliseconds).
    baseline_woi : tuple
        Baseline window of interest for comparison, specified as start and end time (in
        milliseconds).
    mesh : nibabel.gifti.GiftiImage
        Gifti surface mesh.
    n_layers : int
        Number of layers in the cortical model.
    perc_thresh : float
        Percentile threshold for determining significant changes in power.
    mu_matrix : ndarray, optional
        Lead field matrix (source x sensor). Default is None.
    chunk_size : int, optional
        Number of vertices to load source time series from at once. If None, will load all at the
        same time. Default is None.

    Returns
    -------
    laminar_t_statistic : float
        The t-statistic for laminar power comparison.
    laminar_p_value : float
        The p-value associated with the laminar power comparison.
    deg_of_freedom : int
        Degrees of freedom for the statistical test.
    roi_idx : list of int
        Indices of vertices considered as ROIs.

    """

    verts_per_surf = int(mesh.darrays[0].data.shape[0] / n_layers)
    pial_vertices = np.arange(verts_per_surf)
    white_vertices = (n_layers - 1) * verts_per_surf + np.arange(verts_per_surf)

    # Load data incrementally
    def load_and_compute_power(vertices, time_indices, chunk_size, n_trials):
        incremental_power = np.zeros((len(vertices), n_trials))
        if chunk_size is None:
            chunk_size = len(vertices)
        for start in range(0, len(vertices), chunk_size):
            end = min(start + chunk_size, len(vertices))
            vertex_slice = vertices[start:end]
            ts_chunk, _, _ = load_source_time_series(
                data_fname, vertices=vertex_slice, mu_matrix=mu_matrix
            )
            # Compute variance directly on loaded chunk
            power_chunk = np.var(ts_chunk[:, time_indices, :], axis=1)
            incremental_power[start:end,:] = power_chunk
            del ts_chunk  # Free up memory
        return incremental_power

    # Load time indices
    ts_v, time, _ = load_source_time_series(
        data_fname,
        vertices=pial_vertices[:1],
        mu_matrix=mu_matrix
    )
    n_trials = ts_v.shape[2]
    base_t_idx = np.where((time >= baseline_woi[0]) & (time < baseline_woi[1]))[0]
    exp_t_idx = np.where((time >= woi[0]) & (time < woi[1]))[0]

    # Compute power changes for pial and white matter layers
    pial_base_power = load_and_compute_power(pial_vertices, base_t_idx, chunk_size, n_trials)
    pial_exp_power = load_and_compute_power(pial_vertices, exp_t_idx, chunk_size, n_trials)
    pial_power_change = (pial_exp_power - pial_base_power) / pial_base_power

    white_base_power = load_and_compute_power(white_vertices, base_t_idx, chunk_size, n_trials)
    white_exp_power = load_and_compute_power(white_vertices, exp_t_idx, chunk_size, n_trials)
    white_power_change = (white_exp_power - white_base_power) / white_base_power

    # Define ROI
    pial_t_statistic, _, _ = ttest_rel_corrected((pial_exp_power - pial_base_power), axis=-1)
    white_t_statistic, _, _ = ttest_rel_corrected((white_exp_power - white_base_power), axis=-1)

    pial_thresh = np.percentile(pial_t_statistic, perc_thresh)
    white_thresh = np.percentile(white_t_statistic, perc_thresh)
    roi_idx = np.where((pial_t_statistic > pial_thresh) | (white_t_statistic > white_thresh))[0]

    pial_roi_power_change = np.mean(pial_power_change[roi_idx, :], axis=0)
    white_roi_power_change = np.mean(white_power_change[roi_idx, :], axis=0)

    # Compare power t statistic
    laminar_t_statistic, deg_of_freedom, laminar_p_value = ttest_rel_corrected(
        np.abs(pial_roi_power_change) - np.abs(white_roi_power_change),
        axis=-1
    )

    return laminar_t_statistic, laminar_p_value, deg_of_freedom, roi_idx
