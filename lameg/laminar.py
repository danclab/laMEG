"""
Laminar analysis tools for MEG source reconstruction and depth-resolved model comparison.

This module provides functions for performing and evaluating laminar source reconstructions of MEG
data using SPM. It integrates surface-based forward models, inversion algorithms, and analytical
tools for quantifying laminar-specific neural activity. The main focus is on comparing model
evidence across cortical depths, estimating current source density profiles, and identifying
regions of interest showing laminar modulation in power.

Key functionalities
-------------------
- **Model comparison:** Run Empirical Bayesian Beamformer (EBB) or Multiple Sparse Priors (MSP)
  inversions across cortical layers and compare model fits using free energy or cross-validation.
- **Sliding-window inversions:** Estimate time-resolved laminar model evidence to capture dynamic
  changes in cortical processing.
- **Current source density (CSD):** Compute laminar CSD profiles using the *StandardCSD* method
  from the *elephant* library, with optional layer-wise interpolation.
- **ROI analysis:** Identify regions of interest based on layer-specific power modulations between
  task and baseline periods.

Notes
-----
- All inversion routines depend on SPM's standalone interface for forward modeling and source
  reconstruction.
- Surface geometry and layer definitions are handled through `LayerSurfaceSet` objects.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import elephant
import neo
import quantities as pq

from lameg.invert import (invert_ebb, coregister, load_source_time_series, invert_msp,
                          invert_sliding_window_msp)
from lameg.util import ttest_rel_corrected


def model_comparison(fid_coords, data_fname, surf_set, stage='ds', orientation='link_vector',
                     fixed=True, method='EBB', viz=True, spm_instance=None, coregister_kwargs=None,
                     invert_kwargs=None):
    """
    Perform laminar model comparison by evaluating free energy and/or cross-validation error across
    cortical surfaces.

    This function coregisters MEG data to a subject's anatomical MRI and performs source
    reconstruction separately on each layer within a `LayerSurfaceSet`, using either the
    Empirical Bayesian Beamformer (EBB) or Multiple Sparse Priors (MSP) algorithm. It returns
    the free energy and cross-validation error for each layer, enabling laminar model comparison
    based on model evidence.

    Parameters
    ----------
    fid_coords : dict
        Dictionary of fiducial landmark coordinates, e.g.:
        ``{'nas': [x, y, z], 'lpa': [x, y, z], 'rpa': [x, y, z]}``
        Values must be expressed in MEG headspace coordinates (millimeters).
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    surf_set : LayerSurfaceSet
        The subject's surface set containing laminar meshes (e.g., pial, white, intermediate).
    stage : str, optional
        Surface mesh processing stage (default: 'ds').
    orientation : str, optional
        Dipole orientation model used for inversion (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    method : {'EBB', 'MSP'}, optional
        Source reconstruction method to use (default: 'EBB').
    viz : bool, optional
        Whether to display SPM visualizations during coregistration and inversion (default: True).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.
    coregister_kwargs : dict, optional
        Additional keyword arguments passed to the `coregister()` function.
    invert_kwargs : dict, optional
        Additional keyword arguments passed to the selected inversion function (`invert_ebb()` or
        `invert_msp()`).

    Returns
    -------
    f_vals : np.ndarray
        Array of free energy values (one per layer), representing model evidence.
    cv_errs : np.ndarray
        Array of cross-validation errors corresponding to each layer.

    Notes
    -----
    - The function iterates over all layer surfaces defined in `surf_set`.
    - A forward model is created for each layer surface within the SPM data object
    - Free energy serves as a quantitative parametric model comparison metric, with higher values
      indicating better model evidence.
    - Cross-validation serves as a quantitative nonparameteric model comparison metric, with lower
      values indicating better model evidence.
    - Requires prior generation of laminar surfaces and preprocessing of MEG data into
      SPM-compatible format.
    - Uses the Nolte single-shell MEG forward model as defined in SPM.
    """

    if coregister_kwargs is None:
        coregister_kwargs = {}
    if invert_kwargs is None:
        invert_kwargs = {}

    f_vals = []
    cv_errs = []
    layer_names = surf_set.get_layer_names()

    for layer_name in layer_names:
        coregister(
            fid_coords,
            data_fname,
            surf_set,
            layer_name=layer_name,
            stage=stage,
            orientation=orientation,
            fixed=fixed,
            viz=viz,
            spm_instance=spm_instance,
            **coregister_kwargs
        )

        f_val = np.nan
        cv_err = np.nan
        if method == 'EBB':
            [f_val, cv_err] = invert_ebb(
                data_fname,
                surf_set,
                layer_name=layer_name,
                stage=stage,
                orientation=orientation,
                fixed=fixed,
                n_spatial_modes='all',
                viz=viz,
                spm_instance=spm_instance,
                **invert_kwargs
            )
        elif method == 'MSP':
            [f_val, cv_err] = invert_msp(
                data_fname,
                surf_set,
                layer_name=layer_name,
                stage=stage,
                orientation=orientation,
                fixed=fixed,
                n_spatial_modes='all',
                viz=viz,
                spm_instance=spm_instance,
                **invert_kwargs
            )
        f_vals.append(f_val)
        cv_errs.append(cv_err)

    f_vals = np.array(f_vals)
    cv_errs = np.array(cv_errs)

    return f_vals, cv_errs


def sliding_window_model_comparison(prior, fid_coords, data_fname, surf_set, stage='ds',
                                    orientation='link_vector', fixed=True, viz=True,
                                    spm_instance=None, coregister_kwargs=None, invert_kwargs=None):
    """
    Perform laminar model comparison over time using sliding-window MSP inversions.

    This function performs time-resolved source reconstruction across multiple laminar surfaces
    using the Multiple Sparse Priors (MSP) algorithm within sliding time windows. For each layer
    in a `LayerSurfaceSet`, the function coregisters MEG data to the MRI, performs MSP inversion
    over successive windows, and computes the corresponding free energy values. The result is a
    layer-by-time representation of model evidence, suitable for laminar model comparison.

    Parameters
    ----------
    prior : int
        Index of the vertex to be used as the MSP prior (0-based Python indexing).
    fid_coords : dict
        Dictionary of fiducial landmark coordinates, e.g.:
        ``{'nas': [x, y, z], 'lpa': [x, y, z], 'rpa': [x, y, z]}``
        Values must be expressed in MEG headspace coordinates (millimeters).
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file).
    surf_set : LayerSurfaceSet
        The subject's surface set containing laminar meshes (e.g., pial, white, intermediate).
    stage : str, optional
        Surface mesh processing stage (default: 'ds').
    orientation : str, optional
        Dipole orientation model used for inversion (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    viz : bool, optional
        Whether to display SPM visualizations during coregistration and inversion (default: True).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.
    coregister_kwargs : dict, optional
        Additional keyword arguments passed to the `coregister()` function.
    invert_kwargs : dict, optional
        Additional keyword arguments passed to the `invert_sliding_window_msp()` function.

    Returns
    -------
    f_vals : np.ndarray, shape (n_layers, n_windows)
        Free energy values for each layer and time window.
    wois : np.ndarray, shape (n_windows, 2)
        Time windows of interest in milliseconds.

    Notes
    -----
    - This function applies the **Multiple Sparse Priors (MSP)** algorithm within sliding time
      windows, providing dynamic estimates of model evidence.
    - A forward model is created for each layer surface within the SPM data object
    - Free energy serves as a model comparison metric across both depth and time; higher values
      indicate better model evidence.
    - The `prior` index is internally adjusted to match MATLAB's 1-based indexing convention.
    - Requires prior MEG-MRI coregistration via `coregister()`.
    - Typically used to examine how laminar model evidence evolves during task-related dynamics.
    """

    if coregister_kwargs is None:
        coregister_kwargs = {}
    if invert_kwargs is None:
        invert_kwargs = {}

    f_vals = []
    wois = []
    layer_names = surf_set.get_layer_names()

    for layer_name in layer_names:
        coregister(
            fid_coords,
            data_fname,
            surf_set,
            layer_name=layer_name,
            stage=stage,
            orientation=orientation,
            fixed=fixed,
            viz=viz,
            spm_instance=spm_instance,
            **coregister_kwargs
        )

        [mesh_fvals, wois] = invert_sliding_window_msp(
            prior,
            data_fname,
            surf_set,
            layer_name=layer_name,
            stage=stage,
            orientation=orientation,
            fixed=fixed,
            n_spatial_modes='all',
            viz=viz,
            spm_instance=spm_instance,
            **invert_kwargs
        )
        f_vals.append(mesh_fvals)

    f_vals = np.vstack(f_vals)

    return f_vals, wois


def compute_csd(signal, thickness, sfreq, method='KCSD1D', smoothing=None, **kwargs):
    """
    Compute laminar current source density (CSD) from multi-layer neural signals.

    This function estimates the CSD profile across cortical depth using methods from the
    `elephant.current_source_density` module (typically StandardCSD or KCSD1D). The input
    should be a laminar time series (layers × time) spanning a known cortical thickness.
    Optionally, a smoothed, interpolated CSD can be computed across layers.

    Parameters
    ----------
    signal : ndarray, shape (n_layers, n_times)
        Laminar signal (e.g., LFP or current) with layers along the first axis and time along the
        second.
    thickness : float
        Cortical thickness in millimeters.
    sfreq : float
        Sampling frequency of the signal in Hertz.
    method : {'StandardCSD', 'KCSD1D'}, optional
        The method used for CSD estimation. Default is 'KCSD1D'.
    smoothing : {'linear', 'cubic', 'quadratic'} or None, optional
        Interpolation method for across-layer smoothing. If None, no smoothing is applied. Only
        applies to StandardCSD
    **kwargs
        Additional arguments passed to the underlying CSD method (e.g., `lambdas`, `Rs` for
        KCSD1D).

    Returns
    -------
    csd : ndarray, shape (n_layers, n_times) if smoothing is None, otherwise (500, n_times)
        The estimated current source density.  Layer-interpolated CSD if `smoothing` is specified
        (StandardCSD only).

    Notes
    -----
    - Smoothing uses `scipy.interpolate.interp1d` independently at each time point.
    - For `KCSD1D`, the signal is internally converted to microvolts before CSD estimation.
    - Ensure signals are continuous and layer-aligned with uniform spacing.
    """

    coords = pq.Quantity(np.linspace(0, thickness, num=signal.shape[0]).reshape(-1,1)) * pq.mm

    if method == 'StandardCSD':
        signal = neo.core.AnalogSignal(
            signal.T, units="T", sampling_rate=sfreq * pq.Hz
        )
        csd = elephant.current_source_density.estimate_csd(
            signal,
            coords,
            method = method
        ).as_array().T
        csd[:2,:]=0
        csd[-2:,:]=0

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
            return smoothed
        return csd

    if method == 'KCSD1D':
        lambdas = kwargs.get('lambdas', None)
        r_vals = kwargs.get('Rs', None)
        # create the AnalogSignal with units in microvolts
        signal = neo.core.AnalogSignal(
            signal.T * 1e6, units="uV",  # Use microvolts (µV) as the unit
            sampling_rate=sfreq * pq.Hz
        )

        kcsd = elephant.current_source_density.estimate_csd(
            signal,
            coords,
            method=method,
            lambdas=lambdas,
            Rs=r_vals
        ).as_array().T

        return kcsd

    raise ValueError('method must be either StandardCSD or KCSD1D')


# pylint: disable=R0915
def roi_power_comparison(data_fname, woi, baseline_woi, perc_thresh, surf_set, mu_matrix=None,
                         chunk_size=100, roi_idx=None, roi_mode='threshold'):
    """
    Compute laminar power changes and identify or reuse regions of interest (ROIs) based on
    layer-specific modulation.

    This function quantifies power changes between a window of interest (WOI) and a baseline
    window in both pial and white matter layers. ROI selection is controlled by `roi_mode`:

    - 'threshold':
        Compute the ROI from all vertices using the percentile threshold.
    - 'restrict_threshold':
        Use `roi_idx` as a candidate search space, and define the final ROI by thresholding
        only within that subset.
    - 'direct':
        Use `roi_idx` directly as the ROI without any additional thresholding.

    Parameters
    ----------
    data_fname : str
        Path to the MEG dataset (SPM-compatible .mat file) containing source time series.
    woi : tuple of float
        Window of interest in milliseconds, specified as (start, end).
    baseline_woi : tuple of float
        Baseline window in milliseconds, specified as (start, end), used for power normalization.
    perc_thresh : float
        Percentile threshold (0-100) for defining significant power changes.
    surf_set : LayerSurfaceSet
        Subject's laminar surface set containing pial and white matter meshes.
    stage : str, optional
        Surface mesh processing stage (default: 'ds').
    orientation : str, optional
        Dipole orientation model used for inversion (default: 'link_vector').
    fixed : bool, optional
        Whether to use fixed dipole orientations across layers (default: True).
    mu_matrix : np.ndarray or None, optional
        Precomputed source reconstruction matrix. If None, it is derived automatically from
        inversion data (default: None).
    chunk_size : int, optional
        Number of vertices processed per iteration to optimize memory usage (default: 1000).
    roi_idx : array_like or None, optional
        Vertex indices used depending on `roi_mode`:
        - ignored if `roi_mode='threshold'`
        - candidate search space if `roi_mode='restrict_threshold'`
        - final ROI if `roi_mode='direct'`
    roi_mode : {'threshold', 'restrict_threshold', 'direct'}, optional
        Strategy for ROI definition (default: 'threshold').

    Returns
    -------
    laminar_t_statistic : np.ndarray
        Paired-sample t-statistic comparing absolute power modulation between pial and white
        matter layers within the ROI.
    laminar_p_value : np.ndarray
        Corresponding p-value of the laminar t-test.
    deg_of_freedom : int
        Degrees of freedom for the laminar comparison.
    roi_idx : np.ndarray
        Indices of vertices defining the analyzed ROI.

    Notes
    -----
    - Power is computed as the mean Hilbert envelope within each time window.
    - Power change is calculated as WOI - baseline per trial and vertex.
    - In 'restrict_threshold' mode, percentile thresholds are computed only across the candidate
      vertices in `roi_idx`, and the returned `roi_idx` is the subset that survives thresholding.
    """

    valid_roi_modes = {'threshold', 'restrict_threshold', 'direct'}
    if roi_mode not in valid_roi_modes:
        raise ValueError(
            f"roi_mode must be one of {valid_roi_modes}, got {roi_mode!r}"
        )

    if roi_mode in {'restrict_threshold', 'direct'} and roi_idx is None:
        raise ValueError(
            f"roi_idx must be provided when roi_mode='{roi_mode}'"
        )

    verts_per_surf = surf_set.get_vertices_per_layer()
    n_layers = surf_set.n_layers

    pial_vertices = np.arange(verts_per_surf)

    # Load time indices
    ts_v, time, _ = load_source_time_series(
        data_fname,
        vertices=pial_vertices[:1],
        mu_matrix=mu_matrix
    )
    n_trials = ts_v.shape[2]
    base_t_idx = np.where((time >= baseline_woi[0]) & (time < baseline_woi[1]))[0]
    exp_t_idx = np.where((time >= woi[0]) & (time < woi[1]))[0]

    def load_and_compute_power(vertices, n_trials, chunk_size):
        vertices = np.asarray(vertices, dtype=int)
        base_incremental_power = np.zeros((len(vertices), n_trials))
        exp_incremental_power = np.zeros((len(vertices), n_trials))

        if len(vertices) == 0:
            return base_incremental_power, exp_incremental_power

        chunk_size = min(chunk_size, len(vertices))

        for start in range(0, len(vertices), chunk_size):
            end = min(start + chunk_size, len(vertices))
            vertex_slice = vertices[start:end]

            ts_chunk, _, _ = load_source_time_series(
                data_fname,
                vertices=vertex_slice,
                mu_matrix=mu_matrix
            )
            envelope = np.abs(hilbert(ts_chunk, axis=1))

            base_power_chunk = np.squeeze(np.mean(envelope[:, base_t_idx, :], axis=1))
            exp_power_chunk = np.squeeze(np.mean(envelope[:, exp_t_idx, :], axis=1))

            if base_power_chunk.ndim == 1:
                base_power_chunk = base_power_chunk[np.newaxis, :]
            if exp_power_chunk.ndim == 1:
                exp_power_chunk = exp_power_chunk[np.newaxis, :]

            base_incremental_power[start:end, :] = base_power_chunk
            exp_incremental_power[start:end, :] = exp_power_chunk

            del ts_chunk, envelope

        return base_incremental_power, exp_incremental_power

    if roi_mode == 'threshold':
        candidate_roi_idx = pial_vertices
    else:
        candidate_roi_idx = np.asarray(roi_idx, dtype=int)

    if roi_mode in {'threshold', 'restrict_threshold'}:
        pial_candidate_idx = candidate_roi_idx
        white_candidate_idx = (n_layers - 1) * verts_per_surf + candidate_roi_idx

        pial_base_power, pial_exp_power = load_and_compute_power(
            pial_candidate_idx,
            n_trials,
            chunk_size
        )
        white_base_power, white_exp_power = load_and_compute_power(
            white_candidate_idx,
            n_trials,
            chunk_size
        )

        pial_power_change = pial_exp_power - pial_base_power
        white_power_change = white_exp_power - white_base_power

        pial_t_statistic, _, _ = ttest_rel_corrected(
            pial_power_change,
            axis=-1
        )
        white_t_statistic, _, _ = ttest_rel_corrected(
            white_power_change,
            axis=-1
        )

        pial_thresh = np.percentile(pial_t_statistic, perc_thresh)
        white_thresh = np.percentile(white_t_statistic, perc_thresh)

        keep_mask = (
            (pial_t_statistic > pial_thresh) |
            (white_t_statistic > white_thresh)
        )

        roi_idx = candidate_roi_idx[keep_mask]

        if roi_idx.size == 0:
            raise ValueError(
                "No vertices survived thresholding. "
                "Try lowering perc_thresh or checking roi_idx."
            )

        pial_roi_power_change = np.mean(pial_power_change[keep_mask, :], axis=0)
        white_roi_power_change = np.mean(white_power_change[keep_mask, :], axis=0)

    elif roi_mode == 'direct':
        roi_idx = np.asarray(roi_idx, dtype=int)
        pial_roi_idx = roi_idx
        white_roi_idx = (n_layers - 1) * verts_per_surf + roi_idx

        pial_base_power, pial_exp_power = load_and_compute_power(
            pial_roi_idx,
            n_trials,
            min(len(pial_roi_idx), chunk_size)
        )
        white_base_power, white_exp_power = load_and_compute_power(
            white_roi_idx,
            n_trials,
            min(len(white_roi_idx), chunk_size)
        )

        pial_power_change = pial_exp_power - pial_base_power
        white_power_change = white_exp_power - white_base_power

        pial_roi_power_change = np.mean(pial_power_change, axis=0)
        white_roi_power_change = np.mean(white_power_change, axis=0)

    laminar_t_statistic, deg_of_freedom, laminar_p_value = ttest_rel_corrected(
        np.abs(pial_roi_power_change) - np.abs(white_roi_power_change),
        axis=-1
    )

    return laminar_t_statistic, laminar_p_value, deg_of_freedom, roi_idx
