import numpy as np
from scipy.interpolate import interp1d
import elephant
import neo
import quantities as pq

from lameg.invert import invert_ebb, invert_msp, invert_sliding_window, coregister, load_source_time_series
from lameg.util import matlab_context, ttest_rel_corrected


def model_comparison(nas, lpa, rpa, mri_fname, mesh_fnames, data_fname, method='EBB', gain_mat_fnames=None,
                     viz=True, mat_eng=None, **kwargs):
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
    method (str, optional): Source reconstruction method, either 'EBB' or 'MSP'. Default is 'EBB'.
    gain_mat_fnames (list, optional): List of filenames of the precomputed gain matrices, one for each mesh in
                                      mesh_fnames. If None, they will be computed. Default is None
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    **kwargs: Additional keyword arguments are passed directly to the source reconstruction functions (invert_ebb or
              invert_msp).

    Returns:
    list: A list containing the free energy values corresponding to each mesh, and the cross validation error for each
          mesh.

    Notes:
    - Free energy is used as a measure of model fit, with higher values indicating better fit.
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    if gain_mat_fnames is None:
        gain_mat_fnames=[None for _ in mesh_fnames]

    f_vals = []
    cv_errs = []
    with matlab_context(mat_eng) as eng:
        for l_idx, mesh_fname in enumerate(mesh_fnames):
            coregister(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, viz=viz, mat_eng=eng)
            if method == 'EBB':
                [f_val, cv_err] = invert_ebb(mesh_fname, data_fname, 1, gain_mat_fname=gain_mat_fnames[l_idx],
                                             viz=viz, mat_eng=eng, **kwargs)
            elif method == 'MSP':
                [f_val, cv_err] = invert_msp(mesh_fname, data_fname, 1, gain_mat_fname=gain_mat_fnames[l_idx],
                                             viz=viz, mat_eng=eng, **kwargs)
            f_vals.append(f_val)
            cv_errs.append(cv_err)

    f_vals = np.array(f_vals)
    cv_errs = np.array(cv_errs)

    return f_vals, cv_errs


def sliding_window_model_comparison(prior, nas, lpa, rpa, mri_fname, mesh_fnames, data_fname, gain_mat_fnames=None,
                                    viz=True, mat_eng=None, **kwargs):
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
    gain_mat_fnames (list, optional): List of filenames of the precomputed gain matrices, one for each mesh in
                                      mesh_fnames. If None, they will be computed. Default is None
    viz (boolean, optional): Whether or not to show SPM visualization. Default is True
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    **kwargs: Additional keyword arguments are passed directly to the invert_sliding_window function.

    Returns:
    tuple: A tuple containing a list of free energy values for each mesh and the windows of interest (wois).

    Notes:
    - Free energy is used as a measure of model fit, with higher values indicating better fit.
    - The function requires MATLAB and SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    - The prior index is adjusted by adding 1 to align with MATLAB's 1-based indexing.
    """
    if gain_mat_fnames is None:
        gain_mat_fnames = [None for _ in mesh_fnames]

    f_vals = []
    wois = []
    with matlab_context(mat_eng) as eng:
        for l_idx, mesh_fname in enumerate(mesh_fnames):
            coregister(nas, lpa, rpa, mri_fname, mesh_fname, data_fname, viz=viz, mat_eng=eng)

            [mesh_fvals, wois] = invert_sliding_window(prior, mesh_fname, data_fname, 1,
                                                            gain_mat_fname=gain_mat_fnames[l_idx], viz=viz,
                                                            mat_eng=eng, **kwargs)
            f_vals.append(mesh_fvals)

    f_vals = np.array(f_vals)

    return f_vals, wois


def compute_csd(signal, thickness, sfreq, smoothing=None):
    """
    Compute the laminar Current Source Density (CSD) from a given signal.

    This function calculates CSD using the Standard CSD method. It takes a multi-layered neural signal, typically
    from laminar probes, and computes the CSD. An optional smoothing step can be applied to the CSD output.

    Parameters:
    signal (numpy.ndarray): The neural signal matrix, where rows correspond to different layers and columns to time
                            points.
    thickness (float): The laminar thickness of the cortex from which the signal was recorded, in millimeters.
    sfreq (float): Sampling frequency of the signal in Hertz.
    smoothing (str, optional): Specifies the kind of smoothing to apply to the CSD. Acceptable values are those
                               compatible with scipy.interpolate.interp1d, such as 'linear', 'nearest', 'zero',
                               'slinear', 'quadratic', 'cubic', etc. If None, no smoothing is applied. Default is None.

    Returns:
    list: A list containing the CSD matrix as the first element. If smoothing is applied, the second element
          is the smoothed CSD matrix. The CSD matrix dimensions are layers x time points.

    Notes:
    - The function requires the 'neo', 'quantities' (pq), 'numpy' (np), 'elephant', and 'scipy.interpolate.interp1d'
      libraries.
    - The CSD is calculated using the Standard CSD method provided by the 'elephant' package.
    - Smoothing is applied across layers and is independent for each time point.
    - The returned CSD matrix has the same number of time points as the input signal but can have a different number
      of layers if smoothing is applied.
    """
    sig = neo.core.AnalogSignal(
        signal.T, units="T", sampling_rate=sfreq*pq.Hz
    )
    th = pq.Quantity(np.linspace(0, thickness, num=signal.shape[0]).reshape(-1,1)) * pq.mm

    csd = elephant.current_source_density.estimate_csd(
        sig, th,
        method = "StandardCSD"
    ).as_array().T

    ret_vals=[csd]

    if smoothing is not None:
        layers, time = csd.shape
        smoothed = []
        x = np.linspace(0, 1, num=layers)
        xs = np.linspace(0, 1, num=500)
        for t in range(time):
            fx = interp1d(x, csd[:, t], kind=smoothing)
            ys = fx(xs)
            smoothed.append(ys)
        smoothed = np.array(smoothed).T
        ret_vals.append(smoothed)

    return ret_vals


def roi_power_comparison(data_fname, woi, baseline_woi, mesh, n_layers, perc_thresh, mu_matrix=None):
    """
    Computes and compares power changes in pial and white matter layers to define
    regions of interest (ROI) based on significant power shifts.

    This function calculates power changes in the pial and white matter layers during a specified
    window of interest (WOI) and a baseline window. It identifies ROIs by comparing these changes
    against a percentile threshold and performs a relative comparison of power changes between
    the layers to assess laminar differences.

    Parameters:
    data_fname (str): Filename of the data file containing source time series.
    woi (tuple): Window of interest, specified as a start and end time (in milliseconds).
    baseline_woi (tuple): Baseline window of interest for comparison, specified as start and end time (in milliseconds).
    mesh (nibabel.gifti.GiftiImage): Gifti surface mesh.
    n_layers (int): Number of layers in the cortical model.
    perc_thresh (float): Percentile threshold for determining significant changes in power.
    mu_matrix (ndarray, optional): Lead field matrix (source x sensor). Default is None.

    Returns:
    tuple: Contains laminar t-statistic, laminar p-value, and indices of vertices considered as ROIs.
    """
    verts_per_surf = int(mesh.darrays[0].data.shape[0] / n_layers)
    pial_vertices = np.arange(verts_per_surf)
    white_vertices = (n_layers - 1) * verts_per_surf + np.arange(verts_per_surf)

    pial_layer_ts, time = load_source_time_series(
        data_fname,
        vertices=pial_vertices,
        mu_matrix=mu_matrix
    )
    white_layer_ts, time = load_source_time_series(
        data_fname,
        vertices=white_vertices,
        mu_matrix=mu_matrix
    )

    base_t_idx = np.where((time >= baseline_woi[0]) & (time < baseline_woi[1]))[0]
    exp_t_idx = np.where((time >= woi[0]) & (time < woi[1]))[0]

    # Pial power
    pial_base_power = np.squeeze(np.var(pial_layer_ts[:, base_t_idx, :], axis=1))
    pial_exp_power = np.squeeze(np.var(pial_layer_ts[:, exp_t_idx, :], axis=1))
    pial_power_change = (pial_exp_power - pial_base_power) / pial_base_power

    # White matter power
    white_base_power = np.squeeze(np.var(white_layer_ts[:, base_t_idx, :], axis=1))
    white_exp_power = np.squeeze(np.var(white_layer_ts[:, exp_t_idx, :], axis=1))
    white_power_change = (white_exp_power - white_base_power) / white_base_power

    # Define ROI
    pial_t_statistic, pial_p_value = ttest_rel_corrected((pial_exp_power - pial_base_power), axis=-1)
    white_t_statistic, white_p_value = ttest_rel_corrected((white_exp_power - white_base_power), axis=-1)

    pial_thresh = np.percentile(pial_t_statistic, perc_thresh)
    white_thresh = np.percentile(white_t_statistic, perc_thresh)
    roi_idx = np.where((pial_t_statistic > pial_thresh) | (white_t_statistic > white_thresh))[0]

    pial_roi_power_change = np.mean(pial_power_change[roi_idx, :], axis=0)
    white_roi_power_change = np.mean(white_power_change[roi_idx, :], axis=0)

    # Compare power t statistic should be positive (more power in pial layer)
    laminar_t_statistic, laminar_p_value = ttest_rel_corrected(
        np.abs(pial_roi_power_change) - np.abs(white_roi_power_change),
        axis=-1
    )
    df = len(pial_roi_power_change)-1

    return laminar_t_statistic, laminar_p_value, df, roi_idx
