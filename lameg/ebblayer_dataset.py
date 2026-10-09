"""Dataset-level entry point for the experimental Python EBBlayer backend.

This module joins SPM numerical preparation to the validated Python
IND/SUM/DIFF inversion and official SPM ReML. It does not coregister the
forward model or modify the SPM M/EEG dataset, and does not replace the
legacy DANC-backed ``lameg.invert.invert_ebb_layer``.

Requires an existing forward model, its cached geodesic smoothing kernel,
and the matching MATLAB Runtime when actually invoking official SPM.
Ordinary module imports do not start MATLAB or import spm-python.

The default ``mode='compute'`` derives the inversion inputs from the SPM
M/EEG dataset; ``mode='saved'`` reuses a previous inversion's projectors
for regression testing only.
"""

import numpy as np

from lameg.ebblayer_inversion import invert_ebb_layer_python
from lameg.spm_prepare import prepare_ebblayer_from_spm


def _resolve_topk(value, n_pairs, name):
    """Limit default top-K to the available anatomical layer pairs."""
    if value is None:
        return min(2, n_pairs)
    if (not isinstance(value, (int, np.integer)) or
            isinstance(value, (bool, np.bool_)) or
            not (1 <= value <= n_pairs)):
        raise ValueError("{} must be an integer between 1 and {}".format(
            name, n_pairs))
    return int(value)


def invert_ebb_layer_from_spm(
        data_fname, kernel_fname, n_layers,
        mode="compute", n_spatial_modes=60, n_temp_modes=4,
        foi=(0, 48), woi=None, hann_windowing=False,
        spatial_modes_file=None, inversion_idx=0, noise_floor=None,
        sum_pair_topk=None, diff_pair_topk=None, runtime_dir=None,
        return_metadata=False, return_priors=False,
        eval_runner=None, reml_runner=None):
    """Invert a coregistered SPM M/EEG dataset using Python EBBlayer.

    Parameters
    ----------
    data_fname : str or Path
        SPM M/EEG .mat filename (with accompanying .dat as required by SPM).
        Must already contain a coregistered source/forward model.
    kernel_fname : str or Path
        Cached sparse geodesic smoother with QG and matching mesh faces.
    n_layers : int
        Number of cortical surfaces; vertices are ordered layer-major.
    mode : {'compute', 'saved'}
        ``compute`` independently derives temporal projectors, sensor
        covariance and (if no spatial-mode file exists) spatial modes.
        ``saved`` reuses a completed DANC inversion's projectors; this is
        primarily for exact regression against historical outputs.
    n_spatial_modes : int
        Number of spatial modes when mode='compute'.
    n_temp_modes : int
        Number of temporal modes when mode='compute'.
    foi : tuple(float, float)
        DCT frequency limits in Hz.
    woi : tuple(float, float) or None
        Time window of interest in milliseconds (None means entire epoch).
    hann_windowing : bool
        Apply Hann windowing to the temporal data.
    spatial_modes_file : str or Path or None
        Explicit spatial-mode file; if omitted, spm_prepare automatically
        reuses the dataset's *_testmodes.mat if it exists, else computes
        modes with SPM. Ignored in saved mode.
    inversion_idx : int
        Zero-based SPM inversion/head-model index.
    noise_floor : float or None
        Fixed noise-floor setting; defaults to SPM's exp(-5).
    sum_pair_topk, diff_pair_topk : int or None
        Number of retained SUM/DIFF hypotheses per column. None defaults
        to min(2, number of possible pairs), i.e. 1 for two layers and 2
        for eleven layers. Positive integers exceeding the available pair
        count are rejected.
    runtime_dir : path-like or None
        Location of the matching MATLAB Runtime passed to both components.
    return_metadata : bool
        If True, return a tuple (inversion_result, preparation_metadata).
        Preparation metadata contains spatial/temporal projectors and
        1-based channel, trial and sample indices.
    return_priors : bool
        Retain full per-pair prior diagnostics in the inversion result.
        This can be memory intensive for many layers.
    eval_runner, reml_runner : callable or None
        Dependency-injection hooks for unit tests. Normal runs omit these.

    Returns
    -------
    EBBlayerInversionResult or tuple
        In-memory source operator, posterior marginal variances, free
        energies, source weights and selection diagnostics, optionally
        accompanied by preparation metadata. The original dataset is not
        explicitly rewritten and no source operator is serialized to it.

    Notes
    -----
    This function expects a valid existing forward model and cached QG. It
    does not generate the surface, compute geodesic smoothing, coregister
    the MEG/MRI data, or write an inverse solution into the SPM dataset.
    SPM may create a gain-matrix cache during fresh preparation.
    """
    if (not isinstance(n_layers, (int, np.integer)) or
            isinstance(n_layers, (bool, np.bool_)) or n_layers < 2):
        raise ValueError("n_layers must be an integer >= 2")
    n_pairs = n_layers * (n_layers - 1) // 2
    sum_k = _resolve_topk(sum_pair_topk, n_pairs, "sum_pair_topk")
    diff_k = _resolve_topk(diff_pair_topk, n_pairs, "diff_pair_topk")

    prepared_output = prepare_ebblayer_from_spm(
        data_fname=data_fname,
        kernel_fname=kernel_fname,
        n_layers=int(n_layers),
        mode=mode,
        n_spatial_modes=n_spatial_modes,
        n_temp_modes=n_temp_modes,
        foi=foi,
        woi=woi,
        hann_windowing=hann_windowing,
        spatial_modes_file=spatial_modes_file,
        inversion_idx=inversion_idx,
        noise_floor=noise_floor,
        runtime_dir=runtime_dir,
        eval_runner=eval_runner,
        return_metadata=return_metadata,
    )

    if return_metadata:
        prepared, metadata = prepared_output
    else:
        prepared = prepared_output

    result = invert_ebb_layer_python(
        prepared,
        sum_pair_topk=sum_k,
        diff_pair_topk=diff_k,
        runtime_dir=runtime_dir,
        reml_runner=reml_runner,
        return_priors=return_priors,
    )
    if return_metadata:
        return result, metadata
    return result
