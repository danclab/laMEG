"""Orchestrate Python EBBlayer inversion from SPM-prepared numerical inputs.

This module implements the source-posterior computation after an SPM forward
model has already been projected onto spatial and temporal modes. It does NOT
coregister sensors, prepare SPM modes, read/write SPM M/EEG datasets, or replace
``lameg.invert.invert_ebb_layer``. Those integration boundaries are deliberate.

The numerical operations follow DANC's ``spm_eeg_invert_classic.m``:

1. Construct trace-normalized IND/SUM/DIFF sparse source priors in Python.
2. Call official ``spm_reml_sc`` to estimate source-family weights.
3. Combine the full, non-diagonal source covariances.
4. Call official ReML again to estimate a final source scale.
5. Form the source reconstruction operator M and posterior marginal variance Cq.

Sources are layer-major, as in ``lameg.ebblayer``. The caller must provide
``AYYA`` as the SUM of projected data outer products, not the average.

This module is Python 3.7-compatible and does not import official SPM at import
time. Running the default ReML backend requires spm-python and its matching
MATLAB Runtime. If necessary, set LD_PRELOAD before starting Python.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse, csc_matrix

from lameg.ebblayer import build_ebblayer_priors


@dataclass
class PreparedEBBlayerData:
    """SPM-prepared inputs for one inversion window.

    ul : ndarray, (n_spatial_modes, n_sources)
        Projected lead field. Sources use *layer-major* ordering.
    ayya : ndarray, (n_spatial_modes, n_spatial_modes)
        Sum of projected sensor-data outer products (not sample-normalized).
    qg : sparse matrix, (n_sources, n_sources)
        Source smoothing kernel matching ``ul`` and the forward-model mesh.
    qe : ndarray, (n_spatial_modes, n_spatial_modes)
        Trace-normalized projected sensor-noise covariance.
    q0 : ndarray, (n_spatial_modes, n_spatial_modes)
        Fixed covariance passed to both ReML stages. First-stage noise
        component is ``qe``; second-stage noise component is ``q0``.
    n_samples : int or float
        Number of samples projected into the spatial/temporal mode space.
    n_layers : int
        Number of cortical layers, with an identical source count per layer.

    ``ul``, ``ayya``, ``qe``, and ``q0`` must already be in the SAME reduced
    sensor-mode space. Construction validates dimensions but deliberately
    does not copy large arrays or recompute the forward model.
    """

    ul: object
    ayya: object
    qg: object
    qe: object
    q0: object
    n_samples: float
    n_layers: int

    def __post_init__(self):
        ul = np.asarray(self.ul)
        if ul.ndim != 2 or not ul.shape[0] or not ul.shape[1]:
            raise ValueError("ul must be a nonempty (n_modes, n_sources) matrix")
        n_modes, n_sources = ul.shape
        if (not isinstance(self.n_layers, (int, np.integer)) or
                isinstance(self.n_layers, (bool, np.bool_)) or
                self.n_layers < 2 or n_sources % self.n_layers):
            raise ValueError("n_layers must be >= 2 and divide the source count")
        if not issparse(self.qg) or self.qg.shape != (n_sources, n_sources):
            raise ValueError("qg must be a square sparse smoothing matrix matching ul")
        if not np.all(np.isfinite(self.qg.data)):
            raise ValueError("qg contains non-finite values")
        for name in ("ul", "ayya", "qe", "q0"):
            value = np.asarray(getattr(self, name))
            required = (n_modes, n_sources) if name == "ul" else (n_modes, n_modes)
            if value.shape != required:
                raise ValueError("{} must have shape {}".format(name, required))
            if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
                raise ValueError("{} must contain real numerical values".format(name))
            if not np.all(np.isfinite(value)):
                raise ValueError("{} contains non-finite values".format(name))
        if not np.isfinite(self.n_samples) or self.n_samples <= 0:
            raise ValueError("n_samples must be finite and positive")


@dataclass
class EBBlayerInversionResult:
    """Numerical inversion outputs; no modifications to any SPM dataset."""

    operator: np.ndarray                   # (n_sources, n_modes), DANC's M
    posterior_variance: np.ndarray         # (n_sources,), DANC's Cq
    source_covariance: csc_matrix          # full weighted source prior, incl. off-diagonals
    source_weights: np.ndarray             # IND/SUM/DIFF from first-stage ReML
    final_source_scale: float              # second-stage ReML source hyperparameter
    source_free_energy: float              # first-stage F
    free_energy: float                     # second-stage F
    sensor_covariance: np.ndarray          # second-stage Cy
    n_sum_retained: int
    n_diff_retained: int
    n_sum_endpoints: int
    priors: object = None                  # optional complete build_ebblayer_priors dict


def _checked_reml_result(value, count, modes, stage):
    """Check ReML's contract before using results for posterior inference."""
    if not isinstance(value, dict):
        raise RuntimeError("{} ReML did not return a result dictionary".format(stage))
    try:
        h = np.asarray(value["hyperparameters"], dtype=np.float64).reshape(-1)
        cy = np.asarray(value["covariance"], dtype=np.float64)
        f = float(value["free_energy"])
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("{} ReML returned invalid results".format(stage)) from exc
    if (h.shape != (count,) or cy.shape != (modes, modes) or
            not np.all(np.isfinite(h)) or not np.all(np.isfinite(cy)) or
            not np.isfinite(f)):
        raise RuntimeError("{} ReML returned invalid shapes or values".format(stage))
    return h, cy, f


def invert_ebb_layer_python(prepared, sum_pair_topk=2, diff_pair_topk=2,
                            runtime_dir=None, reml_runner=None,
                            return_priors=False):
    """Invert SPM-prepared data using Python priors and official SPM ReML.

    Parameters
    ----------
    prepared : PreparedEBBlayerData
        Inputs from a single SPM inversion window. Preparation itself is
        deliberately outside this function.
    sum_pair_topk, diff_pair_topk : int
        Independent pair counts per cortical column. With 2 layers set both
        to 1; with 11 layers the DANC default is 2.
    runtime_dir : path-like or None
        Passed to ``lameg.spm_reml.run_reml`` for official MATLAB Runtime.
    reml_runner : callable or None
        An optional injectable ReML runner with the same signature as
        ``run_reml``; chiefly useful for testing without official SPM.
    return_priors : bool
        Whether to retain the large per-pair diagnostic arrays in the result.

    Returns
    -------
    EBBlayerInversionResult
        Reconstruction operator, posterior marginal variances, source-family
        weights and model evidence. It does not save data to an SPM M/EEG file.

    Notes
    -----
    Source-space covariance blocks must stay full; replacing them by their
    diagonal yields a different inversion. DANC's two-stage covariance layout:

        stage 1 components: [Qe, L_ind, L_sum, L_diff], V=Q0
        stage 2 components: [Q0, UL @ qp @ UL.T], V=Q0

    M = (h_final * UL @ qp).T / Cy, where ``qp`` is the weighted *full*
    source-space covariance. Cq is its posterior marginal variance.
    """
    if not isinstance(prepared, PreparedEBBlayerData):
        raise TypeError("prepared must be a PreparedEBBlayerData instance")
    if reml_runner is None:
        # Lazy import: ordinary Python 3.7 and DANC users don't need spm-python.
        from lameg.spm_reml import run_reml
        reml_runner = run_reml
    if not callable(reml_runner):
        raise TypeError("reml_runner must be callable")

    priors = build_ebblayer_priors(
        prepared.ul, prepared.ayya, prepared.qg, prepared.n_layers,
        sum_pair_topk=sum_pair_topk, diff_pair_topk=diff_pair_topk)

    stage1 = reml_runner(
        prepared.ayya, [prepared.qe] + priors["sensor_q"],
        prepared.n_samples, prepared.q0, runtime_dir=runtime_dir)
    h1, _, source_f = _checked_reml_result(stage1, 4, prepared.ayya.shape[0],
                                           "First-stage")
    source_weights = h1[1:].copy()

    # Keep the complete non-diagonal covariance (SUM/DIFF cross-layer blocks).
    source_covariance = sum(
        (float(weight) * q for weight, q in zip(source_weights,
                                                priors["source_q"])),
        csc_matrix(prepared.qg.shape))
    source_covariance = source_covariance.tocsc()
    lqp = np.asarray(prepared.ul @ source_covariance, dtype=np.float64)
    lqpl = lqp @ prepared.ul.T

    stage2 = reml_runner(
        prepared.ayya, [prepared.q0, lqpl],
        prepared.n_samples, prepared.q0, runtime_dir=runtime_dir)
    h2, cy, final_f = _checked_reml_result(stage2, 2, prepared.ayya.shape[0],
                                           "Second-stage")
    scale = float(h2[1])

    # MATLAB: M = (h_final * UL * qp)' / Cy.
    # Transpose+solve is equivalent to MATLAB's right matrix division.
    try:
        operator = np.linalg.solve(cy.T, scale * lqp).T
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Second-stage sensor covariance is singular") from exc

    # DANC: Cq = hp*diag(qp) - sum((hp*UL*qp).*M')'.
    # einsum avoids allocating another n_sources x n_modes dense matrix.
    marginal = (scale * source_covariance.diagonal() -
                scale * np.einsum("ij,ji->j", lqp, operator))
    if not np.all(np.isfinite(operator)) or not np.all(np.isfinite(marginal)):
        raise RuntimeError("Source posterior contains non-finite values")

    return EBBlayerInversionResult(
        operator=operator,
        posterior_variance=marginal,
        source_covariance=source_covariance,
        source_weights=source_weights,
        final_source_scale=scale,
        source_free_energy=source_f,
        free_energy=final_f,
        sensor_covariance=cy,
        n_sum_retained=int(np.count_nonzero(priors["keep_sum"])),
        n_diff_retained=int(np.count_nonzero(priors["keep_diff"])),
        n_sum_endpoints=int(np.count_nonzero(priors["endpoint"])),
        priors=priors if return_priors else None,
    )
