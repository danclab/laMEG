"""Numerical EBBlayer source-prior construction in Python.

This module implements the IND, continuous-interior SUM and DIFF source
covariance families used by the DANC SPM ``EBBlayer`` inversion. It is
independent of MATLAB, SPM Runtime, and the laMEG inversion interface.

The source ordering is *layer-major*: vertex j in layer k has index
``k * vertices_per_layer + j``. The smoothing kernel must use exactly the
same source ordering as the projected lead field.

The returned source covariances retain off-diagonal cross-layer blocks.
Each family is normalized so ``trace(UL @ Q @ UL.T) == 1`` (up to numerical
precision), matching DANC's ``spm_eeg_invert_classic.m`` implementation.

Reference validation (sub-104, 5-mm geodesic smoothing): Python versions
of the algorithm previously reproduced DANC prior covariances for 2 and
11 layers with relative errors near machine precision. This file is the
first production-oriented extraction; regression-test it independently
before integrating with the inversion API.
"""

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags, issparse


_TINY = 1e-30
_SUM_ROOT_TOL = 1e-12
_FAMILIES = ("IND", "SUM", "DIFF")


def _spm_inv(a):
    """Match the default diagonal regularization in SPM's spm_inv.m."""
    n, m = a.shape
    if n != m:
        raise ValueError("Data covariance must be square")
    norminf = np.linalg.norm(a, np.inf)
    tol = max(abs(np.spacing(norminf)) * n, np.exp(-32))
    return np.linalg.inv(a + np.eye(n) * tol)


def _positive_score(den, num):
    """Positive EBB Rayleigh score, retaining DANC's division order."""
    score = np.zeros(den.shape, dtype=np.float64)
    valid = (np.isfinite(den) & np.isfinite(num) &
             (den > _TINY) & (num > _TINY))
    score[valid] = (1.0 / num[valid]) / (1.0 / den[valid])
    score[~np.isfinite(score) | (score <= 0)] = 0.0
    return score


def _optimize_sum(gaa, gab, gbb, naa, nab, nbb):
    """Optimize q_a + r q_b over r>=0; preserve DANC's strict comparisons.

    Returns the unfiltered best score, mixing ratio, endpoint mask, and
    eligible-interior mask. Endpoint solutions are excluded from SUM priors.
    """
    size = gaa.size
    best = np.zeros(size, dtype=np.float64)
    ratio = np.full(size, np.nan, dtype=np.float64)
    endpoint = np.ones(size, dtype=bool)

    # Test r=0 first.
    valid = (np.isfinite(gaa) & np.isfinite(naa) &
             (gaa > _TINY) & (naa > _TINY))
    candidate = np.zeros(size)
    candidate[valid] = gaa[valid] / (4 * naa[valid])
    update = valid & np.isfinite(candidate) & (candidate > best)
    best[update], ratio[update], endpoint[update] = (
        candidate[update], 0.0, True)

    a2 = gbb * nab - gab * nbb
    a1 = gbb * naa - gaa * nbb
    a0 = gab * naa - gaa * nab
    scale = np.maximum(np.maximum(np.abs(a2), np.abs(a1)), np.abs(a0))
    valid_scale = np.isfinite(scale) & (scale > 0)
    threshold = _SUM_ROOT_TOL * scale
    quadratic = valid_scale & (np.abs(a2) > threshold)
    linear = valid_scale & ~quadratic & (np.abs(a1) > threshold)
    disc = a1 * a1 - 4 * a2 * a0
    valid_disc = quadratic & np.isfinite(disc) & (disc >= 0)

    def consider(r, available):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            den = gaa + 2 * r * gab + (r ** 2) * gbb
            num = naa + 2 * r * nab + (r ** 2) * nbb
            candidate_score = den / (4 * num)
        valid_r = (available & np.isfinite(r) & (r > 0) &
                   np.isfinite(den) & (den > _TINY) &
                   np.isfinite(num) & (num > _TINY) &
                   np.isfinite(candidate_score) & (candidate_score > best))
        best[valid_r] = candidate_score[valid_r]
        ratio[valid_r] = r[valid_r]
        endpoint[valid_r] = False

    r1 = np.full(size, np.nan)
    r2 = np.full(size, np.nan)
    rlin = np.full(size, np.nan)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        square_root = np.sqrt(np.maximum(disc, 0))
        r1[valid_disc] = ((-a1[valid_disc] + square_root[valid_disc]) /
                          (2 * a2[valid_disc]))
        r2[valid_disc] = ((-a1[valid_disc] - square_root[valid_disc]) /
                          (2 * a2[valid_disc]))
        rlin[linear] = -a0[linear] / a1[linear]
    consider(r1, valid_disc)
    consider(r2, valid_disc)
    consider(rlin, linear)

    # Test r=Inf last, with strict > as in DANC.
    valid = (np.isfinite(gbb) & np.isfinite(nbb) &
             (gbb > _TINY) & (nbb > _TINY))
    candidate = np.zeros(size)
    candidate[valid] = gbb[valid] / (4 * nbb[valid])
    update = valid & np.isfinite(candidate) & (candidate > best)
    best[update], ratio[update], endpoint[update] = (
        candidate[update], np.inf, True)

    interior = (~endpoint & np.isfinite(ratio) & (ratio > 0) &
                np.isfinite(best) & (best > 0))
    return best, ratio, endpoint, interior


def _select_topk(scores, k):
    """Select positive top-K hypotheses independently at each column.

    MATLAB's descending sort retains the earlier pair in the case of a tie;
    NumPy's argmax implements that behavior for the present pair ordering.
    """
    x = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    n_columns, n_pairs = x.shape
    if not 1 <= k <= n_pairs:
        raise ValueError("TOP-K must be between 1 and the number of pairs")
    work = x.copy()
    chosen = np.zeros((n_columns, n_pairs), dtype=bool)
    rows = np.arange(n_columns)
    for _ in range(k):
        winner = np.argmax(work, axis=1)
        eligible = work[rows, winner] > 0
        chosen[rows[eligible], winner[eligible]] = True
        work[rows, winner] = -np.inf
    return chosen


def _recover_underfilled_diff(scores, selected, smoothed_leads, invcov,
                              pairs, topk):
    """Recover DIFF scores lost to subtraction cancellation.

    The equivalent expression ``gaa + gbb - 2*gab`` can round to zero
    when smoothed leadfields from adjacent layers are nearly identical.
    Only revisit columns with fewer than TOP-K positive hypotheses, so
    ordinary DANC-compatible score/rank calculations remain untouched.
    Recompute using the actual difference leadfield to avoid cancellation.
    """
    underfilled = np.flatnonzero(np.count_nonzero(selected, axis=1) < topk)
    for vertex in underfilled:
        local = smoothed_leads[:, vertex, :]
        # Truly empty smoothing patches must remain unselected.
        if not np.any(local):
            continue
        for p, (a, b) in enumerate(pairs):
            if scores[vertex, p] > 0:
                continue
            delta = local[a] - local[b]
            den = float(np.dot(delta, delta))
            num = float(delta @ invcov @ delta)
            if (np.isfinite(den) and den > _TINY and
                    np.isfinite(num) and num > _TINY):
                value = den / (4.0 * num)
                if np.isfinite(value) and value > 0:
                    scores[vertex, p] = value
    # Retain DANC's deterministic earlier-pair tie break.
    return scores, _select_topk(scores, topk)


def _source_covariance(diagonal, row_parts, col_parts, value_parts, n):
    """Assemble sparse covariance including both cross-layer block halves."""
    base = diags(diagonal.ravel(), format="csc")
    if not row_parts:
        return base
    row = np.concatenate(row_parts)
    col = np.concatenate(col_parts)
    val = np.concatenate(value_parts)
    off = coo_matrix(
        (np.concatenate((val, val)),
         (np.concatenate((row, col)), np.concatenate((col, row)))),
        shape=(n, n),
    ).tocsc()
    return (base + off).tocsc()


def build_ebblayer_priors(ul, ayya, qg, n_layers,
                          sum_pair_topk=2, diff_pair_topk=2):
    """Construct DANC-compatible IND/SUM/DIFF priors from reduced leadfields.

    Parameters
    ----------
    ul : array_like, shape (n_modes, n_sources)
        Source-to-spatial-mode forward operator. Sources must be layer-major.
    ayya : array_like, shape (n_modes, n_modes)
        Sum of projected sensor/temporal covariance outer products. Do not
        divide by sample count; this matches the DANC ``AYYA`` convention.
    qg : scipy.sparse matrix, shape (n_sources, n_sources)
        Per-layer geodesic smoothing operator. No cross-layer smoothing.
    n_layers : int
        Number of cortical surfaces (>=2). All layers must have the same
        number of vertices.
    sum_pair_topk, diff_pair_topk : int
        Independent counts of retained candidate pairs per cortical column.
        For two layers, set both to 1; there is only one possible pair.

    Returns
    -------
    dict
        ``source_q``: list of three trace-normalized CSC source covariances
        in IND, SUM, DIFF order; ``sensor_q``: corresponding n_modes-by-n_modes
        matrices; ``pairs``: zero-based layer-pair tuples; ``traces``: pre-trace
        normalization constants; and full diagnostics (scores, ratios, masks).

    Notes
    -----
    This implements only prior construction. ReML, source posterior inference,
    and SPM dataset serialization are deliberately outside this module.
    """
    ul = np.asarray(ul, dtype=np.float64)
    ayya = np.asarray(ayya, dtype=np.float64)
    if ul.ndim != 2 or ayya.ndim != 2:
        raise ValueError("UL and AYYA must both be 2D arrays")
    n_modes, n_sources = ul.shape
    if ayya.shape != (n_modes, n_modes):
        raise ValueError("AYYA shape does not match UL spatial-mode dimension")
    if not np.all(np.isfinite(ul)) or not np.all(np.isfinite(ayya)):
        raise ValueError("UL and AYYA must contain finite values")
    if not isinstance(n_layers, (int, np.integer)) or n_layers < 2:
        raise ValueError("n_layers must be an integer >= 2")
    if n_sources % n_layers:
        raise ValueError("Every layer must have the same number of sources")
    if not issparse(qg) or qg.shape != (n_sources, n_sources):
        raise ValueError("QG must be a square sparse source smoothing matrix")
    if not np.all(np.isfinite(qg.data)):
        raise ValueError("QG must contain finite values")
    n_columns = n_sources // n_layers
    pairs = list(combinations(range(n_layers), 2))
    n_pairs = len(pairs)
    for k in (sum_pair_topk, diff_pair_topk):
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= n_pairs:
            raise ValueError("Each TOP-K must be an integer in [1, n_layer_pairs]")

    # B[layer, vertex, mode] = UL @ QG[:, layer * n_columns + vertex].
    b = np.asarray(qg.T @ ul.T, dtype=np.float64).reshape(
        n_layers, n_columns, n_modes)
    invcov = _spm_inv(ayya)
    ib = (b.reshape(n_sources, n_modes) @ invcov).reshape(
        n_layers, n_columns, n_modes)

    all_b = b.reshape(n_sources, n_modes)
    all_ib = ib.reshape(n_sources, n_modes)
    ind = _positive_score(
        np.einsum("ij,ij->i", all_b, all_b),
        np.einsum("ij,ij->i", all_b, all_ib),
    )
    max_ind = float(np.max(ind))
    if not np.isfinite(max_ind) or max_ind <= 0:
        raise ValueError("Independent prior has no positive source weights")
    ind /= max_ind

    score_sum = np.zeros((n_columns, n_pairs), dtype=np.float64)
    score_sum_raw = np.zeros_like(score_sum)
    score_diff = np.zeros_like(score_sum)
    mixing = np.full_like(score_sum, np.nan)
    endpoint = np.ones(score_sum.shape, dtype=bool)
    interior = np.zeros(score_sum.shape, dtype=bool)

    for p, (a, c) in enumerate(pairs):
        u, v = b[a], b[c]
        iu, iv = ib[a], ib[c]
        gaa = np.einsum("ij,ij->i", u, u)
        gbb = np.einsum("ij,ij->i", v, v)
        gab = np.einsum("ij,ij->i", u, v)
        naa = np.einsum("ij,ij->i", u, iu)
        nbb = np.einsum("ij,ij->i", v, iv)
        nab = (np.einsum("ij,ij->i", u, iv) +
               np.einsum("ij,ij->i", v, iu)) / 2.0
        raw, ratio, ep, inside = _optimize_sum(
            gaa, gab, gbb, naa, nab, nbb)
        score_sum_raw[:, p] = raw
        mixing[:, p] = ratio
        endpoint[:, p] = ep
        interior[:, p] = inside
        score_sum[:, p] = np.where(inside, raw, 0.0)
        score_diff[:, p] = _positive_score(
            gaa + gbb - 2 * gab, naa + nbb - 2 * nab) / 4.0

    keep_sum = _select_topk(score_sum, sum_pair_topk)
    keep_diff = _select_topk(score_diff, diff_pair_topk)
    score_diff, keep_diff = _recover_underfilled_diff(
        score_diff, keep_diff, b, invcov, pairs, diff_pair_topk)
    if np.any(keep_sum & ~interior):
        raise AssertionError("SUM endpoint incorrectly retained")

    dsum = np.zeros((n_layers, n_columns), dtype=np.float64)
    ddiff = np.zeros_like(dsum)
    sr, sc, sv = [], [], []
    dr, dc, dv = [], [], []
    for p, (a, c) in enumerate(pairs):
        indices = np.flatnonzero(keep_sum[:, p])
        if indices.size:
            rs = mixing[indices, p]
            vals = score_sum[indices, p]
            if not np.all(np.isfinite(rs) & (rs > 0)):
                raise ValueError("Invalid retained SUM mixing ratio")
            factor = 2.0 / (1.0 + rs ** 2)
            dsum[a, indices] += vals * factor
            dsum[c, indices] += vals * factor * (rs ** 2)
            sr.append(a * n_columns + indices)
            sc.append(c * n_columns + indices)
            sv.append(vals * factor * rs)
        indices = np.flatnonzero(keep_diff[:, p])
        if indices.size:
            vals = score_diff[indices, p]
            ddiff[a, indices] += vals
            ddiff[c, indices] += vals
            dr.append(a * n_columns + indices)
            dc.append(c * n_columns + indices)
            dv.append(-vals)

    max_sum = float(np.max(dsum))
    max_diff = float(np.max(ddiff))
    if min(max_sum, max_diff) <= 0:
        raise ValueError("A SUM or DIFF prior has no positive covariance entries")

    qind = diags(ind, format="csc")
    qsum = _source_covariance(dsum / max_sum,
                              sr, sc, [x / max_sum for x in sv], n_sources)
    qdiff = _source_covariance(ddiff / max_diff,
                               dr, dc, [x / max_diff for x in dv], n_sources)

    source_q = []
    sensor_q = []
    traces = []
    for name, q in zip(_FAMILIES, (qind, qsum, qdiff)):
        sensor = np.asarray((ul @ q) @ ul.T, dtype=np.float64)
        trace = float(np.trace(sensor))
        if not np.isfinite(trace) or trace <= 0:
            raise ValueError("{} prior has nonpositive sensor trace".format(name))
        source_q.append((q / trace).tocsc())
        sensor_q.append(sensor / trace)
        traces.append(trace)

    return {
        "pairs": pairs,
        "source_q": source_q,
        "sensor_q": sensor_q,
        "traces": traces,
        "score_sum_raw": score_sum_raw,
        "score_sum": score_sum,
        "score_diff": score_diff,
        "mixing": mixing,
        "endpoint": endpoint,
        "interior": interior,
        "keep_sum": keep_sum,
        "keep_diff": keep_diff,
        "n_layers": n_layers,
        "vertices_per_layer": n_columns,
    }
