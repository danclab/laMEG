"""Call official SPM's compiled ``spm_reml_sc`` from Python.

The caller supplies already projected covariance components, in SPM's
expected order (noise first, then source priors).  This deliberately does
not construct EBBlayer priors or change laMEG's existing DANC SPM backend.

``spm-python`` and MATLAB Runtime are imported only when ReML is invoked.
The optional ``runtime_dir`` argument supports installed builds for which
MATLAB Runtime autodetection fails.  On Linux, required linker settings
(e.g., LD_PRELOAD) must be configured *before starting Python*.

Compatible with the Python 3.7 syntax used by laMEG.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import issparse


def _dense(value):
    """Convert MATLAB sparse/dense numeric arrays to NumPy."""
    return np.asarray(value.toarray() if issparse(value) else value)


def _matlab_quote(path):
    """Quote an absolute path for a MATLAB single-quoted string literal."""
    return str(path).replace("'", "''")


def _official_runner(runtime_dir):
    """Resolve the compiled official SPM entry point on first use.

    The installed R2025b build currently needs the runtime-version shim.
    It changes matlab_runtime.impl's process-global version resolver, so
    callers must not mix different MATLAB Runtime versions in one process.
    """
    if runtime_dir is not None:
        runtime_root = os.path.realpath(os.path.expanduser(str(runtime_dir)))
        if not os.path.isdir(runtime_root):
            raise FileNotFoundError("MATLAB Runtime directory not found: " + runtime_root)
        import matlab_runtime.impl as mr
        from matlab_runtime.utils import guess_matlab_version
        mr.guess_pymatlab_version = (
            lambda module: guess_matlab_version(runtime_root)
        )

    try:
        from spm import Runtime
    except ImportError as exc:
        raise RuntimeError(
            "Official spm-python is not installed in this environment"
        ) from exc

    return lambda code: Runtime.call("eval", code, nargout=0)


def run_reml(ayya, components, n_samples, fixed_covariance,
             runtime_dir=None, eval_runner=None):
    """Estimate covariance hyperparameters using official ``spm_reml_sc``.

    Parameters
    ----------
    ayya : (n_modes, n_modes) array
        Sum of projected data outer products (not divided by ``n_samples``).
    components : sequence of (n_modes, n_modes) arrays
        Covariance components in MATLAB's cell-array order.  For EBBlayer
        stage 1 use ``[Qe, Lind, Lsum, Ldiff]``; for stage 2 use
        ``[Q0, LQPL]``.  These are not re-normalized here.
    n_samples : positive number
        Number of projected temporal-mode samples, e.g., 240.
    fixed_covariance : (n_modes, n_modes) array
        Final argument ``V`` of ``spm_reml_sc`` (DANC passes ``Q0``).
    runtime_dir : path-like or None
        Optional installed MATLAB Runtime root, e.g.
        ``~/MATLAB/MATLAB_Runtime/R2025b``.  Leave None for autodetection.
    eval_runner : callable or None
        Optional test hook accepting the MATLAB command as one string.
        Normal callers should leave None.

    Returns
    -------
    dict with ``covariance`` (2D ndarray), ``hyperparameters`` (1D ndarray),
    and ``free_energy`` (float).  The hyperparameter vector includes *all*
    passed components, including the noise term at index zero.

    Notes
    -----
    This function only performs ReML; it does not compute the posterior
    source operator, update an SPM dataset, or initialize a parallel pool.
    All communication is through temporary MAT files, cleaned on exit.
    """
    ayya = _validated_matrix("ayya", ayya)
    if ayya.shape[0] != ayya.shape[1]:
        raise ValueError("ayya must be square")
    shape = ayya.shape
    fixed = _validated_matrix("fixed_covariance", fixed_covariance, shape)

    try:
        matrices = list(components)
    except TypeError as exc:
        raise ValueError("components must be a nonempty sequence of matrices") from exc
    if not matrices:
        raise ValueError("components must be nonempty")
    matrices = [
        _validated_matrix("component {}".format(i), q, shape)
        for i, q in enumerate(matrices)
    ]

    try:
        samples = float(n_samples)
    except (ValueError, TypeError) as exc:
        raise ValueError("n_samples must be finite and positive") from exc
    if not np.isfinite(samples) or samples <= 0:
        raise ValueError("n_samples must be finite and positive")

    if eval_runner is not None and not callable(eval_runner):
        raise TypeError("eval_runner must be callable")
    runner = eval_runner if eval_runner is not None else _official_runner(runtime_dir)

    with TemporaryDirectory(prefix="lameg_spm_reml_") as folder:
        inp = Path(folder) / "input.mat"
        out = Path(folder) / "output.mat"
        payload = {"AYYA": ayya, "V": fixed, "Nn": samples}
        names = []
        for index, q in enumerate(matrices):
            name = "Q{:03d}".format(index + 1)
            names.append("S." + name)
            payload[name] = q
        savemat(str(inp), payload, do_compression=True)

        # Keep MATLAB statements on one line, separated by semicolons.
        # Flattening a multiline try/catch/end previously caused syntax errors.
        code = (
            "S=load('{}'); ".format(_matlab_quote(inp))
            + "Q={" + ",".join(names) + "}; "
            + "[C,h,Ph,F]=spm_reml_sc(S.AYYA,[],Q,S.Nn,-4,16,S.V); "
            + "h=full(h(:)); "
            + "save('{}','C','h','F','-v7');".format(_matlab_quote(out))
        )
        runner(code)
        if not out.is_file():
            raise RuntimeError("Official SPM did not write ReML output")
        result = loadmat(str(out))

    covariance = _dense(result["C"]).astype(np.float64, copy=False)
    weights = _dense(result["h"]).astype(np.float64, copy=False).ravel()
    free_energy = float(_dense(result["F"]).item())
    if covariance.shape != shape:
        raise RuntimeError("Unexpected ReML covariance shape: {}".format(covariance.shape))
    if weights.shape != (len(matrices),):
        raise RuntimeError("Unexpected ReML hyperparameter count: {}".format(weights.size))
    if (not np.all(np.isfinite(covariance)) or
            not np.all(np.isfinite(weights)) or
            not np.isfinite(free_energy)):
        raise RuntimeError("SPM returned non-finite ReML results")

    return {
        "covariance": covariance,
        "hyperparameters": weights,
        "free_energy": free_energy,
    }


def _validated_matrix(name, matrix, shape=None):
    arr = _dense(matrix)
    if arr.ndim != 2 or (shape is not None and arr.shape != shape):
        raise ValueError("{} must be a 2D matrix of shape {}".format(
            name, shape if shape is not None else "(n_modes, n_modes)"))
    if np.iscomplexobj(arr):
        raise ValueError("{} must be real-valued".format(name))
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("{} must contain real numerical values".format(name)) from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("{} must contain finite values".format(name))
    return arr
