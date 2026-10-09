"""Prepare Python EBBlayer inversion inputs from an SPM M/EEG dataset.

An experimental, non-mutating numerical preparation boundary. Given an SPM
M/EEG file with an existing coregistered forward model and a *cached* sparse
geodesic smoother, compute the same reduced-sensor data used by DANC's
``spm_eeg_invert_classic`` without invoking that inversion routine.

Two explicitly different modes are supported:

* ``mode='saved'`` reads projectors stored by an *earlier completed* SPM
  inversion. This is an exact-reference regression path, NOT a fresh-prep path.
* ``mode='compute'`` rebuilds spatial/temporal modes from a coregistered
  dataset. It uses an optional existing SPM spatial-mode file, otherwise SPM's
  ``spm_svd``. It does not require a previous source inversion.

The MATLAB code is executed by the *official* compiled spm-python Runtime,
and data exchange uses temporary MATLAB v7 files. SPM may create/reuse its
own gain-matrix cache when ``spm_eeg_lgainmat`` runs in compute mode; the
function never explicitly saves/overwrites the input M/EEG dataset.

Python 3.7 compatible. No MATLAB or SPM dependency at import time.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, issparse

from lameg.ebblayer_inversion import PreparedEBBlayerData
from lameg.spm_reml import _official_runner, _matlab_quote


def _load_cached_kernel(path):
    """Read cached SPM MATLAB v7.3 sparse QG without densifying it."""
    try:
        with h5py.File(str(path), "r") as file:
            q = file["QG"]
            if not isinstance(q, h5py.Group):
                raise ValueError("QG must be a MATLAB sparse matrix")
            data = np.asarray(q["data"]).ravel()
            ir = np.asarray(q["ir"]).ravel().astype(np.int64)
            jc = np.asarray(q["jc"]).ravel().astype(np.int64)
            if jc.size < 2 or jc[0] != 0 or jc[-1] != len(data):
                raise ValueError("Invalid MATLAB sparse QG column pointers")
            shape = (len(jc) - 1, len(jc) - 1)
            if ir.size != data.size or (ir.size and (ir.min() < 0 or ir.max() >= shape[0])):
                raise ValueError("Invalid MATLAB sparse QG row indices")
            return csc_matrix((data, ir, jc), shape=shape)
    except OSError as exc:
        raise ValueError("Kernel must be a MATLAB v7.3 file with sparse QG") from exc


def _matlab_statements(data_file, kernel_file, output_file, mode,
                       n_spatial_modes, n_temp_modes, foi, woi,
                       hann_windowing, spatial_modes_file, inversion_idx,
                       noise_floor):
    """Generate semicolon-delimited MATLAB. No path modifications/addpath."""
    if mode not in ("saved", "compute"):
        raise ValueError("mode must be 'saved' or 'compute'")
    val = int(inversion_idx) + 1
    commands = [
        "D=spm_eeg_load('{}')".format(_matlab_quote(data_file)),
        "val={}".format(val),
        "D.val=val",
        "assert(numel(D.inv)>=val,'SPM dataset has no inversion/head model at index')",
        "K=load('{}','faces')".format(_matlab_quote(kernel_file)),
        "assert(isfield(K,'faces'),'Smoothing kernel has no faces metadata')",
        "assert(isequal(double(K.faces),double(D.inv{val}.mesh.tess_mni.face))," \
        "'Smoothing kernel mesh faces do not match dataset')",
        "clear K",
    ]
    if mode == "saved":
        commands.extend([
            "assert(isfield(D.inv{val},'inverse'),'No completed inversion available')",
            "inv=D.inv{val}.inverse",
            "UL=full(inv.L)",
            "A=full(inv.U{1})",
            "S=full(inv.T)",
            "Ic=inv.Ic",
            "if iscell(Ic); Ic=Ic{1}; end",
            "It=inv.It",
            "Ik=inv.Ik",
            "assert(size(A,1)==size(UL,1),'Saved spatial projector mismatch')",
            "assert(size(S,2)>0,'Empty saved temporal projector')",
        ])
    else:
        commands.extend([
            "[L,D]=spm_eeg_lgainmat(D)",
            "modality=D.inv{val}.forward.modality",
            "if strcmp(modality,'MEG'); "
            "Ic=setdiff(union(D.indchantype('MEG'),D.indchantype('MEGPLANAR')),badchannels(D)); "
            "else; Ic=setdiff(D.indchantype(modality),badchannels(D)); end",
            "assert(size(L,1)==numel(Ic),'Leadfield and sensor-channel mismatch')",
        ])
        if spatial_modes_file is not None:
            commands.extend([
                "sm=load('{}','U','megind','testchans')".format(
                    _matlab_quote(spatial_modes_file)),
                "assert(iscell(sm.U) && numel(sm.U)==1," \
                "'Expected one SPM spatial-mode block')",
                "assert(isempty(sm.testchans),'Nonzero held-out channels are unsupported')",
                "assert(isequal(double(sm.megind(:)),double(Ic(:)))," \
                "'Spatial-mode channel ordering differs from lead fields')",
                "A=full(sm.U{1})",
                "assert(size(A,1)=={},'Wrong number of spatial modes')".format(n_spatial_modes),
                "assert(size(A,2)==size(L,1),'Spatial modes and leadfield mismatch')",
            ])
        else:
            commands.extend([
                "[Usp,~,~]=spm_svd(L*L',1e-12)",
                "assert(size(Usp,2)>={}, 'Insufficient spatial modes')".format(n_spatial_modes),
                "A=full(Usp(:,1:{})')".format(n_spatial_modes),
            ])
        commands.extend([
            "UL=full(A*L)",
            "clear L Usp",
            "w=[{} {}]".format(woi[0], woi[1]) if woi is not None
            else "w=1000*[min(D.time) max(D.time)]",
            "ind=(w/1000-D.timeonset)*D.fsample+1",
            "It=fix(max(1,ind(1)):min(ind(2),length(D.time)))",
            "assert(numel(It)>=2,'Insufficient time samples')",
            "pst=1000*D.time",
            "pst=pst(It)",
            "dur=(pst(end)-pst(1))/1000",
            "assert(dur>0,'Time window duration must be positive')",
            "dct=(It-It(1))/2/dur",
            "T=spm_dctmtx(numel(It),numel(It))",
            "j=find(dct>={} & dct<={})".format(foi[0], foi[1]),
            "T=T(:,j)",
            "assert(size(T,2)>={},'Insufficient DCT modes')".format(n_temp_modes),
            "if isfield(D.inv{val},'inverse') && isfield(D.inv{val}.inverse,'trials') && "
            "~isempty(D.inv{val}.inverse.trials); "
            "trial=D.inv{val}.inverse.trials; else; trial=D.condlist; end",
            "badtrialind=D.badtrials",
            "Ik=[]",
            "for jj=1:numel(trial); c=D.indtrial(trial{jj}); "
            "[~,ib]=intersect(c,badtrialind); c=c(setxor(1:numel(c),ib)); "
            "Ik=[Ik c]; end",
            "assert(~isempty(Ik),'No usable trials')",
            "YTY=zeros(size(T,2),size(T,2))",
            "if {}; W=spdiags(spm_hanning(numel(It)),0,numel(It),numel(It)); "
            "else; W=1; end".format(1 if hann_windowing else 0),
            "YYt=zeros(numel(It),numel(It))",
            "for ii=1:numel(Ik); Y=A*double(D(Ic,It,Ik(ii))); YYt=YYt+Y'*Y; end",
            "YYt=W'*(YYt/numel(Ik))*W",
            "YTY=T'*YYt*T",
            "[Uv,~]=svd(YTY)",
            "S=T*Uv(:,1:{})".format(n_temp_modes),
            "clear YYt YTY Uv T W",
        ])
    commands.extend([
        "assert(size(UL,2)==size(D.inv{val}.mesh.tess_mni.vert,1)," \
        "'Leadfield/source mesh vertex count mismatch')",
        "AYYA=zeros(size(A,1),size(A,1))",
        "for ii=1:numel(Ik); Y=A*double(D(Ic,It,Ik(ii)))*S; "
        "AYYA=AYYA+Y*Y'; end",
        "Nn=numel(Ik)*size(S,2)",
        "if isfield(D.inv{val},'inverse') && isfield(D.inv{val}.inverse,'QE'); "
        "QE=D.inv{val}.inverse.QE; else; QE=1; end",
        "if isfield(D.inv{{val}},'inverse') && isfield(D.inv{{val}}.inverse,'Qe0'); "
        "Qe0=D.inv{{val}}.inverse.Qe0; else; Qe0={:.17g}; end".format(noise_floor),
        "AQeA=A*QE*A'",
        "Qe=AQeA/trace(AQeA)",
        "Q0=Qe0*trace(AYYA)*Qe/Nn",
        "assert(all(isfinite(AYYA(:))) && all(isfinite(Q0(:)))," \
        "'Nonfinite projected covariance')",
        "fprintf('Prepared: %d spatial modes, %d sources, %d trials, %d temporal modes\\n'," \
        "size(UL,1),size(UL,2),numel(Ik),size(S,2))",
        "save('{}','UL','AYYA','Qe','Q0','Nn','A','S','Ic','It','Ik','-v7')".format(
            _matlab_quote(output_file)),
    ])
    return "; ".join(commands) + ";"


def prepare_ebblayer_from_spm(data_fname, kernel_fname, n_layers,
                               mode="compute", n_spatial_modes=60,
                               n_temp_modes=4, foi=(0, 48), woi=None,
                               hann_windowing=False, spatial_modes_file=None,
                               inversion_idx=0, noise_floor=None,
                               runtime_dir=None, eval_runner=None,
                               return_metadata=False):
    """Prepare a ``PreparedEBBlayerData`` from a coregistered SPM dataset.

    ``mode='compute'`` computes temporal modes and data covariances afresh;
    if a spatial modes file is given it reuses its channel projector. For
    exact reference regression ``mode='saved'`` uses projectors from an
    already completed DANC/SPM inversion and ignores mode/window options.

    This stage requires a pre-existing 5-mm (or otherwise chosen) sparse QG
    kernel. It never computes geodesic smoothing. The input dataset must be
    accessible through SPM (including its companion .dat). A forward model
    must have been previously coregistered.

    ``eval_runner`` is a test hook, not generally needed; normal usage
    invokes the compiled official SPM Runtime and requires the matching
    MATLAB Runtime. Returns the PreparedEBBlayerData alone, or a tuple of
    that object and a dictionary of spatial/temporal projector metadata.
    """
    if mode not in ("compute", "saved"):
        raise ValueError("mode must be 'compute' or 'saved'")
    if (not isinstance(n_layers, (int, np.integer)) or
            isinstance(n_layers, (bool, np.bool_)) or n_layers < 2):
        raise ValueError("n_layers must be an integer >= 2")
    for name, value in [("n_spatial_modes", n_spatial_modes),
                        ("n_temp_modes", n_temp_modes)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("{} must be a positive integer".format(name))
    if not isinstance(inversion_idx, (int, np.integer)) or inversion_idx < 0:
        raise ValueError("inversion_idx must be a nonnegative integer")
    if not isinstance(hann_windowing, (bool, np.bool_)):
        raise ValueError("hann_windowing must be boolean")
    if len(foi) != 2 or not all(np.isfinite(foi)) or foi[0] < 0 or foi[0] >= foi[1]:
        raise ValueError("foi must be an increasing nonnegative frequency interval")
    if woi is not None and (len(woi) != 2 or not all(np.isfinite(woi)) or
                            woi[0] >= woi[1]):
        raise ValueError("woi must be an increasing finite time interval in milliseconds")
    if noise_floor is None:
        noise_floor = float(np.exp(-5))
    if not np.isfinite(noise_floor) or noise_floor < 0:
        raise ValueError("noise_floor must be finite and nonnegative")
    if eval_runner is not None and not callable(eval_runner):
        raise TypeError("eval_runner must be callable")

    data_file = Path(data_fname).expanduser().resolve()
    kernel_file = Path(kernel_fname).expanduser().resolve()
    for name, path in [("SPM dataset", data_file), ("Smoothing kernel", kernel_file)]:
        if not path.is_file():
            raise FileNotFoundError("{} not found: {}".format(name, path))
    qg = _load_cached_kernel(kernel_file)
    if qg.shape[0] % int(n_layers):
        raise ValueError("Kernel source count must be divisible by n_layers")

    if mode == "saved":
        modes_file = None
    elif spatial_modes_file is not None:
        modes_file = Path(spatial_modes_file).expanduser().resolve()
        if not modes_file.is_file():
            raise FileNotFoundError("Spatial modes file not found: {}".format(modes_file))
    else:
        # Automatically reuse the DANC/SPM spatial mode file if present.
        candidate = data_file.with_name(data_file.stem + "_testmodes.mat")
        modes_file = candidate if candidate.is_file() else None

    with TemporaryDirectory(prefix="lameg_spm_prepare_") as folder:
        output = Path(folder) / "prepared.mat"
        code = _matlab_statements(data_file, kernel_file, output, mode,
                                  n_spatial_modes, n_temp_modes, foi, woi,
                                  hann_windowing, modes_file, inversion_idx,
                                  noise_floor)
        runner = eval_runner if eval_runner is not None else _official_runner(runtime_dir)
        runner(code)
        if not output.is_file():
            raise RuntimeError("SPM did not produce the preparation output")
        result = loadmat(str(output))

    def dense(name):
        if name not in result:
            raise RuntimeError("SPM preparation omitted '{}'".format(name))
        x = result[name]
        return np.asarray(x.toarray() if issparse(x) else x)

    ul = np.asarray(dense("UL"), dtype=np.float64)
    data = PreparedEBBlayerData(
        ul=ul,
        ayya=np.asarray(dense("AYYA"), dtype=np.float64),
        qg=qg,
        qe=np.asarray(dense("Qe"), dtype=np.float64),
        q0=np.asarray(dense("Q0"), dtype=np.float64),
        n_samples=float(dense("Nn").item()),
        n_layers=int(n_layers),
    )
    if not return_metadata:
        return data
    metadata = {
        "spatial_projector": dense("A"),
        "temporal_projector": dense("S"),
        "channels": np.asarray(dense("Ic"), dtype=np.int64).ravel(),  # MATLAB 1-based
        "times": np.asarray(dense("It"), dtype=np.int64).ravel(),      # MATLAB 1-based
        "trials": np.asarray(dense("Ik"), dtype=np.int64).ravel(),     # MATLAB 1-based
        "mode": mode,
        "spatial_modes_file": str(modes_file) if modes_file is not None else None,
    }
    return data, metadata
