"""Compute disjoint groups of MEG pickups in independent DUNEuro processes."""

from contextlib import ExitStack
from importlib import import_module
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from joblib import Parallel, delayed, parallel_backend
import numpy as np


def _write_pickup_fields(path, pickup_fields, mesh, source_rr_m, source_nn, coil_rr_m, coil_nn,
                         initialize_mpi):
    """Keep each transfer matrix and the driver that constructed it in the same process."""
    if initialize_mpi:
        # A fresh Python worker doesn't inherit its parent's initialized MPI runtime.
        mpi = import_module('mpi4py.MPI')
        if not mpi.Is_initialized():
            raise RuntimeError("Enable mpi4py automatic initialization in FEM worker processes.")
    # Each file contains this worker's pickup values in source-major order.
    # Writing normally also reports a full filesystem as an I/O error. Workers
    # return no large arrays through the joblib result queue.
    with path.open('wb') as stream:
        for _, field in pickup_fields(mesh, source_rr_m, source_nn, coil_rr_m, coil_nn):
            np.ascontiguousarray(field.T, dtype=np.float64).tofile(stream)


def solve_parallel(mesh, source_rr_m, source_nn, coil_rr_m, coil_nn, channel_weights,
                   n_jobs, output_dir, *, pickup_fields, verbose=False):
    """Return the channel-by-source SI gain using one serial FEM driver per pickup group."""
    n_sources = len(source_rr_m)
    n_pickups = len(coil_rr_m)
    n_workers = min(n_jobs, n_pickups)
    # MPI-enabled DUNE installations require a matching mpi4py runtime. Mirror
    # the caller's initialization when run directly from Python or Jupyter.
    mpi = sys.modules.get('mpi4py.MPI')
    initialize_mpi = mpi is not None and mpi.Is_initialized()
    if initialize_mpi:
        launcher_ranks = ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK', 'MV2_COMM_WORLD_RANK')
        if mpi.Is_finalized() or mpi.COMM_WORLD.Get_size() > 1 or any(
                key in os.environ for key in launcher_ranks):
            raise ValueError("Start FEM workers from a standalone Python process with active "
                             "MPI, or run with n_jobs=1 in an MPI job.")
    # Use contiguous groups so both ends of every signed channel combination
    # can later be placed back in their exact original pickup order.
    edges = np.linspace(0, n_pickups, n_workers + 1, dtype=int)
    groups = [slice(begin, end) for begin, end in zip(edges[:-1], edges[1:])]

    with TemporaryDirectory(prefix='duneuro-', dir=output_dir) as scratch:
        files = [Path(scratch) / f'pickups-{group.start}.bin' for group in groups]
        # Loky launches fresh processes even in Jupyter. Large mesh and dipole
        # arrays are memory-mapped read-only; each solver owns its native grid,
        # preconditioner and its own transfer rows. Restrict nested BLAS threads.
        tasks = (
            delayed(_write_pickup_fields)(path, pickup_fields, mesh, source_rr_m, source_nn,
                                          coil_rr_m[group], coil_nn[group], initialize_mpi)
            for path, group in zip(files, groups)
        )
        if verbose:
            print(f'Solving {n_pickups:,} physical pickups for {n_sources:,} dipoles '
                  f'with {n_workers} DUNEuro workers.', flush=True)
        with parallel_backend('loky', inner_max_num_threads=1):
            Parallel(n_jobs=n_workers, temp_folder=scratch, mmap_mode='r',
                     max_nbytes='1M', verbose=10 if verbose else 0)(tasks)

        if verbose:
            print('Pickup fields complete. Applying channel weights.', flush=True)
        gain = np.empty((len(channel_weights), n_sources))
        # Read only one slab of sources at a time, then apply the full physical
        # channel montage (including reference compensation) exactly once.
        with ExitStack() as stack:
            readers = [stack.enter_context(path.open('rb')) for path in files]
            for start in range(0, n_sources, 256):
                stop = min(start + 256, n_sources)
                fields = np.empty((stop - start, n_pickups))
                for group, reader in zip(groups, readers):
                    width = group.stop - group.start
                    values = np.fromfile(reader, dtype=np.float64, count=(stop - start) * width)
                    fields[:, group] = values.reshape(stop - start, width)
                gain[:, start:stop] = channel_weights @ fields.T
    return gain
