import json

import os
import matlab.engine
import numpy as np

from util import get_spm_path
from surf import smoothmesh_multilayer_mm


def invert_ebb(out_dir, nas, lpa, rpa, mri_fname, mesh_fname, data_file, n_layers, patch_size=5,
               n_temp_modes=4, foi=[0, 256], woi=[-np.inf, np.inf], mat_eng=None):

    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    close_matlab=False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab=True

    mesh_base=os.path.split(os.path.splitext(mesh_fname)[0])[-1]
    data_fname=os.path.split(data_file)[-1]
    coreg_fname = os.path.join(out_dir, f'{mesh_base}.{data_fname}')

    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    F,MU=mat_eng.invert_ebb(data_file, coreg_fname, mri_fname, mesh_fname, matlab.double(nas), matlab.double(lpa),
                            matlab.double(rpa), float(patch_size), float(n_temp_modes), matlab.double(foi),
                            matlab.double(woi), spm_path, nargout=2)

    if close_matlab:
        mat_eng.close()

    return [coreg_fname, F, MU]


def invert_msp(out_dir, nas, lpa, rpa, mri_fname, mesh_fname, data_file, n_layers, priors=[], patch_size=5,
               n_temp_modes=4, foi=[0, 256], woi=[-np.inf, np.inf], mat_eng=None):

    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    close_matlab=False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab=True

    mesh_base=os.path.split(os.path.splitext(mesh_fname)[0])[-1]
    data_fname=os.path.split(data_file)[-1]
    coreg_fname = os.path.join(out_dir, f'{mesh_base}.{data_fname}')

    priors = [x + 1 for x in priors]
    if isinstance(woi, np.ndarray):
        woi = woi.tolist()

    F,MU=mat_eng.invert_msp(data_file, coreg_fname, mri_fname, mesh_fname, matlab.double(nas), matlab.double(lpa),
                            matlab.double(rpa), matlab.double(priors), float(patch_size), float(n_temp_modes),
                            matlab.double(foi), matlab.double(woi), spm_path, nargout=2)

    if close_matlab:
        mat_eng.close()

    return [coreg_fname, F, MU]


def invert_sliding_window(out_dir, prior, nas, lpa, rpa, mri_fname, mesh_fname, data_file, n_layers, patch_size=5,
                          n_temp_modes=4, win_size=10, win_overlap=True, foi=[0, 256], mat_eng=None):

    spm_path = get_spm_path()

    print(f'Smoothing {mesh_fname}')
    _ = smoothmesh_multilayer_mm(mesh_fname, patch_size, n_layers, n_jobs=-1)

    close_matlab = False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab = True

    prior = prior+1.0
    mesh_base = os.path.split(os.path.splitext(mesh_fname)[0])[-1]
    data_fname = os.path.split(data_file)[-1]
    coreg_fname = os.path.join(out_dir, f'{mesh_base}.{data_fname}')

    F,wois = mat_eng.invert_sliding_window(float(prior), data_file, coreg_fname, mri_fname, mesh_fname, matlab.double(nas),
                                           matlab.double(lpa), matlab.double(rpa), float(patch_size),
                                           float(n_temp_modes), float(win_size), win_overlap, matlab.double(foi),
                                           spm_path, nargout=2)

    if close_matlab:
        mat_eng.close()

    return [coreg_fname, F, wois]


def load_source_time_series(data_D, inv_D, vertices=[], mat_eng=None):

    spm_path = get_spm_path()

    close_matlab = False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab = True

    vertices = [x + 1 for x in vertices]
    source_ts = mat_eng.load_source_time_series(data_D, inv_D, matlab.double(vertices), spm_path, nargout=1)

    if close_matlab:
        mat_eng.close()

    return source_ts

