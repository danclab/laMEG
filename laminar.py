import numpy as np

from invert import invert_ebb, invert_msp, invert_sliding_window


def model_comparison(out_dir, nas, lpa, rpa, mri_fname, mesh_fnames, sim_fname, patch_size=5, n_temp_modes=4,
                     foi=[0, 256], woi=[-np.inf, np.inf], method='EBB', priors=[], mat_eng=None):
    f_vals=[]
    for mesh_fname in mesh_fnames:
        if method=='EBB':
            [_, f_val, _] = invert_ebb(out_dir, nas, lpa, rpa, mri_fname, mesh_fname, sim_fname, 1,
                                       patch_size=patch_size, n_temp_modes=n_temp_modes, foi=foi, woi=woi,
                                       mat_eng=mat_eng)
        elif method=='MSP':
            [_, f_val, _] = invert_msp(out_dir, nas, lpa, rpa, mri_fname, mesh_fname, sim_fname, 1,
                                       priors=priors, patch_size=patch_size, n_temp_modes=n_temp_modes, foi=foi,
                                       woi=woi, mat_eng=mat_eng)
        f_vals.append(f_val)
    return f_vals

def sliding_window_model_comparison(out_dir, prior, nas, lpa, rpa, mri_fname, mesh_fnames, data_file, patch_size=5,
                                    n_temp_modes=4, win_size=10, win_overlap=True, mat_eng=None):
    f_vals=[]
    for mesh_fname in mesh_fnames:
        [_, mesh_fvals, wois] = invert_sliding_window(out_dir, prior, nas, lpa, rpa, mri_fname, mesh_fname, data_file,
                                                      1, patch_size=patch_size, n_temp_modes=n_temp_modes,
                                                      win_size=win_size, win_overlap=win_overlap, mat_eng=mat_eng)
        f_vals.append(mesh_fvals)
    return f_vals, wois