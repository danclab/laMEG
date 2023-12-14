import matlab.engine
import numpy as np

from util import get_spm_path


def run_current_density_simulation(data_file, prefix, sim_vertices, sim_signals, dipole_moments, sim_patch_sizes, SNR,
                                   sim_woi=[-np.inf, np.inf], mat_eng=None):
    spm_path = get_spm_path()

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]
    sim_vertices=[x+1 for x in sim_vertices]
    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]
    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    close_matlab = False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab = True

    sim_fname=mat_eng.simulate(
        data_file,
        prefix,
        matlab.double(sim_vertices),
        matlab.double(sim_woi),
        matlab.double(sim_signals.tolist()),
        matlab.double([]),
        matlab.double(dipole_moments),
        matlab.double(sim_patch_sizes),
        float(SNR),
        spm_path,
        nargout=1
    )

    if close_matlab:
        mat_eng.close()

    return sim_fname


def run_dipole_simulation(data_file, prefix, sim_vertices, sim_signals, dipole_orientations, dipole_moments, sim_patch_sizes,
                          SNR, sim_woi=[-np.inf, np.inf], mat_eng=None):
    spm_path = get_spm_path()

    if np.isscalar(sim_vertices):
        sim_vertices=[sim_vertices]
    sim_vertices = [x + 1 for x in sim_vertices]
    if np.isscalar(dipole_moments):
        dipole_moments=[dipole_moments]
    if np.isscalar(sim_patch_sizes):
        sim_patch_sizes=[sim_patch_sizes]

    close_matlab = False
    if mat_eng is None:
        mat_eng = matlab.engine.start_matlab()
        close_matlab = True

    sim_fname=mat_eng.simulate(
        data_file,
        prefix,
        matlab.double(sim_vertices),
        matlab.double(sim_woi),
        matlab.double(sim_signals.tolist()),
        matlab.double(dipole_orientations.tolist()),
        matlab.double(dipole_moments),
        matlab.double(sim_patch_sizes),
        float(SNR),
        spm_path,
        nargout=1
    )

    if close_matlab:
        mat_eng.close()

    return sim_fname