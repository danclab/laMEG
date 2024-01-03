import json
import matlab.engine
import numpy as np
from contextlib import contextmanager

@contextmanager
def matlab_context(eng=None):
    # Start MATLAB engine
    close_matlab=False
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.addpath('./matlab', nargout=0)
        close_matlab=True
    try:
        yield eng
    finally:
        # Close MATLAB engine
        if close_matlab:
            eng.quit()


def get_spm_path():
    with open('settings.json') as settings_file:
        parameters = json.load(settings_file)
    spm_path = parameters["spm_path"]
    return spm_path


def load_meg_sensor_data(data_D, mat_eng=None):
    """
    Load sensor data from a MEG dataset.

    Parameters:
    data_D (str): Filename or path of the MEG/EEG data file.
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.

    Returns:
    ndarray: An array containing the MEG sensor data (channels x time x trial).

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    spm_path = get_spm_path()

    with matlab_context(mat_eng) as eng:
        sensor_data = eng.load_meg_sensor_data(
            data_D,
            spm_path,
            nargout=1
        )

    return np.array(sensor_data)