import json
import matlab.engine
import numpy as np
from contextlib import contextmanager


@contextmanager
def matlab_context(eng=None):
    """
    Context manager for handling MATLAB engine instances.

    Parameters:
    eng (matlab.engine.MatlabEngine, optional): An existing MATLAB engine instance. Default is None.

    Yields:
    matlab.engine.MatlabEngine: A MATLAB engine instance for use within the context.

    Notes:
    - If 'eng' is None, the function starts a new MATLAB engine and adds './matlab' to its path.
    - The new MATLAB engine instance will be closed automatically upon exiting the context.
    - If 'eng' is provided, it will be used as is and not closed automatically.
    - This function is intended for use in a 'with' statement to ensure proper management of MATLAB engine resources.
    """
    # Start MATLAB engine
    close_matlab = False
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.addpath('./matlab', nargout=0)
        close_matlab = True
    try:
        yield eng
    finally:
        # Close MATLAB engine
        if close_matlab:
            eng.quit()


def get_spm_path():
    """
    Retrieve the path to the SPM (Statistical Parametric Mapping) software from a settings file.

    Returns:
    str: The path to the SPM software.

    Notes:
    - The function reads from a 'settings.json' file expected to be in the current working directory.
    - The 'settings.json' file must contain a key "spm_path" with the path to the SPM software as its value.
    - It is assumed that the 'settings.json' file is properly formatted and accessible.
    - If the 'settings.json' file or the "spm_path" key is missing, the function will raise an error.
    """
    with open('settings.json') as settings_file:
        parameters = json.load(settings_file)
    spm_path = parameters["spm_path"]
    return spm_path


def load_meg_sensor_data(data_fname, mat_eng=None):
    """
    Load sensor data from a MEG dataset.

    Parameters:
    data_fname (str): Filename or path of the MEG/EEG data file.
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
            data_fname,
            spm_path,
            nargout=1
        )

    return np.array(sensor_data)
