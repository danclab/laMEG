import os
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
        eng.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), './matlab'), nargout=0)
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
    settings_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
    with open(settings_fname) as settings_file:
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
        sensor_data, ch_names = eng.load_meg_sensor_data(
            data_fname,
            spm_path,
            nargout=2
        )

    return np.array(sensor_data), ch_names


def get_surface_names(n_layers, surf_path, orientation_method):
    """
    Generates a list of filenames for each mesh layer in a multi-layer mesh setup.

    Parameters:
    - n_layers (int): The number of layers in the mesh.
    - surf_path (str): The file path where the mesh files are located.
    - orientation_method (str): The method used for orientation in the naming of mesh files.

    Returns:
    - list: A list of strings, where each string is the full file path to a mesh layer file. The list
            order corresponds to the layers' order, starting from the outermost layer (pial surface)
            to the innermost layer (white matter surface).

    This function assumes a specific naming convention for the mesh files. The outermost layer is
    named as 'pial', the innermost as 'white', and the intermediate layers are named based on their
    relative position between 1 (pial) and 0 (white), with the position formatted to three decimal places.
    Each filename also includes an orientation method specifier.
    """
    # Get name of each mesh that makes up the layers of the multilayer mesh
    layers = np.linspace(1, 0, n_layers)
    layer_fnames = []
    for l, layer in enumerate(layers):
        if layer == 1:
            layer_fnames.append(os.path.join(surf_path, f'pial.ds.{orientation_method}.gii'))
        elif layer > 0 and layer < 1:
            layer_name = '{:.3f}'.format(layer)
            layer_fnames.append(os.path.join(surf_path, f'{layer_name}.ds.{orientation_method}.gii'))
        elif layer == 0:
            layer_fnames.append(os.path.join(surf_path, f'white.ds.{orientation_method}.gii'))
    return layer_fnames
