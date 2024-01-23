import os
import json
import matlab.engine
import numpy as np
from pathlib import Path
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


def check_many(multiple, target, func=None):
    """
    Checks for a presence of strings in a target strings.
    
    Parameters:   
    multiple (list): strings to be found in target string
    target (str): target string
    func (str): "all" or "any", use the fuction to search for any or all strings in the filename.
    
    Notes:
    - this function works really well with if statement for list comprehension
    """
    func_dict = {
        "all": all, "any": any
    }
    if func in func_dict.keys():
        use_func = func_dict[func]
    elif func == None:
        raise ValueError("pick function 'all' or 'any'")    
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return use_func(check_)


def get_files(target_path, suffix, strings=[""], prefix=None, check="all", depth="all"):
    """
    Returns a list of the files with specific extension, prefix and name containing
    specific strings. Either all files in the directory or in this directory.
    
    Parameters:
    target path (str or pathlib.Path or os.Path): the most shallow searched directory
    suffix (str): file extension in "*.ext" format
    strings (list of str): list of strings searched in the file name
    prefix (str): limit the output list to the file manes starting with prefix
    check (str): "all" or "any", use the fuction to search for any or all strings in the filename.
    depth (str): "all" or "one", depth of search (recurrent or shallow)
    
    Notes:
    - returns a list of pathlib.Path objects
    """
    path = Path(target_path)
    if depth=="all":
        subdirs = [subdir for subdir in path.rglob(suffix) if check_many(strings, str(subdir.name), check)]
        subdirs.sort()
        if isinstance(prefix, str):
            subdirs = [subdir for subdir in subdirs if path.name.startswith(prefix)] 
        return subdirs
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir() if all([subdir.is_file(), subdir.suffix==suffix[1:], check_many(strings, str(subdir.name), check)])]
        if isinstance(prefix, str):
            subdirs = [subdir for subdir in subdirs if path.name.startswith(prefix)]
        subdirs.sort()
        return subdirs


def get_directories(target_path, strings=[""], check="all", depth="all"):
    """
    Returns a list of directories in the path (or all subdirectories) containing
    specified strings.
    
    Parameters:
    target path (str or pathlib.Path or os.Path): the most shallow searched directory
    depth (str): "all" or "one", depth of search (recurrent or shallow)
    
    Notes:
    - returns a list of pathlib.Path objects
    """
    path = Path(target_path)
    subdirs = []
    if depth == "all":
        subdirs = [subdir for subdir in path.glob("**/") if check_many(strings, str(subdir), check)]
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir() if subdir.is_dir() if check_many(strings, str(subdir), check)]
    subdirs.sort()
    return subdirs


def make_directory(root_path, extended_dir, check=False):
    """
    Creates a directory along with the intermediate directories.
    
    root_path (str or pathlib.Path or os.Path): the root directory
    extended_dir(str or list): directory or directories to create within root_path
    
    Notes:
    - can return a created path or a False if check=True
    """
    root_path = Path(root_path)
    if isinstance(extended_dir, list):
        root_path = root_path.joinpath(*extended_dir)
    else:
        root_path = root_path.joinpath(extended_dir)
    
    root_path.mkdir(parents=True, exist_ok=True)
    if all([check, root_path.exists()]):
        return root_path
    elif check:
        return root_path.exists()
