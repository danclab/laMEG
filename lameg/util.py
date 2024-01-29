import os
import json
import matlab.engine
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import nibabel as nib
from scipy.spatial import KDTree
from scipy.stats import t


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
    ndarray: An array containing the MEG data timestamps
    list: A list of channel names

    Notes:
    - The function requires MATLAB and DANC_SPM12 to be installed and accessible.
    - If `mat_eng` is not provided, the function will start a new MATLAB engine instance.
    - The function will automatically close the MATLAB engine if it was started within the function.
    """
    spm_path = get_spm_path()

    with matlab_context(mat_eng) as eng:
        sensor_data, time, ch_names = eng.load_meg_sensor_data(
            data_fname,
            spm_path,
            nargout=3
        )

    return np.array(sensor_data), np.squeeze(np.array(time)), ch_names


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


def fif_spm_conversion(mne_file, res4_file, output_path, output_name, prefix="spm_", epoched=None, create_path=False, mat_eng=None):
    """
    Converts *.fif file to SPM data format.
    
    Parameters:
    mne_file (str or pathlib.Path or os.Path): path to the "*-raw.fif" or "*-epo.fif" file
    res4_file (str or pathlib.Path or os.Path): location of the sensor position data. *.res4 for CTF
    output_path (str or pathlib.Path or os.Path): location of the converted file
    output_name (str): a core name of the output file, excluding the extension and path
    prefix (str): a string appended to the output_name after conversion. Default: "spm_"
    epoched (bool): Specify if the data is epoched (True) or not (False), default None will raise an error
    create_path (bool): if True create the non-existent subdirectories of the output path
    mat_eng (matlab.engine.MatlabEngine, optional): Instance of MATLAB engine. Default is None.
    
    Notes:
    """
    
    if epoched==None:
        raise ValueError("Please specify if the data is epoched (True) or not (False)")
    else:
        epoched = int(epoched)
    
    spm_path = get_spm_path()
    
    # clean things up for matlab
    mne_file = str(mne_file)
    output_path = str(output_path)
    output_name = str(output_name)
    res4_file = str(res4_file)
    
    if create_path:
        make_directory(output_path)
    
    with matlab_context(mat_eng) as eng:
        eng.convert_mne_to_spm(
            res4_file, mne_file, output_path,
            output_name, prefix, epoched, spm_path,
            nargout=0
        )


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


def check_maj(list_to_check):
    list_len = len(list_to_check)
    majority = list_len//2 + 1
    if len(set(list_to_check[:majority])) == 1:
        return list_to_check[0]
    else:
        item, count = np.unique(list_to_check, return_counts=True)
        return item[np.argmax(count)]
    

def transform_atlas(paths_annot, paths_fsavg_sphere, paths_fsnat_sphere, surface_path, surface_ds_path, surface_nodeep_path, return_dict=False):
    '''
    Transform fsaverage *.annot file atlas to vertex labels in the native space downsampled layer.
    
    Parameters:
    paths_annot [list of directories]: list of paths to Freesurfer atlas annotation files for left and right hemispheres
    paths_fsavg_sphere [list of directories]: list of paths to left and right FSAVERAGE registration spheres (in the asset folder)
    paths_fsnat_sphere [list of directories]: list of paths to left and right FSNATIVE registration spheres (individual for the subject)
    surface_path [directory]: path to a high resolution subject surface containing both hemispheres
    surface_ds_path [directory]: path to a downsampled subject surface containing both hemispheres
    surface_nodeep_path [directory]: path to a downsampled subject surface containing both hemispheres with corpus callosum removed
    
    Notes:
        - original atlas colours per vertex in format ready for trimesh,
        - labels per vertex
        - (optional) label to colour dictionary
    '''
    # combining annotation files
    annot_data_lr = [nib.freesurfer.io.read_annot(i) for i in paths_annot]
    annots_lr = {i: [] for i in ["label_ix", "color", "label", ]}
    for ix in range(len(annot_data_lr)):
        annots_lr["label_ix"].append(annot_data_lr[ix][0])
        annots_lr["color"].append(annot_data_lr[ix][1])
        annots_lr["label"].append(np.array(annot_data_lr[ix][2]))
    
    fsavg_spheres = [nib.load(i).agg_data()[0] for i in paths_fsavg_sphere]
    fsnat_spheres = [nib.load(i).agg_data()[0] for i in paths_fsnat_sphere]
    
    # mapping atlas labels between fsavereage and fsnative spheres
    nat_annots = []
    for lr in range(len(fsavg_spheres)):
        tree = KDTree(fsavg_spheres[lr], leaf_size=20)
        nat_dict_sub = {}
        for i in range(fsnat_spheres[lr].shape[0]):
            distance, index = tree.query([fsnat_spheres[lr][i]], k=5)
            distance = distance.flatten()
            index = index.flatten()
            label_indexes = annots_lr["label_ix"][lr][index].flatten()
            label_index = check_maj(label_indexes)
            label = annots_lr["label"][lr][label_index]
            nat_dict_sub[i] = [distance, index, label_indexes, label]
        nat_annots.append(nat_dict_sub)
    
    pial = nib.load(surface_path).agg_data()[0]
    pial_ds = nib.load(surface_ds_path).agg_data()[0]
    
    # mapping fsnative brain to downsampled brain
    pial_tree = KDTree(pial, leaf_size=10)
    pial_2_ds_map = []
    for i in range(pial_ds.shape[0]):
        dist, pial_index = pial_tree.query([pial_ds[i]], k=1)
        pial_2_ds_map.append(pial_index)
    pial_2_ds_map = np.array(pial_2_ds_map).flatten()
    
    # concatenating the annotations
    annots_order = np.array(
        [nat_annots[0][i][3] for i in nat_annots[0].keys()] + 
        [nat_annots[1][i][3] for i in nat_annots[1].keys()]
    )
    
    # selecting the annotation for the downsampled pial
    nat_pial_annot = annots_order[pial_2_ds_map]
    
    # removing the DEEP structures
    nodeep = nib.load(surface_nodeep_path).agg_data()[0]
    tree = KDTree(pial_ds, leaf_size=10)
    indices = [tree.query([nodeep[i]], k=1)[1].flatten()[0] for i in range(nodeep.shape[0])]
    nat_pial_annot = nat_pial_annot[indices]
    
    color = np.concatenate(annots_lr["color"])
    labels = np.concatenate(annots_lr["label"])
    
    lab_col_map = {lab: color[ix] for ix, lab in enumerate(labels)}
    mesh_colors = [lab_col_map[lab][:3].flatten() for lab in nat_pial_annot]
    if return_dict:
        return mesh_colors, nat_pial_annot, lab_col_map
    else:
        return mesh_colors, nat_pial_annot


def fsavg_vals_to_native(values, fsavg_sphere_paths, fsnat_sphere_paths, surface_path, surface_ds_path, surface_nodeep_path):
    """
    Transform values in fsaverage vertex order that contains values to vertex values in the native space downsampled.
    
    Parameters:
    paths_annot [list of directories]: list of paths to Freesurfer atlas annotation files for left and right hemispheres
    paths_fsavg_sphere [list of directories]: list of paths to left and right FSAVERAGE registration spheres (in the asset folder)
    paths_fsnat_sphere [list of directories]: list of paths to left and right FSNATIVE registration spheres (individual for the subject)
    surface_path [directory]: path to a high resolution subject surface containing both hemispheres
    surface_ds_path [directory]: path to a downsampled subject surface containing both hemispheres
    surface_nodeep_path [directory]: path to a downsampled subject surface containing both hemispheres with corpus callosum removed
    
    Notes:
        - values 
    """
    
    fsavg_spheres = [nib.load(i).agg_data()[0] for i in fsavg_sphere_paths]
    fsnat_spheres = [nib.load(i).agg_data()[0] for i in fsnat_sphere_paths]
    pial = nib.load(surface_path).agg_data()[0]
    pial_ds = nib.load(surface_ds_path).agg_data()[0]
    
    # values from fsaverage to fsnative
    fsnat_vx_values = []
    for lr in range(len(fsavg_spheres)):
        tree = KDTree(fsavg_spheres[lr], leaf_size=20)
        vx_value = []
        for xyz_ix in range(fsnat_spheres[lr].shape[0]):
            dist, vx_index = tree.query([fsnat_spheres[lr][xyz_ix]], k=1)
            vx_value.append(values[lr][vx_index].flatten())
        fsnat_vx_values.append(np.array(vx_value))
    
    fsnat_vx_values = np.concatenate(fsnat_vx_values)
    
    # mapping fsnative brain to downsampled brain
    pial_tree = KDTree(pial, leaf_size=10)
    pial_2_ds_map = []
    for i in range(pial_ds.shape[0]):
        dist, pial_index = pial_tree.query([pial_ds[i]], k=1)
        pial_2_ds_map.append(pial_index)
    pial_2_ds_map = np.array(pial_2_ds_map).flatten()
    
    # downsampled
    # but maybe mean of neighbouring vertices 
    fsnat_ds_vx_values = fsnat_vx_values[pial_2_ds_map]
    
    # removing deep structures
    nodeep = nib.load(surface_nodeep_path).agg_data()[0]
    tree = KDTree(pial_ds, leaf_size=10)
    indices = [tree.query([nodeep[i]], k=1)[1].flatten()[0] for i in range(nodeep.shape[0])]
    
    fsnat_ds_vx_values = fsnat_ds_vx_values[indices]
    
    return fsnat_ds_vx_values.flatten()



def ttest_rel_corrected(x, correction=0, tail=0, axis=0):
    """
    Perform a corrected paired t-test on a sample of data.

    This function handles missing data (NaNs) and applies a variance correction to the t-test calculation.
    It computes the t-statistic and corresponding p-value for the hypothesis test.

    Parameters:
    x (array_like): A 2-D array containing the sample data. NaN values are allowed and are handled appropriately.
    correction (float, optional): The correction value to be added to the variance to avoid division by zero issues.
                                  If set to 0 (default), an automatic correction of 0.01 * max(variance) is applied.
    tail (int, optional): Specifies the type of t-test to be performed.
                          0 for a two-tailed test (default), 1 for a right one-tailed test, -1 for a left one-tailed
                          test.
    axis (int, optional): Axis along which to perform the t-test. Default is 0.

    Returns:
    tuple: A tuple containing the t-statistic (float) and the p-value (float) for the test.

    Notes:
    - The function handles NaNs by computing the sample size, mean, and variance ignoring NaNs.
    - The degrees of freedom (df) for the t-test is computed as maximum(sample size - 1, 0).
    - The standard error of the mean (ser) is adjusted with the variance correction.
    - The p-value is computed based on the specified tail type of the t-test.
    """
    # Handle NaNs
    nans = np.isnan(x)
    if np.any(nans):
        samplesize = np.sum(~nans, axis=axis)
    else:
        samplesize = x.shape[axis]

    df = np.maximum(samplesize - 1, 0)
    xmean = np.nanmean(x, axis=axis)
    varpop = np.nanvar(x, axis=axis)

    # Apply correction
    if correction == 0:
        correction = 0.01 * np.max(varpop)

    corrsdpop = np.sqrt(varpop + correction)
    ser = corrsdpop / np.sqrt(samplesize)
    tval = (xmean - 0) / ser

    # Compute p-value
    p = np.inf
    if tail == 0:  # two-tailed test
        p = 2 * t.sf(np.abs(tval), df)
    elif tail == 1:  # right one-tailed test
        p = t.sf(-tval, df)
    elif tail == -1:  # left one-tailed test
        p = t.cdf(tval, df)

    return tval, p