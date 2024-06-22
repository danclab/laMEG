import json
import os
import tempfile

import numpy as np
import h5py
from pathlib import Path
from contextlib import contextmanager
import nibabel as nib
from scipy.io import savemat
from scipy.spatial import KDTree
from scipy.stats import t
import spm_standalone


@contextmanager
def spm_context(spm=None):
    """
    Context manager for handling standalone SPM instances.

    Parameters:
    spm (spm_standalone, optional): An existing standalone instance. Default is None.

    Yields:
    spm_standalone: A standalone SPM instance for use within the context.

    Notes:
    - If 'spm' is None, the function starts a new standalone SPM instance.
    - The new standalone SPM instance will be closed automatically upon exiting the context.
    - If 'spm' is provided, it will be used as is and not closed automatically.
    - This function is intended for use in a 'with' statement to ensure proper management of standalone SPM resources.
    """
    # Start standalone SPM
    settings_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
    with open(settings_fname) as settings_file:
        parameters = json.load(settings_file)

    close_spm = False
    if spm is None:
        spm = spm_standalone.initialize()
        spm.spm_standalone(
            "eval",
            f"""
            try
                parpool({parameters['num_workers']});
            catch ME
            end
            """,
            nargout=0
        )
        close_spm = True
    try:
        yield spm
    finally:
        # Close standalone SPM
        if close_spm:
            spm.spm_standalone(
                "eval",
                "delete(gcp('nocreate'));",
                nargout=0
            )
            spm.terminate()
            del spm


def batch(cfg, viz=True, spm_instance=None):
    cfg = {"matlabbatch": [cfg]}
    f, name = tempfile.mkstemp(suffix=".mat")
    savemat(f, cfg)

    with spm_context(spm_instance) as spm:
        spm.spm_standalone(
            "eval",
            f"load('{name}'); spm('defaults', 'EEG'); spm_get_defaults('cmdline',{int(not viz)}); spm_jobman('run', matlabbatch);",
            nargout=0
        )
    os.remove(name)


def get_assets():
    current_path = Path(os.getcwd())
    asset_path = current_path.joinpath("assets", "big_brain_layer_thickness")
    get_files(asset_path, "*.gii", strings=["tpl-fsaverage"])


def load_meg_sensor_data(data_fname):
    """
    Load sensor data from a MEG dataset.

    Parameters:
    data_fname (str): Filename or path of the MEG/EEG data file.

    Returns:
    ndarray: An array containing the MEG sensor data (channels x time x trial).
    ndarray: An array containing the MEG data timestamps
    list: A list of channel names
    """

    good_meg_channels = []  # List to store the indices of good MEG channels
    ch_names = []

    with h5py.File(data_fname, 'r') as file:
        time_onset = file['D']['timeOnset'][()][0, 0]
        fs = file['D']['Fsample'][()][0, 0]
        n_samples = int(file['D']['Nsamples'][()][0, 0])
        n_chans = file['D']['channels']['type'][:].shape[0]
        chan_bad = np.array([int(file[file['D']['channels']['bad'][:][i, 0]][()][0, 0]) for i in range(n_chans)])

        for i in range(n_chans):
            chan_type_data = file[file['D']['channels']['type'][:][i, 0]][()]
            chan_type = ''.join(chr(code) for code in chan_type_data)

            chan_name_data = file[file['D']['channels']['label'][:][i, 0]][()]
            chan_name = ''.join(chr(code) for code in chan_name_data)

            # Check if the channel type includes 'MEG' and the channel is not marked as bad
            if 'MEG' in chan_type and chan_bad[i] == 0:
                good_meg_channels.append(i)  # Add the channel index to the list
                ch_names.append(chan_name)

        # Access dimensions, convert to tuple of ints assuming it's 2D and transpose if necessary
        data_dims = tuple(int(dim) for dim in file['D']['data']['dim'][:, 0])

        # Get filename stored as integers and convert to string
        data_fname_data = file['D']['data']['fname'][:]
        data_filename = ''.join(chr(code) for code in data_fname_data)

    good_meg_channels = np.array(good_meg_channels)

    # Read binary data using extracted filename
    with open(data_filename, 'rb') as file:
        dtype = np.dtype('<f4')
        # Read the data assuming type float32; adjust dtype if necessary
        data_array = np.fromfile(file, dtype=dtype)

        # Reshape data according to extracted dimensions
        sensor_data = data_array.reshape(data_dims, order='F')[good_meg_channels, :]

    time = np.arange(n_samples) / fs + time_onset

    return sensor_data, time, ch_names


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


def fif_spm_conversion(mne_file, res4_file, output_path, prefix="spm_", epoched=None, create_path=False,
                       spm_instance=None):
    """
    Converts *.fif file to SPM data format.

    Parameters:
    mne_file (str or pathlib.Path or os.Path): path to the "*-raw.fif" or "*-epo.fif" file
    res4_file (str or pathlib.Path or os.Path): location of the sensor position data. *.res4 for CTF
    output_path (str or pathlib.Path or os.Path): location of the converted file
    prefix (str): a string appended to the output_name after conversion. Default: "spm_"
    epoched (bool): Specify if the data is epoched (True) or not (False), default None will raise an error
    create_path (bool): if True create the non-existent subdirectories of the output path
    spm_instance (spm_standalone, optional): Instance of standalone SPM. Default is None.

    Notes:
        - If `spm_instance` is not provided, the function will start a new standalone SPM instance.
        - The function will automatically close the standalone SPM instance if it was started within the function.
    """

    if epoched == None:
        raise ValueError("Please specify if the data is epoched (True) or not (False)")
    else:
        epoched = int(epoched)

    # clean things up for matlab
    mne_file = str(mne_file)
    output_path = str(output_path)
    res4_file = str(res4_file)

    if create_path:
        make_directory(output_path)

    with spm_context(spm_instance) as spm:
        spm.convert_mne_to_spm(
                res4_file, mne_file, output_path,
                prefix, float(epoched), nargout=0
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
    if depth == "all":
        subdirs = [subdir for subdir in path.rglob(suffix) if check_many(strings, str(subdir.name), check)]
        subdirs.sort()
        if isinstance(prefix, str):
            subdirs = [subdir for subdir in subdirs if path.name.startswith(prefix)]
        return subdirs
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir() if
                   all([subdir.is_file(), subdir.suffix == suffix[1:], check_many(strings, str(subdir.name), check)])]
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
    majority = list_len // 2 + 1
    if len(set(list_to_check[:majority])) == 1:
        return list_to_check[0]
    else:
        item, count = np.unique(list_to_check, return_counts=True)
        return item[np.argmax(count)]


def convert_fsaverage_to_native(subj_id, hemi, vert_idx):
    """
    Convert a vertex index from fsaverage to a subject's native surface space.

    This function takes a vertex index from the fsaverage template surface and finds the corresponding
    vertex index in a subject's native surface space. It loads the fsaverage spherical surface, identifies the
    coordinates of the given vertex index, and then finds the nearest corresponding vertex on the subject's
    registered spherical surface. If the hemisphere is right, it adjusts the index by adding the number of vertices
    in the left hemisphere pial surface so that it matches the combined hemishere mesh. It returns the adjusted vertex
    index in the subject's native space.

    Parameters:
    subj_id (str): The subject identifier for which the conversion is being performed.
    hemi (str): Hemisphere specifier ('lh' for left hemisphere, 'rh' for right hemisphere).
    vert_idx (int): Index of the vertex in the fsaverage surface to be converted.

    Returns:
    int: Index of the vertex on the subject's native surface that corresponds to the input vertex index.
    """

    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    # Load fsaverage sphere
    fsaverage_sphere_vertices, fsaverage_sphere_faces = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', f'{hemi}.sphere.reg')
    )

    # Get the coordinate of the corresponding vertex on the fsaverage sphere
    fsave_sphere_coord = fsaverage_sphere_vertices[vert_idx, :]

    # Load subject registered sphere
    subj_sphere = nib.load(os.path.join(fs_subject_dir, 'surf', f'{hemi}.sphere.reg'))

    # Get the index of the nearest vertex on the subject sphere
    kdtree = KDTree(subj_sphere.darrays[0].data)
    dist, subj_v_idx = kdtree.query(fsave_sphere_coord, k=1)

    # Adjust vertex index for right hemishphere
    if hemi=='rh':
        lh_vertices, lh_faces = nib.freesurfer.read_geometry(os.path.join(fs_subject_dir, 'surf', 'lh.pial'))
        subj_v_idx += lh_vertices.shape[0]

    return subj_v_idx


def convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord):
    """
    Convert coordinates from a subject's native surface space to the fsaverage surface space.

    This function maps a vertex coordinate from a subject's native combined pial surface to the corresponding
    vertex index in the fsaverage template space. It does this by determining which hemisphere the
    vertex belongs to based on the closest match in the left and right hemispheres' pial surfaces.
    It then finds the nearest vertex in the subject's registered spherical surface, maps this to the
    nearest vertex in the fsaverage spherical surface, and returns the index of this fsaverage vertex.

    Parameters:
    subj_id (str): The subject identifier for which the conversion is being performed.
    subj_surf_dir (str): The path containing the laMEG-processed subject surfaces
    subj_coord (array-like): The x, y, z coordinates on the subject's combined hemisphere pial surface to be converted.

    Returns:
    str: The hemisphere the vertex is found in ('lh' for left hemisphere, 'rh' for right hemisphere).
    int: Index of the vertex on the fsaverage spherical surface that corresponds to the input coordinates.
    """
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    # Figure out hemisphere
    subj_lh = nib.load(os.path.join(subj_surf_dir, 'lh.pial.gii'))
    lh_vertices = subj_lh.darrays[0].data
    subj_rh = nib.load(os.path.join(subj_surf_dir, 'rh.pial.gii'))
    rh_vertices = subj_rh.darrays[0].data

    kdtree = KDTree(lh_vertices)
    lh_dist, lh_vert_idx = kdtree.query(subj_coord, k=1)

    kdtree = KDTree(rh_vertices)
    rh_dist, rh_vert_idx = kdtree.query(subj_coord, k=1)

    if lh_dist < rh_dist:
        hemi = 'lh'
        vert_idx = lh_vert_idx
    else:
        hemi = 'rh'
        vert_idx = rh_vert_idx

    subj_sphere_vertices, subj_faces = nib.freesurfer.read_geometry(os.path.join(fs_subject_dir, 'surf', f'{hemi}.sphere.reg'))

    # Get the coordinate of the corresponding vertex on the registered subject sphere
    subj_sphere_coord = subj_sphere_vertices[vert_idx, :]

    # Load FS-average sphere
    fsaverage_sphere_vertices, fsaverage_sphere_faces = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', f'{hemi}.sphere.reg')
    )

    # Get the index of the nearest vertex on the fsaverage sphere
    kdtree = KDTree(fsaverage_sphere_vertices)
    dist, fsave_v_idx = kdtree.query(subj_sphere_coord, k=1)

    return hemi, fsave_v_idx


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


def calc_prop(x):
    """
    from independent thickness to overall proportion, while respecting the zeros
    """
    sum_ = np.sum(x)
    if sum_ == 0.0:
        return x
    else:
        x = np.cumsum(x) / sum_
        return x
    

def big_brain_proportional_layer_boundaries(overwrite=False):
    """
    Get the proportional layer boundaries (6 values between 0 and 1) from 
    fsaverage converted Big Brain atlas, included in the laMEG.
    
    Function uses the included fsaverage converted Big Brain cortical thickness atlas to calculate
    normalised distances between cortical layer (from layer 1 to layer 6) boundaries (values between 0 and 1).
    To speed up the computation, the results are stored in the numpy dictionary.
    
    Parameters:
    overwrite (bool): overwrite the existing file
    
    Returns:
    bb_data (dict): dictionary (keys: "lh", "rh") with arrays containing layer boundaries for each vertex in the hemisphere
    
    """

    asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './assets')

    BBL_file = os.path.join(asset_path, "proportional_layer_boundaries.npy")
    if any([not os.path.exists(BBL_file), overwrite]):
        bb_l_paths = get_files(asset_path, "*.gii", strings=["tpl-fsaverage", "hemi-L"])
        bb_l_paths.sort()
        bb_r_paths = get_files(asset_path, "*.gii", strings=["tpl-fsaverage", "hemi-R"])
        bb_r_paths.sort()
        bb_data = {
            "lh": np.array([nib.load(p).agg_data() for p in bb_l_paths]),
            "rh": np.array([nib.load(p).agg_data() for p in bb_r_paths])
        }
        
        bb_data = {
           k: np.apply_along_axis(calc_prop, 0, bb_data[k]) for k in bb_data.keys()
        }
        np.save(BBL_file, bb_data)
        return bb_data
    
    else:
        bb_data = np.load(BBL_file, allow_pickle=True).item()
        return bb_data


def get_BB_layer_boundaries(subj_id, subj_surf_dir, subj_coord):
    """
    Get the cortical layer boundaries based on Big Brain atlas for a specified coordinate
    in the subject's downsampled combined space.
    
    Function maps a vertex coordinate from a subject's native combined pial surface to the corresponding
    vertex index in the fsaverage template space, in a specific hemisphere. Then, the proportional layer
    boundaries (6 values between 0 and 1) from fsaverage converted Big Brain atlas are returned (from layer 1 to layer 6).
    To get the subjects prpportional values, those values have to be multiplied by the observed cortical thickness.
    
    Parameters:
    subj_id (str): The subject identifier for which the conversion is being performed.
    subj_surf_dir (str): The path containing the laMEG-processed subject surfaces
    subj_coord (array-like): The x, y, z coordinates on the subject's combined hemisphere pial surface to be converted.
    
    Returns:
    vert_bb_prop (array-like): proportional layer boundaries (6 values between 0 and 1) from fsaverage converted Big Brain atlas

    """
    # convert subj_coord to native + hemisphere
    hemi, fsave_v_idx = convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord)
    
    # compute or read (if the precomputed atlas is present)
    bb_prop = big_brain_proportional_layer_boundaries()
    
    # get the layer boundaries from the fsaverage vertex
    vert_bb_prop = bb_prop[hemi][:,fsave_v_idx]
    
    return vert_bb_prop
