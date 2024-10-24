"""
This module provides tools for interfacing with SPM (Statistical Parametric Mapping) software,
managing MEG sensor data, and working with neuroimaging data in various formats such as NIfTI,
GIFTI, and MATLAB files. It includes functions for batch processing in SPM, converting data
formats, loading and processing sensor data, and working with anatomical structures through
cortical mesh analysis.

Key functionalities:

- Context management for SPM operations.
- Batch processing for neuroimaging tasks.
- Loading MEG sensor data and managing related file operations.
- Utility functions for anatomical and spatial data transformations.
"""

import csv
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import h5py
import nibabel as nib
from scipy.io import savemat, loadmat
from scipy.spatial import KDTree
from scipy.stats import t
import spm_standalone


@contextmanager
def spm_context(spm=None, n_jobs=4):
    """
    Context manager for handling standalone SPM instances.

    Parameters
    ----------
    spm : spm_standalone, optional
        An existing standalone instance. Default is None.
    n_jobs : int
        A number of workers in a MATLAB parpool. Default is 4.

    Yields
    ------
    spm : spm_standalone
        A standalone SPM instance for use within the context.

    Notes
    -----
    - If `spm` is None, the function starts a new standalone SPM instance.
    - The new standalone SPM instance will be closed automatically upon exiting the context.
    - If `spm` is provided, it will be used as is and not closed automatically.
    - This function is intended for use in a `with` statement to ensure proper management of
      standalone SPM resources.
    - Default `n_jobs=4` is suitable for a workstation. Increasing the amount of available workers
      is a good choice for deploying with HPC.
    """

    # Start standalone SPM
    close_spm = False
    if spm is None:
        spm = spm_standalone.initialize()
        spm.spm_standalone(
            "eval",
            f"""
            try
                parpool({n_jobs});
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


def batch(cfg, viz=True, spm_instance=None) -> None:
    """
    Execute a batch processing job in SPM (Statistical Parametric Mapping) using MATLAB.

    This function prepares a configuration for an SPM batch job, saves it to a temporary MATLAB
    file, and executes it within an SPM instance. The function is capable of running any batch
    configuration passed to it, as long as it adheres to SPM's batch configuration structure.
    After processing, it cleans up by deleting the temporary file used for the job.

    Parameters
    ----------
    cfg : dict
        A dictionary containing the configuration settings for the SPM job. The dictionary should
        follow the structure required by SPM's `matlabbatch` system.
    viz : bool, optional
        If True, the SPM GUI will display progress and results, allowing user interaction. If
        False, the process runs entirely in the background. Defaults to True.
    spm_instance : optional
        An instance of an SPM session. If None, a new SPM session is created and used for the job.
        Defaults to None.

    Examples
    --------
    To run an SPM job with a given configuration, you might call the function as follows:

    >>> cfg = {
    >>>     'spm.stats.fmri_spec.dir': ['/path/to/output'],
    >>>     'spm.stats.fmri_spec.timing.units': 'secs',
    >>>     'spm.stats.fmri_spec.sess': {
    >>>         'scans': ['scan1.nii', 'scan2.nii'],
    >>>         'cond': {
    >>>             'name': 'ExampleCondition',
    >>>             'onset': [10, 30],
    >>>             'duration': [1, 1],
    >>>         },
    >>>         'multi': {'regress': {'name': 'movement', 'val': [1, 0, 1, 0]}}
    >>>     }
    >>> }
    >>> batch(cfg, viz=False)

    Notes
    -----
    - The temporary MATLAB file is created in the system's default temp directory.
    - This function assumes that an SPM and MATLAB environment is properly set up and accessible
      through the provided `spm_instance` or through a default SPM environment.
    """

    cfg = {"matlabbatch": [cfg]}
    file, name = tempfile.mkstemp(suffix=".mat")
    savemat(file, cfg)

    with spm_context(spm_instance) as spm:
        spm.spm_standalone(
            "eval",
            f"""
            load('{name}'); 
            spm('defaults', 'EEG');
            spm_get_defaults('cmdline',{int(not viz)}); 
            spm_jobman('run', matlabbatch);
            """,
            nargout=0
        )
    os.remove(name)


def load_meg_sensor_data(data_fname):
    """
    Load sensor data from a MEG dataset.

    Parameters
    ----------
    data_fname : str
        Filename or path of the MEG data file.

    Returns
    -------
    sensor_data : ndarray
        An array containing the MEG sensor data (channels x time x trial).
    time : ndarray
        An array containing the MEG data timestamps.
    ch_names : list
        A list of channel names.
    """

    good_meg_channels = []  # List to store the indices of good MEG channels
    ch_names = []

    try:
        with h5py.File(data_fname, 'r') as file:
            time_onset = file['D']['timeOnset'][()][0, 0]
            fsample = file['D']['Fsample'][()][0, 0]
            n_samples = int(file['D']['Nsamples'][()][0, 0])
            n_chans = file['D']['channels']['type'][:].shape[0]
            chan_bad = np.array([int(file[file['D']['channels']['bad'][:][i, 0]][()][0, 0])
                                 for i in range(n_chans)])

            for i in range(n_chans):
                chan_type_data = file[file['D']['channels']['type'][:][i, 0]][()]
                chan_type = ''.join(chr(code) for code in chan_type_data)

                chan_name_data = file[file['D']['channels']['label'][:][i, 0]][()]
                chan_name = ''.join(chr(code) for code in chan_name_data)

                # Check if the channel type includes 'MEG' and the channel is not marked as bad
                if 'MEG' in chan_type and chan_bad[i] == 0:
                    good_meg_channels.append(i)  # Add the channel index to the list
                    ch_names.append(chan_name)

            # Access dimensions, convert to tuple of ints assuming it's 2D and transpose if
            # necessary
            data_dims = tuple(int(dim) for dim in file['D']['data']['dim'][:, 0])

            # Get filename stored as integers and convert to string
            data_fname_data = file['D']['data']['fname'][:]
            data_filename = ''.join(chr(code) for code in data_fname_data)
    except OSError:
        mat_contents = loadmat(data_fname)
        # Assuming these fields exist; update these as per actual .mat file structure
        time_onset = mat_contents['D'][0][0]['timeOnset'][0,0]
        fsample = mat_contents['D'][0][0]['Fsample'][0,0]
        n_samples = mat_contents['D'][0][0]['Nsamples'][0,0]
        n_chans = mat_contents['D'][0][0]['channels'].shape[1]
        chan_bad = np.array([int(channel[0][0])
                             for channel in mat_contents['D'][0][0]['channels'][:]['bad'][0, :]])

        for i in range(n_chans):
            chan_type = mat_contents['D'][0][0]['channels'][:]['type'][0,i][0]
            chan_name = mat_contents['D'][0][0]['channels'][:]['label'][0,i][0]
            # Check if the channel type includes 'MEG' and the channel is not marked as bad
            if 'MEG' in chan_type and chan_bad[i] == 0:
                good_meg_channels.append(i)  # Add the channel index to the list
                ch_names.append(chan_name)

        data_dims = tuple(int(dim) for dim in mat_contents['D'][0][0]['data'][:]['dim'][:,0][0][0])
        data_filename = mat_contents['D'][0][0]['data'][:]['fname'][:,0][0][0]

    # Determine the directory of the data_fname
    data_dir = os.path.dirname(data_fname)
    full_data_path = os.path.join(data_dir, os.path.split(data_filename)[-1])

    # Check if the data file exists at the path specified in the MAT file
    if not os.path.exists(data_filename):
        # If the file isn't found at the specified path, check the same directory as the .mat file
        data_filename = full_data_path

    # Confirm the existence of the file before attempting to open it
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"Data file not found in specified location: "
                                f"{data_filename}")

    good_meg_channels = np.array(good_meg_channels)

    # Read binary data using extracted filename
    with open(data_filename, 'rb') as file:
        dtype = np.dtype('<f4')
        # Read the data assuming type float32; adjust dtype if necessary
        data_array = np.fromfile(file, dtype=dtype)

        # Reshape data according to extracted dimensions
        sensor_data = data_array.reshape(data_dims, order='F')[good_meg_channels, :]

    time = np.arange(n_samples) / fsample + time_onset

    return sensor_data, time, ch_names


def get_surface_names(n_layers, surf_path, orientation_method):
    """
    Generate a list of filenames for each mesh layer in a multi-layer mesh setup.

    Parameters
    ----------
    n_layers : int
        The number of layers in the mesh.
    surf_path : str
        The file path where the mesh files are located.
    orientation_method : str
        The method used for orientation in the naming of mesh files.

    Returns
    -------
    layer_fnames : list
        A list of strings, where each string is the full file path to a mesh layer file. The list
        order corresponds to the layers' order, starting from the outermost layer (pial surface)
        to the innermost layer (white matter surface).

    Notes
    -----
    This function assumes a specific naming convention for the mesh files. The outermost layer is
    named as 'pial', the innermost as 'white', and the intermediate layers are named based on their
    relative position between 1 (pial) and 0 (white), with the position formatted to three decimal
    places. Each filename also includes an orientation method specifier.
    """

    # Get name of each mesh that makes up the layers of the multilayer mesh
    layers = np.linspace(1, 0, n_layers)
    layer_fnames = []
    for layer in layers:

        if layer == 1:
            fname = os.path.join(surf_path, f'pial.ds.{orientation_method}.gii')
        elif layer == 0:
            fname = os.path.join(surf_path, f'white.ds.{orientation_method}.gii')
        else:
            fname = os.path.join(surf_path, f'{layer:.3f}.ds.{orientation_method}.gii')

        if fname is not None and os.path.exists(fname):
            layer_fnames.append(fname)
        else:
            raise FileNotFoundError(f"Unable to locate {fname}. Check surf_path")

    return layer_fnames


def ctf_fif_spm_conversion(mne_file, res4_file, output_path, epoched, prefix="spm_",
                           spm_instance=None):
    """
    Convert a \*.fif file containing data from a CTF scanner to SPM data format.

    Parameters
    ----------
    mne_file : str or pathlib.Path or os.Path
        Path to the "\*-raw.fif" or "\*-epo.fif" file.
    res4_file : str or pathlib.Path or os.Path
        Location of the sensor position data (\*.res4 for CTF).
    output_path : str or pathlib.Path or os.Path
        Location of the converted file.
    epoched : bool
        Specify if the data is epoched (True) or not (False).
    prefix : str
        A string appended to the output file name after conversion. Default: ``"spm_"``.
    spm_instance : spm_standalone, optional
        Instance of standalone SPM. Default is None.

    Notes
    -----
    - If ``spm_instance`` is not provided, the function will start a new standalone SPM instance.
    - The function will automatically close the standalone SPM instance if it was started
      within the function.
    """

    epoched = int(epoched)

    # clean things up for matlab
    mne_file = str(mne_file)
    output_path = str(output_path)
    res4_file = str(res4_file)

    with spm_context(spm_instance) as spm:
        spm.convert_ctf_fif_to_spm(
                res4_file, mne_file, output_path,
                prefix, float(epoched), nargout=0
            )


def check_many(multiple, target, func=None):
    """
    Check for the presence of strings in a target string.

    Parameters
    ----------
    multiple : list
        List of strings to be found in the target string.
    target : str
        The target string in which to search for the specified strings.
    func : str
        Specifies the search mode: "all" to check if all strings are present, or "any" to check if
        any string is present.

    Notes
    -----
    - This function works well with `if` statements in list comprehensions.
    """

    func_dict = {
        "all": all, "any": any
    }
    if func in func_dict:
        use_func = func_dict[func]
    else:
        raise ValueError("pick function 'all' or 'any'")
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return use_func(check_)


def get_files(target_path, suffix, strings=(""), prefix=None, check="all", depth="all"):
    """
    Return a list of files with a specific extension, prefix, and name containing specific strings.

    Searches either all files in the target directory or within a specified directory.

    Parameters
    ----------
    target_path : str or pathlib.Path or os.Path
        The most shallow searched directory.
    suffix : str
        File extension in "\*.ext" format.
    strings : list of str
        List of strings to search for in the file name.
    prefix : str
        Limits the output list to file names starting with this prefix.
    check : str
        Specifies the search mode: "all" to check if all strings are present, or "any" to check if
        any string is present.
    depth : str
        Specifies the depth of the search: "all" for recursive search, "one" for shallow search.

    Returns
    -------
    subdirs : list
        List of pathlib.Path objects representing the found files.
    """

    path = Path(target_path)
    files = []
    if depth == "all":
        files = [file for file in path.rglob(suffix)
                 if file.is_file() and file.suffix == suffix[1:] and
                 check_many(strings, file.name, check)]
    elif depth == "one":
        files = [file for file in path.iterdir()
                 if file.is_file() and file.suffix == suffix[1:] and
                 check_many(strings, file.name, check)]

    if isinstance(prefix, str):
        files = [file for file in files if file.name.startswith(prefix)]
    files.sort(key=lambda x: x.name)
    return files


def get_directories(target_path, strings=(""), check="all", depth="all"):
    """
    Return a list of directories in the path (or all subdirectories) containing specified strings.

    Parameters
    ----------
    target_path : str or pathlib.Path or os.Path
        The most shallow searched directory.
    depth : str
        Specifies the depth of the search: "all" for recursive search, "one" for shallow search.

    Returns
    -------
    subdirs : list
        List of pathlib.Path objects representing the found directories.
    """

    path = Path(target_path)
    subdirs = []
    if depth == "all":
        subdirs = [subdir for subdir in path.glob("**/")
                   if subdir.is_dir() and check_many(strings, str(subdir), check)]
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir()
                   if subdir.is_dir() and check_many(strings, str(subdir), check)]
    # pylint: disable=unnecessary-lambda
    subdirs.sort(key=lambda x: str(x))
    return subdirs


def make_directory(root_path, extended_dir):
    """
    Create a directory along with intermediate directories.

    Parameters
    ----------
    root_path : str or pathlib.Path or os.Path
        The root directory.
    extended_dir : str or list
        Directory or directories to create within `root_path`.

    Returns
    -------
    root_path : str or pathlib.Path or os.Path
        The updated root directory.
    """

    root_path = Path(root_path)
    if isinstance(extended_dir, list):
        root_path = root_path.joinpath(*extended_dir)
    else:
        root_path = root_path.joinpath(extended_dir)

    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def convert_fsaverage_to_native(subj_id, hemi, vert_idx):
    """
    Convert a vertex index from fsaverage to a subject's native surface space.

    This function takes a vertex index from the fsaverage template surface and finds the
    corresponding vertex index in a subject's native surface space. It loads the fsaverage
    spherical surface, identifies the coordinates of the given vertex index, and then finds the
    nearest corresponding vertex on the subject's registered spherical surface. If the hemisphere
    is right, it adjusts the index by adding the number of vertices in the left hemisphere pial
    surface so that it matches the combined hemisphere mesh. It returns the adjusted vertex index in
    the subject's native space.

    Parameters
    ----------
    subj_id : str
        The subject identifier for which the conversion is being performed.
    hemi : str
        Hemisphere specifier ('lh' for left hemisphere, 'rh' for right hemisphere).
    vert_idx : int
        Index of the vertex in the fsaverage surface to be converted.

    Returns
    -------
    subj_v_idx : int
        Index of the vertex on the subject's native surface that corresponds to the input vertex
        index.
    """

    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    # Load fsaverage sphere
    fsaverage_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', f'{hemi}.sphere.reg')
    )

    # Get the coordinate of the corresponding vertex on the fsaverage sphere
    fsave_sphere_coord = fsaverage_sphere_vertices[vert_idx, :]

    # Load subject registered sphere
    subj_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subject_dir, 'surf', f'{hemi}.sphere.reg')
    )

    # Get the index of the nearest vertex on the subject sphere
    kdtree = KDTree(subj_sphere_vertices)
    _, subj_v_idx = kdtree.query(fsave_sphere_coord, k=1)

    # Adjust vertex index for right hemishphere
    if hemi=='rh':
        lh_vertices, _ = nib.freesurfer.read_geometry(
            os.path.join(fs_subject_dir, 'surf', 'lh.pial')
        )
        subj_v_idx += lh_vertices.shape[0]

    return subj_v_idx


def convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord):
    """
    Convert coordinates from a subject's native surface space to the fsaverage surface space.

    This function maps a vertex coordinate from a subject's native combined pial surface to the
    corresponding vertex index in the fsaverage template space. It determines which hemisphere the
    vertex belongs to based on the closest match in the left and right hemispheres' pial surfaces.
    It then finds the nearest vertex in the subject's registered spherical surface, maps this to
    the nearest vertex in the fsaverage spherical surface, and returns the index of this fsaverage
    vertex.

    Parameters
    ----------
    subj_id : str
        The subject identifier for which the conversion is being performed.
    subj_surf_dir : str
        The path containing the laMEG-processed subject surfaces.
    subj_coord : array-like
        The x, y, z coordinates on the subject's combined hemisphere pial surface to be converted.

    Returns
    -------
    hemi : str
        The hemisphere the vertex is found in ('lh' for left hemisphere, 'rh' for right
        hemisphere).
    fsave_v_idx : int
        Index of the vertex on the fsaverage spherical surface that corresponds to the input
        coordinates.
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

    subj_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subject_dir, 'surf', f'{hemi}.sphere.reg')
    )

    # Get the coordinate of the corresponding vertex on the registered subject sphere
    subj_sphere_coord = subj_sphere_vertices[vert_idx, :]

    # Load FS-average sphere
    fsaverage_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', f'{hemi}.sphere.reg')
    )

    # Get the index of the nearest vertex on the fsaverage sphere
    kdtree = KDTree(fsaverage_sphere_vertices)
    _, fsave_v_idx = kdtree.query(subj_sphere_coord, k=1)

    return hemi, fsave_v_idx


def ttest_rel_corrected(data, correction=0, tail=0, axis=0):
    """
    Perform a corrected paired t-test on a sample of data.

    This function handles missing data (NaNs) and applies a variance correction to the t-test
    calculation. It computes the t-statistic and corresponding p-value for the hypothesis test.

    Parameters
    ----------
    data : array_like
        A 2-D array containing the sample data. NaN values are allowed and are handled
        appropriately.
    correction : float, optional
        The correction value to be added to the variance to avoid division by zero issues. If set
        to 0 (default), an automatic correction of 0.01 \* max(variance) is applied.
    tail : int, optional
        Specifies the type of t-test to be performed:
        - 0 for a two-tailed test (default),
        - 1 for a right one-tailed test,
        - -1 for a left one-tailed test.
    axis : int, optional
        Axis along which to perform the t-test. Default is 0.

    Returns
    -------
    t-statistic : float
        The computed t-statistic for the test.
    degrees_of_freedom : int
        The degrees of freedom for the t-test.
    p-value : float
        The p-value for the test.

    Notes
    -----
    - The function handles NaNs by computing the sample size, mean, and variance while ignoring
      NaNs.
    - The degrees of freedom for the t-test are computed as `max(sample size - 1, 0)`.
    - The standard error of the mean is adjusted with the variance correction.
    - The p-value is computed based on the specified tail type of the t-test.
    """

    # Handle NaNs
    nans = np.isnan(data)
    if np.any(nans):
        samplesize = np.sum(~nans, axis=axis)
    else:
        samplesize = data.shape[axis]

    deg_of_freedom = np.maximum(samplesize - 1, 0)
    xmean = np.nanmean(data, axis=axis)
    varpop = np.nanvar(data, axis=axis)

    # Apply correction
    if correction == 0:
        correction = 0.01 * np.max(varpop)

    corrsdpop = np.sqrt(varpop + correction)
    ser = corrsdpop / np.sqrt(samplesize)
    tval = (xmean - 0) / ser

    # Compute p-value
    p_val = np.inf
    if tail == 0:  # two-tailed test
        p_val = 2 * t.sf(np.abs(tval), deg_of_freedom)
    elif tail == 1:  # right one-tailed test
        p_val = t.sf(-tval, deg_of_freedom)
    elif tail == -1:  # left one-tailed test
        p_val = t.cdf(tval, deg_of_freedom)

    return tval, deg_of_freedom, p_val


def calc_prop(vec):
    """
    Convert independent thickness values to cumulative proportions, while respecting zeros.

    This function calculates the cumulative sum of the input vector and normalizes it by the total
    sum of the vector. If the total sum is zero, the original vector is returned.

    Parameters
    ----------
    vec : array-like
        Input array of thickness values.

    Returns
    -------
    vec : array-like
        The cumulative proportion of the input values, normalized by the total sum. If the sum of
        the input values is zero, the original vector is returned.
    """

    sum_ = np.sum(vec)
    if sum_ == 0.0:
        return vec
    vec = np.cumsum(vec) / sum_
    return vec


def big_brain_proportional_layer_boundaries(overwrite=False):
    """
    Get the proportional layer boundaries (6 values between 0 and 1) from the fsaverage-converted
    Big Brain atlas included in laMEG.

    This function uses the fsaverage-converted Big Brain cortical thickness atlas to calculate
    normalized distances between cortical layer boundaries (from layer 1 to layer 6) with values
    between 0 and 1. To speed up computation, the results are stored in a NumPy dictionary.

    Parameters
    ----------
    overwrite : bool
        Whether to overwrite the existing file.

    Returns
    -------
    bb_data : dict
        Dictionary with keys "lh" (left hemisphere) and "rh" (right hemisphere) containing arrays
        with layer boundaries for each vertex in the hemisphere.
    """

    asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './assets')

    bbl_file = os.path.join(asset_path, "proportional_layer_boundaries.npy")
    if any([not os.path.exists(bbl_file), overwrite]):
        bb_l_paths = get_files(asset_path, "*.gii", strings=["tpl-fsaverage", "hemi-L"])
        bb_l_paths.sort()
        bb_r_paths = get_files(asset_path, "*.gii", strings=["tpl-fsaverage", "hemi-R"])
        bb_r_paths.sort()
        bb_data = {
            "lh": np.array([nib.load(p).agg_data() for p in bb_l_paths]),
            "rh": np.array([nib.load(p).agg_data() for p in bb_r_paths])
        }

        bb_data = {
           key: np.apply_along_axis(calc_prop, 0, value) for key, value in bb_data.items()
        }
        np.save(bbl_file, bb_data)
        return bb_data

    bb_data = np.load(bbl_file, allow_pickle=True).item()

    return bb_data


def get_bigbrain_layer_boundaries(subj_id, subj_surf_dir, subj_coord):
    """
    Get the cortical layer boundaries based on Big Brain atlas for a specified coordinate in the
    subject's downsampled combined space.

    This function maps a vertex coordinate from a subject's native combined pial surface to the
    corresponding vertex index in the fsaverage template space for a specific hemisphere. Then,
    the proportional layer boundaries (6 values between 0 and 1) from the fsaverage-converted Big
    Brain atlas are returned (from layer 1 to layer 6). To get the subject's proportional values,
    these values must be multiplied by the observed cortical thickness.

    Parameters
    ----------
    subj_id : str
        The subject identifier for which the conversion is being performed.
    subj_surf_dir : str
        The path containing the laMEG-processed subject surfaces.
    subj_coord : array-like
        The x, y, z coordinates on the subject's combined hemisphere pial surface to be converted.

    Returns
    -------
    vert_bb_prop : array-like
        Proportional layer boundaries (6 values between 0 and 1) from the fsaverage-converted Big
        Brain atlas.
    """

    # convert subj_coord to native + hemisphere
    hemi, fsave_v_idx = convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord)

    # compute or read (if the precomputed atlas is present)
    bb_prop = big_brain_proportional_layer_boundaries()

    # get the layer boundaries from the fsaverage vertex
    vert_bb_prop = bb_prop[hemi][:,fsave_v_idx]

    return vert_bb_prop


def get_fiducial_coords(subj_id, fname, col_delimiter='\t', subject_column='subj_id',
                        nas_column='nas', lpa_column='lpa', rpa_column='rpa', val_delimiter=','):
    """
    Fetch fiducial coordinates from a tab-separated values (TSV) file for a given subject ID.

    Parameters
    ----------
    subj_id : str
        The subject ID to search for in the TSV file.
    fname : str
        Path to the TSV file.
    col_delimiter : str, optional
        Column delimiter when reading file. Default is \t.
    subject_column : str, optional
        Column name for subject. Default is subj_id.
    nas_column : str, optional
        Column name for nas coordinate. Default is nas.
    lpa_column : str, optional
        Column name for lpa coordinate. Default is lpa.
    rpa_column : str, optional
        Column name for rpa coordinate. Default is rpa.
    val_delimiter : str, optional
        Value delimiter when reading file. Default is ,.

    Returns
    -------
    NAS : list
        List of floats representing the NASion fiducial coordinates.
    LPA : list
        List of floats representing the Left Preauricular fiducial coordinates.
    RPA : list
        List of floats representing the Right Preauricular fiducial coordinates.
    """

    with open(fname, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=col_delimiter)
        for row in reader:
            if row[subject_column] == subj_id:
                nas = [float(i) for i in row[nas_column].split(val_delimiter)]
                lpa = [float(i) for i in row[lpa_column].split(val_delimiter)]
                rpa = [float(i) for i in row[rpa_column].split(val_delimiter)]
                return nas, lpa, rpa

    return None, None, None  # Return None for each if no matching subj_id is found
