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

# pylint: disable=C0302
import csv
import os
import subprocess
import tempfile
from pathlib import Path
from contextlib import contextmanager

import mne
import numpy as np
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import vtk
from mne.coreg import Coregistration
from mne.io import _empty_info
from mne.transforms import apply_trans
from scipy.io import savemat, loadmat
from scipy.spatial import cKDTree  # pylint: disable=E0611
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
        An array containing the MEG data timestamps (in ms).
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

    time = (np.arange(n_samples) / fsample + time_onset)*1000 # Convert time to milliseconds

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


def convert_fsaverage_to_native(subj_id, subj_surf_dir, hemi, vert_idx=None):
    """
    Map vertex indices from fsaverage spherical registration space to a subject's native,
    downsampled pial surface space.

    This function performs two mappings:
     1. It finds, via nearest-neighbor search on the spherical registration surface
        (`?h.sphere.reg`), the subject's full-resolution vertex that corresponds to each
        fsaverage vertex.
     2. It then finds the corresponding vertex (if any) in the subject's downsampled pial
        surface (`pial.ds.gii`), returning only those vertices that have an exact match
        (distance == 0).

    Parameters
    ----------
    subj_id : str
       The subject identifier in the FreeSurfer directory (i.e., folder name under $SUBJECTS_DIR).
    subj_surf_dir : str
       Path to the directory containing the subject's laMEG or custom surface files
       (must include `pial.ds.gii` and `{lh,rh}.pial.gii`).
    hemi : {'lh', 'rh'}
       Hemisphere to convert.
    vert_idx : int or array-like of int or None, optional
       Vertex index/indices on the fsaverage surface to be converted. If None, all vertices in
       the specified hemisphere are mapped. Defaults to None.

    Returns
    -------
    subj_v_idx : int or np.ndarray
       - If a single fsaverage vertex index is given, returns an integer index of the
         corresponding vertex in the subject's downsampled pial surface.
       - If multiple vertex indices are given or `vert_idx` is None, returns an array of indices
         on the subject's downsampled pial surface corresponding to the input fsaverage vertices.
         Only vertices with exact matches (distance == 0) are included.

    Notes
    -----
    This method uses nearest-neighbor mapping on the spherical registration surfaces
    (`fsaverage/?h.sphere.reg` -> `subject/?h.sphere.reg`) to approximate the FreeSurfer morph
    mapping, and then constrains results to the subject's downsampled pial mesh. It assumes that
    the downsampled surface is a vertex subset of the full-resolution pial surface.
    """

    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    # Load fsaverage sphere for the specified hemisphere
    fsaverage_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', f'{hemi}.sphere.reg')
    )
    # Load subject sphere for the specified hemisphere
    subj_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subject_dir, 'surf', f'{hemi}.sphere.reg')
    )
    # Build KDTree for subject sphere
    kdtree = cKDTree(subj_sphere_vertices)

    if vert_idx is None:
        vert_idx = np.arange(fsaverage_sphere_vertices.shape[0])

    _, subj_v_idx_local = kdtree.query(fsaverage_sphere_vertices[vert_idx, :], k=1)
    subj_v_idx_local = np.squeeze(subj_v_idx_local)

    # Load downsampled surface
    subj_ds = nib.load(os.path.join(subj_surf_dir, 'pial.ds.gii'))
    ds_vertices = subj_ds.darrays[0].data

    # Load full-resolution pial surfaces
    subj_fr = nib.load(os.path.join(subj_surf_dir, f'{hemi}.pial.gii'))
    fr_vertices = subj_fr.darrays[0].data

    ds_kdtree = cKDTree(ds_vertices)
    _, v_idx = ds_kdtree.query(fr_vertices[subj_v_idx_local, :])
    v_idx = np.squeeze(v_idx)

    return v_idx



def convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord=None):
    """
    Convert coordinates from a subject's native surface space to the fsaverage surface space.

    This function maps a vertex coordinate from a subject's native combined pial surface
    to the corresponding vertex index in the fsaverage template space. If no coordinate
    is provided, it maps all vertices in the subject's downsampled pial surface to fsaverage.

    Parameters
    ----------
    subj_id : str
        The subject identifier for which the conversion is being performed.
    subj_surf_dir : str
        The path containing the laMEG-processed subject surfaces.
    subj_coord : array-like, optional
        The x, y, z coordinates on the subject's combined hemisphere pial surface to be converted.
        If None, all downsampled pial vertices are mapped to fsaverage.

    Returns
    -------
    hemi : str or list of str
        The hemisphere(s) the vertex is found in ('lh' for left hemisphere, 'rh' for right
        hemisphere').
    fsave_v_idx : int or list of int
        Index or indices of the vertex on the fsaverage spherical surface that corresponds to the
        input coordinates.
    """
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    # Prepare downsampled vertex coordinates
    if subj_coord is None:
        # Load downsampled surface
        subj_ds = nib.load(os.path.join(subj_surf_dir, 'pial.ds.gii'))
        ds_vertices = subj_ds.darrays[0].data
    else:
        ds_vertices = np.array([subj_coord])

    # Load full-resolution pial surfaces
    subj_lh = nib.load(os.path.join(subj_surf_dir, 'lh.pial.gii'))
    lh_vertices = subj_lh.darrays[0].data
    subj_rh = nib.load(os.path.join(subj_surf_dir, 'rh.pial.gii'))
    rh_vertices = subj_rh.darrays[0].data

    # KDTree for finding the closest full-resolution vertex
    lh_kdtree = cKDTree(lh_vertices)
    rh_kdtree = cKDTree(rh_vertices)

    # Get indices of vertices in full-resolution surface
    lh_dists, lh_pial_idx = lh_kdtree.query(ds_vertices, k=1)
    lh_pial_idx = np.squeeze(lh_pial_idx)
    rh_dists, rh_pial_idx = rh_kdtree.query(ds_vertices, k=1)
    rh_pial_idx = np.squeeze(rh_pial_idx)

    # Assign each vertex to the closest full-resolution vertex
    hemis = np.where(lh_dists < rh_dists, 'lh', 'rh')
    pial_vert_indices = np.where(lh_dists < rh_dists, lh_pial_idx, rh_pial_idx)

    # Load fsaverage spheres
    fsaverage_lh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', 'lh.sphere.reg')
    )
    fsaverage_rh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', 'rh.sphere.reg')
    )

    # Load subject registered sphere surfaces
    subj_lh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subject_dir, 'surf', 'lh.sphere.reg')
    )
    subj_rh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subject_dir, 'surf', 'rh.sphere.reg')
    )

    # Precompute KDTree for fsaverage surfaces
    fs_lh_kdtree = cKDTree(fsaverage_lh_sphere_vertices)
    fs_rh_kdtree = cKDTree(fsaverage_rh_sphere_vertices)

    # Select appropriate subject sphere vertices
    subj_sphere_coords = np.array([
        subj_lh_sphere_vertices[idx] if hemi == 'lh' else subj_rh_sphere_vertices[idx]
        for hemi, idx in zip(hemis, pial_vert_indices)
    ])

    # Map to fsaverage
    fsave_v_idx = np.array([
        np.squeeze(fs_lh_kdtree.query(coord, k=1)[1] if hemi == 'lh'
                   else fs_rh_kdtree.query(coord, k=1)[1])
        for hemi, coord in zip(hemis, subj_sphere_coords)
    ])

    # Return results
    if subj_coord is not None:
        return hemis[0], fsave_v_idx[0]
    return hemis.tolist(), fsave_v_idx.tolist()


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


def get_bigbrain_layer_boundaries(subj_id, subj_surf_dir, subj_coord=None):
    """
    Get the cortical layer boundaries based on the Big Brain atlas for a specified coordinate
    in the subject's downsampled combined space. If subj_coord is None, this function returns
    the 6 proportional layer boundaries for every vertex in the downsampled mesh.

    Parameters
    ----------
    subj_id : str
        The subject identifier for which the conversion is being performed.
    subj_surf_dir : str
        The path containing the laMEG-processed subject surfaces.
    subj_coord : array-like or None, optional
        The x, y, z coordinates on the subject's combined hemisphere pial surface to be
        converted. If None, all downsampled pial vertices are mapped. Defaults to None.

    Returns
    -------
    vert_bb_prop : np.ndarray
        A 6 x M array of proportional layer boundaries (rows = 6 layer boundaries,
        columns = vertices), where M is the number of vertices in subj_coord (if provided)
        or in the downsampled mesh (if subj_coord is None). Values range between 0 and 1,
        which must be scaled by the cortical thickness to get layer depths in millimeters.
    """
    # Convert subject coordinate(s) to fsaverage vertex index
    hemi, fsave_v_idx = convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord)

    # Retrieve or compute the Big Brain proportional layer boundaries
    # big_brain_proportional_layer_boundaries() is assumed to return a dict:
    #    {'lh': <6 x N_lh array>, 'rh': <6 x N_rh array>}
    bb_prop = big_brain_proportional_layer_boundaries()

    # If we only have a single coordinate, hemi will be a string; otherwise, it is a list of hemis
    if isinstance(hemi, str):
        # Single coordinate: just index directly
        vert_bb_prop = bb_prop[hemi][:, fsave_v_idx]
    else:
        # Multiple coordinates: build a 6 x M array
        vert_bb_prop = np.zeros((6, len(hemi)))
        for i, (v_h, idx) in enumerate(zip(hemi, fsave_v_idx)):
            vert_bb_prop[:, i] = bb_prop[v_h][:, idx]

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


# pylint: disable=R0915
def coregister_3d_scan_mri(subject_id, lpa, rpa, nas, dig_face_fname, dig_units='mm',
                           out_dir=None):
    """
    Coregister a 3D facial scan to a FreeSurfer MRI for a given subject.

    This function performs an initial fiducial-based alignment followed by
    iterative closest point (ICP) alignment using dense head surface points
    extracted from a 3D scan of the face. It returns the fiducial coordinates
    (LPA, RPA, NAS) transformed into FreeSurfer MRI voxel coordinates.

    Assumes FreeSurfer is installed and configured, and that the SUBJECTS_DIR
    environment variable is set to the location of FreeSurfer subjects.

    Parameters
    ----------
    subject_id : str
        Name of the FreeSurfer subject.
    lpa : array-like, shape (3,)
        Left preauricular point in mm, in the 3D scan coordinate frame.
    rpa : array-like, shape (3,)
        Right preauricular point in mm, in the 3D scan coordinate frame.
    nas : array-like, shape (3,)
        Nasion point in mm, in the 3D scan coordinate frame.
    dig_face_fname : str
        Path to a 3D mesh file of the subject's face (e.g., .stl) containing
        dense surface points including fiducials.
    dig_units : str
        Units of coordinates in the 3D facial scan (m or mm). Default is mm
    out_dir : str or None, optional
        Directory where coregistration visualizations will be saved. If None,
        no figures will be written.

    Returns
    -------
    lpa_t : ndarray, shape (3,)
        LPA coordinate in FreeSurfer MRI voxel space (mm), with CRAS offset applied.
    rpa_t : ndarray, shape (3,)
        RPA coordinate in FreeSurfer MRI voxel space (mm), with CRAS offset applied.
    nas_t : ndarray, shape (3,)
        NAS coordinate in FreeSurfer MRI voxel space (mm), with CRAS offset applied.

    Notes
    -----
    The returned fiducial coordinates are in FreeSurfer MRI voxel space (scanner RAS + CRAS
    offset), not in FreeSurfer surface RAS space. This distinction matters when using the
    coordinates for surface-based analyses or visualizations.
    """
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')

    # Convert fiducials to meters
    lpa = lpa * 1e-3
    rpa = rpa * 1e-3
    nas = nas * 1e-3

    # Get the 3d surface of the face
    reader = vtk.vtkSTLReader() # pylint: disable=E1101
    reader.SetFileName(dig_face_fname)
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    n_points = points.GetNumberOfPoints()
    vertices = np.array([points.GetPoint(i) for i in range(n_points)])
    # Get all vertices as points - round
    points = np.unique(np.round(vertices, 2), axis=0)
    # Convert mm to m
    if dig_units=='mm':
        points = points * 1e-3
    dig = mne.channels.make_dig_montage(
        hsp=points,
        lpa=lpa,
        rpa=rpa,
        nasion=nas,
        coord_frame='head'
    )

    # Create a fake info object for the digital montage
    info = _empty_info(1)
    info["dig"] = dig
    info._unlocked = False # pylint:disable=W0212

    # Plotting options
    plot_kwargs = {
        'subject': subject_id,
        'subjects_dir': fs_subjects_dir,
        'surfaces': "head-dense",
        'dig': True,
        'eeg': [],
        'meg': [],
        'show_axes': True,
        'coord_frame': "mri"
    }

    # Create a coregistration
    coreg = Coregistration(
        info,
        subject_id,
        fs_subjects_dir,
        fiducials='auto',
        on_defects="ignore"
    )
    coreg._setup_digs() # pylint:disable=W0212

    # Visualize initial alignment
    align_fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
    if out_dir is not None:
        screenshot = align_fig.plotter.screenshot()
        fig, axis = plt.subplots(figsize=(10, 10))
        axis.imshow(screenshot, origin='upper')
        axis.set_axis_off()  # Disable axis labels and ticks
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'coreg-initial.png'))

    # Rough fit of fiducials to get rough alignment
    coreg.set_scale_mode('uniform')
    coreg.fit_fiducials(verbose=True)

    # Visualize rough alignment
    align_fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
    if out_dir is not None:
        screenshot = align_fig.plotter.screenshot()
        fig, axis = plt.subplots(figsize=(10, 10))
        axis.imshow(screenshot, origin='upper')
        axis.set_axis_off()  # Disable axis labels and ticks
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'coreg-fit_fiducials.png'))

    # ICP fit - not fitting fiducial locations, just digitised points
    coreg.fit_icp(n_iterations=50, nasion_weight=0.0,  lpa_weight=0.0, rpa_weight=0.0, verbose=True)

    # Visualize final alignment
    align_fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
    if out_dir is not None:
        screenshot = align_fig.plotter.screenshot()
        fig, axis = plt.subplots(figsize=(10, 10))
        axis.imshow(screenshot, origin='upper')
        axis.set_axis_off()  # Disable axis labels and ticks
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'coreg-fit_icp.png'))


    # Apply inverse transformation to fiducial coordinates to get coordinates in FreeSurfer space,
    # convert to mm
    lpa_tkras = apply_trans(coreg.trans, lpa) * 1e3
    rpa_tkras = apply_trans(coreg.trans, rpa) * 1e3
    nas_tkras = apply_trans(coreg.trans, nas) * 1e3

    # Read FreeSurfer matrices from the SAME volume SPM will use (ideally orig.mgz from recon-all)
    orig_mgz = os.path.join(fs_subjects_dir, subject_id, 'mri', 'orig.mgz')

    def _mat(flag):  # returns 4x4
        out = subprocess.check_output(['mri_info', flag, orig_mgz]).decode().strip().split()
        return np.array([float(x) for x in out]).reshape(4, 4)

    t_orig = _mat('--vox2ras-tkr')  # vox - tkRAS (mm)
    n_orig = _mat('--vox2ras')  # vox - scanner RAS (mm)
    it_orig = np.linalg.inv(t_orig)

    def tkras_to_scanner_ras(xyz_mm):
        xyz1 = np.r_[xyz_mm, 1.0]
        return (n_orig @ (it_orig @ xyz1))[:3]

    lpa_spm = tkras_to_scanner_ras(lpa_tkras)
    rpa_spm = tkras_to_scanner_ras(rpa_tkras)
    nas_spm = tkras_to_scanner_ras(nas_tkras)

    return lpa_spm, rpa_spm, nas_spm
