"""
Utility functions for SPM, MEG sensor data management, and neuroanatomical processing.

This module provides a suite of high-level tools to interface with SPM (Statistical Parametric
Mapping), MNE-Python, and FreeSurfer environments. It facilitates batch execution, data format
conversion, sensor-level data handling, and laminar or anatomical computations relevant to
MEG/EEG analysis pipelines. Many functions integrate seamlessly with standalone SPM sessions and
FreeSurfer-derived cortical surfaces.

Core functionalities
--------------------
- **SPM interfacing and batch execution**
  - Context-managed lifecycle for standalone SPM instances (`spm_context`).
  - Automated batch execution of MATLAB-based neuroimaging workflows (`batch`).
  - Conversion of CTF MEG data from MNE `.fif` to SPM `.mat` format (`ctf_fif_spm_conversion`).

- **File and directory utilities**
  - Structured recursive file and directory retrieval (`get_files`, `get_directories`).
  - Directory creation with automatic parent handling (`make_directory`).
  - Flexible substring matching (`check_many`).

- **Statistical and numerical utilities**
  - Corrected paired-sample t-tests with NaN handling (`ttest_rel_corrected`).
  - Conversion of absolute layer thicknesses to proportional coordinates (`calc_prop`).

- **Anatomical and laminar processing**
  - Derivation of normalized laminar boundaries from the fsaverage BigBrain atlas
    (`big_brain_proportional_layer_boundaries`).

- **Coregistration and fiducial handling**
  - Extraction of subject-specific fiducial coordinates from TSV metadata files
    (`get_fiducial_coords`).
  - Coregistration of 3D facial scans to FreeSurfer MRI via fiducial and ICP alignment
    (`coregister_3d_scan_mri`).

Notes
-----
- The module assumes correctly configured environments for SPM standalone, MNE-Python, and
  FreeSurfer.
- Designed for reproducible neuroimaging workflows combining MATLAB-based and Python-based tools.
- File operations are implemented with explicit error handling for compatibility across platforms.
- The laminar utilities rely on fsaverage-space anatomical templates distributed with laMEG.

"""


# pylint: disable=C0302
import csv
import os
import shutil
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
from scipy.stats import t
import spm_standalone


@contextmanager
def spm_context(spm=None, n_jobs=4):
    """
    Context manager for safe initialization and termination of standalone SPM sessions.

    This utility ensures proper lifecycle management of SPM standalone instances when executing
    MATLAB-based analyses (e.g., source reconstruction or simulation). It supports both
    user-supplied SPM sessions and automatic instantiation of temporary ones with optional
    parallelization.

    Parameters
    ----------
    spm : spm_standalone, optional
        Existing SPM standalone instance. If None, a new instance is launched and automatically
        terminated upon context exit (default: None).
    n_jobs : int, optional
        Number of MATLAB parallel workers to initialize via `parpool`. Default is 4.

    Yields
    ------
    spm : spm_standalone
        Active standalone SPM instance usable within the context.

    Notes
    -----
    - Designed for use within a `with` block to ensure safe cleanup of MATLAB processes.
    - If `spm` is None, a new standalone SPM session is created with a parallel pool of size
      `n_jobs`, and both are terminated automatically upon exit.
    - If an existing `spm` instance is provided, it is reused and not terminated at the end of the
      context.
    - The default setting (`n_jobs=4`) is suitable for standard workstations; higher values may
      be beneficial on high-performance computing (HPC) systems.
    - Ensures robustness against MATLAB errors when initializing or closing `parpool`.
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
    Execute an SPM batch job within a managed standalone MATLAB session.

    This function runs an arbitrary SPM batch configuration (`matlabbatch`) by writing it to a
    temporary MATLAB file and executing it within an SPM standalone environment. It supports both
    interactive (visual) and non-interactive (headless) execution, automatically handling session
    setup and cleanup.

    Parameters
    ----------
    cfg : dict
        Dictionary defining the SPM batch configuration, following the standard `matlabbatch`
        format (e.g., as used in `spm_jobman`).
    viz : bool, optional
        Whether to display the SPM GUI during execution (`True`) or run in command-line mode
        (`False`). Default is True.
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Notes
    -----
    - The function automatically wraps `cfg` into a valid `matlabbatch` structure and saves it to
      a temporary `.mat` file before execution.
    - Temporary files are created using the system's default temp directory and deleted after
      the job completes.
    - Runs all standard SPM batch modules, including EEG/MEG preprocessing, coregistration,
      inversion, and statistical analyses.
    - When `viz=False`, `spm_get_defaults('cmdline', 1)` is used to suppress GUI output, making
      the function suitable for automated pipelines and HPC execution.
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
    Load MEG sensor-level data and metadata from an SPM M/EEG dataset.

    This function extracts MEG channel data, timestamps, and channel labels from an SPM-format
    dataset, supporting both HDF5-based (v7.3+) and older MATLAB `.mat` file structures. It filters
    out non-MEG and bad channels, reconstructs the binary data matrix, and returns it in
    sensor × time × trial format (if applicable).

    Parameters
    ----------
    data_fname : str
        Path to the SPM M/EEG `.mat` file containing metadata and reference to the binary data file.

    Returns
    -------
    sensor_data : np.ndarray
        MEG sensor data (channels × time [× trials]) extracted from the referenced binary file.
    time : np.ndarray
        Time vector in milliseconds.
    ch_names : list of str
        Names of the valid MEG channels included in the output.

    Raises
    ------
    FileNotFoundError
        If the binary data file specified in the `.mat` structure cannot be found.
    OSError
        If the `.mat` file cannot be opened as an HDF5 dataset.
    KeyError
        If required fields are missing from the file.

    Notes
    -----
    - Supports both MATLAB v7.3 (HDF5) and earlier formats.
    - Only MEG channels marked as 'good' (not bad) are retained.
    - Binary data are read directly from the file specified in `D.data.fname` using
      Fortran-order reshaping to match SPM's internal data layout.
    - The returned time vector is derived from `timeOnset`, `Fsample`, and `Nsamples`
      fields within the dataset.
    - Non-MEG modalities (e.g., EEG, EOG, EMG) are excluded automatically.
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


def ctf_fif_spm_conversion(mne_file, res4_file, output_path, epoched, prefix="spm_",
                           spm_instance=None):
    """
    Convert MEG data from CTF `.fif` format to SPM-compatible `.mat` format.

    This function converts raw or epoched MEG data acquired from a CTF scanner (in `.fif` format)
    into SPM's M/EEG format using the standalone SPM interface. It requires the corresponding
    CTF `.res4` file for sensor geometry and saves the converted dataset to the specified output
    directory with a customizable prefix.

    Parameters
    ----------
    mne_file : str or pathlib.Path
        Path to the MNE `.fif` file containing MEG data (either `*-raw.fif` or `*-epo.fif`).
    res4_file : str or pathlib.Path
        Path to the CTF `.res4` file containing sensor position and geometry information.
    output_path : str or pathlib.Path
        Destination directory for the converted SPM dataset.
    epoched : bool
        Whether the input data are epoched (`True`) or continuous (`False`).
    prefix : str, optional
        Prefix to prepend to the converted dataset filename (default: `"spm_"`).
    spm_instance : spm_standalone, optional
        Active standalone SPM instance. If None, a temporary instance is created and closed after
        execution.

    Notes
    -----
    - If `spm_instance` is not provided, a new standalone SPM session is automatically launched and
      terminated after completion.
    - The resulting `.mat` file is compatible with all SPM EEG/MEG preprocessing, inversion, and
      model comparison pipelines.
    - Both paths (`mne_file` and `res4_file`) must be valid and refer to files from the same
      acquisition session.
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
    Evaluate whether multiple substrings occur within a target string.

    This utility checks whether all or any of a list of substrings are present in a given target
    string. It is designed for use in conditional logic (e.g., list comprehensions or filtering
    routines) where flexible substring matching is required.

    Parameters
    ----------
    multiple : list of str
        Substrings to search for within the target string.
    target : str
        String in which to search for the specified substrings.
    func : {'all', 'any'}
        Search mode: `'all'` returns True only if all substrings are found, `'any'` returns True
        if at least one substring is found.

    Returns
    -------
    bool
        True if the condition specified by `func` is satisfied; False otherwise.

    Raises
    ------
    ValueError
        If `func` is not `'all'` or `'any'`.

    Notes
    -----
    - Useful for concise logical checks in comprehensions or filters.
    - Matching is case-sensitive and does not support regular expressions.
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
    Retrieve files from a directory matching specified criteria (suffix, prefix, and substrings).

    This function searches a target directory (optionally recursively) for files with a given
    extension and filters them by filename content and prefix. It supports flexible substring
    matching via "all" or "any" logic, making it suitable for controlled file selection in large
    datasets.

    Parameters
    ----------
    target_path : str or pathlib.Path
        Root directory in which to search for files.
    suffix : str
        File extension to match, in the form '*.ext' (e.g., '*.mat' or '*.fif').
    strings : list of str, optional
        Substrings to be matched within each filename. Default is an empty string.
    prefix : str, optional
        Restrict results to filenames beginning with this prefix. Default is None.
    check : {'all', 'any'}, optional
        Search mode: `'all'` requires all substrings in `strings` to be present; `'any'`
        requires at least one. Default is `'all'`.
    depth : {'all', 'one'}, optional
        Search depth: `'all'` performs a recursive search through subdirectories, `'one'`
        limits the search to the top-level directory. Default is `'all'`.

    Returns
    -------
    files : list of pathlib.Path
        Sorted list of paths to files matching the specified criteria.

    Notes
    -----
    - Matching is case-sensitive and exact (no regular expressions).
    - The function internally uses `check_many()` for substring evaluation.
    - Useful for structured data pipelines where file inclusion depends on both naming and
      hierarchical constraints.
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
    Retrieve directories within a path that contain specified substrings in their names.

    This function searches a target directory (optionally recursively) for subdirectories whose
    names match one or more specified substrings. It supports both shallow and recursive search
    modes and flexible matching logic ("all" or "any") for substring inclusion.

    Parameters
    ----------
    target_path : str or pathlib.Path
        Root directory in which to search for subdirectories.
    strings : list of str, optional
        Substrings to be matched within each directory path or name. Default is an empty string.
    check : {'all', 'any'}, optional
        Search mode: `'all'` requires all substrings in `strings` to be present; `'any'` requires
        at least one. Default is `'all'`.
    depth : {'all', 'one'}, optional
        Search depth: `'all'` performs a recursive search through subdirectories, `'one'`
        limits the search to the top-level directory. Default is `'all'`.

    Returns
    -------
    subdirs : list of pathlib.Path
        Sorted list of paths to directories matching the specified criteria.

    Notes
    -----
    - Matching is case-sensitive and performed using `check_many()`.
    - The returned list is sorted lexicographically by directory path.
    - Useful for structured directory traversal and dataset organization tasks.
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
    Create a directory (and all necessary parent directories) within a specified root path.

    This function ensures that the full directory path exists, creating intermediate directories
    as needed. It supports both single directory names and nested paths provided as lists.

    Parameters
    ----------
    root_path : str or pathlib.Path
        The root directory in which to create the new directory or directories.
    extended_dir : str or list of str
        Subdirectory (or sequence of nested subdirectories) to be created within `root_path`.

    Returns
    -------
    root_path : pathlib.Path
        Path object representing the created directory.

    Notes
    -----
    - Existing directories are preserved (`exist_ok=True`).
    - Intermediate directories are automatically created (`parents=True`).
    - Useful for ensuring consistent directory structure in data processing pipelines.
    """

    root_path = Path(root_path)
    if isinstance(extended_dir, list):
        root_path = root_path.joinpath(*extended_dir)
    else:
        root_path = root_path.joinpath(extended_dir)

    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def ttest_rel_corrected(data, correction=0, tail=0, axis=0):
    """
    Perform a corrected paired-sample t-test with NaN handling and variance stabilization.

    This function computes a t-statistic, degrees of freedom, and p-value for paired data while
    accounting for potential missing values (NaNs) and small-sample variance instabilities.
    A correction term is applied to the variance to prevent division by zero or underflow
    when the variance is near zero.

    Parameters
    ----------
    data : array_like
        Input data array (typically representing paired differences), where the t-test is
        computed along the specified axis. NaN values are ignored in statistical computations.
    correction : float, optional
        Variance correction term. If 0 (default), an adaptive correction of
        `0.01 * max(variance)` is applied automatically.
    tail : {0, 1, -1}, optional
        Specifies the type of test:
        - `0`: two-tailed test (default)
        - `1`: right-tailed test
        - `-1`: left-tailed test
    axis : int, optional
        Axis along which the t-test is performed. Default is 0.

    Returns
    -------
    tval : float or ndarray
        Computed t-statistic(s).
    deg_of_freedom : int or ndarray
        Degrees of freedom associated with each test.
    p_val : float or ndarray
        Corresponding p-value(s).

    Notes
    -----
    - Missing values (NaNs) are excluded from mean and variance computations.
    - The standard error includes the correction term to ensure numerical stability.
    - For two-tailed tests, `p = 2 * t.sf(|tval|, df)`; for one-tailed, `p = t.sf(-tval, df)` or
      `p = t.cdf(tval, df)` depending on direction.
    - This test is useful for robust paired-sample comparisons when small variance or missing
      data may otherwise bias standard t-test results.
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
    Compute cumulative proportional values from a vector while preserving zero-sum cases.

    This function converts an array of independent thickness or weight values into cumulative
    proportions that sum to one. If the total sum of the input is zero, the function returns the
    original vector unchanged to avoid division by zero.

    Parameters
    ----------
    vec : array_like
        Input array of non-negative values (e.g., layer thicknesses or weights).

    Returns
    -------
    vec : np.ndarray
        Cumulative proportion vector, normalized by the total sum. If the input sum is zero,
        returns the original vector unchanged.

    Notes
    -----
    - The result ranges from 0 to 1, representing the normalized cumulative distribution.
    - Useful for converting laminar thickness values into depth proportion coordinates.
    """

    sum_ = np.sum(vec)
    if sum_ == 0.0:
        return vec
    vec = np.cumsum(vec) / sum_
    return vec


def big_brain_proportional_layer_boundaries(overwrite=False):
    """
    Retrieve proportional cortical layer boundary coordinates from the fsaverage-converted
    BigBrain atlas included in laMEG.

    This function computes normalized laminar depth coordinates (ranging from 0 to 1) for each
    cortical vertex using the fsaverage-mapped BigBrain histological atlas. The proportional
    layer boundaries (layers 1-6) are derived from absolute thickness values and cached as a
    NumPy dictionary to improve subsequent loading speed.

    Parameters
    ----------
    overwrite : bool, optional
        If True, recomputes proportional boundaries and overwrites the existing cached file.
        Default is False.

    Returns
    -------
    bb_data : dict
        Dictionary containing normalized layer boundaries for each hemisphere:
        - `"lh"`: left hemisphere vertex-wise boundary array (shape: 6 × n_vertices)
        - `"rh"`: right hemisphere vertex-wise boundary array (shape: 6 × n_vertices)

    Notes
    -----
    - The proportional boundaries represent cumulative thickness proportions from layer 1 (pial)
      to layer 6 (white matter).
    - Cached results are stored as `proportional_layer_boundaries.npy` in the module's `assets`
      directory.
    - Uses `calc_prop()` internally to normalize absolute thickness values per vertex.
    - This dataset provides a standard laminar coordinate reference compatible with fsaverage-
      registered MEG/EEG source models.
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


def get_fiducial_coords(subj_id, fname, col_delimiter='\t', subject_column='subj_id',
                        nas_column='nas', lpa_column='lpa', rpa_column='rpa', val_delimiter=','):
    """
    Retrieve fiducial landmark coordinates (NAS, LPA, RPA) for a specified subject from a TSV file.

    This function reads a tab- or comma-delimited text file containing subject-specific
    fiducial landmarks and returns the NASion (NAS), Left Preauricular (LPA), and Right
    Preauricular (RPA) coordinates for the requested subject. Each coordinate entry
    should contain three comma-separated values representing the x, y, and z positions
    (in millimeters, unless otherwise specified).

    Parameters
    ----------
    subj_id : str
        Subject identifier used to locate the corresponding row in the file.
    fname : str or pathlib.Path
        Path to the TSV file containing fiducial coordinates.
    col_delimiter : str, optional
        Character delimiting columns in the file (default: '\\t').
    subject_column : str, optional
        Column name containing subject identifiers (default: 'subj_id').
    nas_column : str, optional
        Column name for NASion coordinates (default: 'nas').
    lpa_column : str, optional
        Column name for Left Preauricular coordinates (default: 'lpa').
    rpa_column : str, optional
        Column name for Right Preauricular coordinates (default: 'rpa').
    val_delimiter : str, optional
        Character delimiting coordinate values within each cell (default: ',').

    Returns
    -------
    fid_coords : dict or None
        Dictionary containing fiducial coordinates:
        ``{'nas': [x, y, z], 'lpa': [x, y, z], 'rpa': [x, y, z]}``.
        Returns ``None`` if the subject is not found in the file.

    Raises
    ------
    ValueError
        If a matching subject is found but one or more fiducial columns are missing or malformed.

    Notes
    -----
    - The file must contain a header row with named columns.
    - Coordinates are typically in head or MRI space, depending on acquisition convention.
    - Commonly used to initialize coregistration in MEG/EEG preprocessing pipelines
      (e.g., for SPM, MNE, or hpMEG analyses).
    """

    with open(fname, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=col_delimiter)
        for row in reader:
            if row[subject_column] == subj_id:
                fid_coords = {
                    'nas': [float(i) for i in row[nas_column].split(val_delimiter)],
                    'lpa': [float(i) for i in row[lpa_column].split(val_delimiter)],
                    'rpa': [float(i) for i in row[rpa_column].split(val_delimiter)]
                }
                return fid_coords

    return None  # Return None if no matching subj_id is found


# pylint: disable=R0915
def coregister_3d_scan_mri(subject_id, lpa, rpa, nas, dig_face_fname, dig_units='mm',
                           out_dir=None):
    """
    Coregister a 3D facial surface scan to a FreeSurfer MRI using fiducial- and ICP-based alignment.

    This function aligns a subject's 3D facial mesh (e.g., STL scan) to their FreeSurfer MRI space.
    It first performs a fiducial-based rigid alignment using the NAS, LPA, and RPA landmarks, then
    refines the fit using iterative closest point (ICP) optimization on dense head-surface points.
    The resulting transformation is applied to the fiducials, returning their coordinates in
    FreeSurfer MRI voxel space (scanner RAS + CRAS offset).

    Parameters
    ----------
    subject_id : str
        Name of the FreeSurfer subject (must exist in `$SUBJECTS_DIR`).
    lpa : array_like, shape (3,)
        Left preauricular fiducial in millimeters (3D scan coordinate frame).
    rpa : array_like, shape (3,)
        Right preauricular fiducial in millimeters (3D scan coordinate frame).
    nas : array_like, shape (3,)
        Nasion fiducial in millimeters (3D scan coordinate frame).
    dig_face_fname : str or pathlib.Path
        Path to the subject's 3D facial mesh (e.g., `.stl`) containing head-surface points.
    dig_units : {'m', 'mm'}, optional
        Units of the 3D facial scan coordinates. Default is `'mm'`.
    out_dir : str or None, optional
        Directory where visualization screenshots of the alignment stages will be saved.
        If None, no figures are written.

    Returns
    -------
    lpa_t : np.ndarray, shape (3,)
        Transformed LPA coordinate in FreeSurfer MRI voxel space (mm).
    rpa_t : np.ndarray, shape (3,)
        Transformed RPA coordinate in FreeSurfer MRI voxel space (mm).
    nas_t : np.ndarray, shape (3,)
        Transformed NAS coordinate in FreeSurfer MRI voxel space (mm).

    Notes
    -----
    - Requires FreeSurfer to be installed and `$SUBJECTS_DIR` to be set.
    - Uses MNE-Python's `Coregistration` and `plot_alignment` utilities for alignment and
      visualization.
    - Fiducial alignment is refined using ICP on dense surface points but excludes fiducial
      weighting during ICP iterations.
    - The returned coordinates are in **scanner RAS + CRAS** space (voxel-aligned), not FreeSurfer
      surface RAS.
    - If `out_dir` is provided, three screenshots are saved:
      `coreg-initial.png`, `coreg-fit_fiducials.png`, and `coreg-fit_icp.png`.
    - This workflow is useful for aligning 3D optical scans or digitized head shapes to MRI space
      prior to MEG/EEG forward modeling.
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


def check_freesurfer_setup():
    """
    Verify that essential FreeSurfer command-line tools are available in the environment.

    This function checks whether the required FreeSurfer binaries (`mris_convert` and `mri_info`)
    can be found in the system's PATH. If one or more binaries are missing, it raises an
    `EnvironmentError` with detailed setup instructions for sourcing the FreeSurfer environment.

    Raises
    ------
    EnvironmentError
        If one or more required FreeSurfer binaries are not found in the system PATH.

    Notes
    -----
    - This check ensures that FreeSurfer has been properly installed and initialized via:
          export FREESURFER_HOME=/path/to/freesurfer
          source $FREESURFER_HOME/SetUpFreeSurfer.sh
    - Typically called before surface conversion or reconstruction routines that depend on
      FreeSurfer utilities.
    """
    required_bins = ['mris_convert', 'mri_info']
    missing_bins = [b for b in required_bins if shutil.which(b) is None]
    if missing_bins:
        msg = (
                "Missing required FreeSurfer binaries: "
                + ", ".join(missing_bins)
                + "\nPlease ensure FreeSurfer is installed and sourced, e.g.:\n"
                  "    export FREESURFER_HOME=/path/to/freesurfer\n"
                  "    source $FREESURFER_HOME/SetUpFreeSurfer.sh"
        )
        raise EnvironmentError(msg)