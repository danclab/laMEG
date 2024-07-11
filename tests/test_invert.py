"""
This module contains the unit tests for the `invert` module from the `lameg` package.
"""

import os
import shutil
import h5py
from lameg.invert import coregister
from lameg.util import get_fiducial_coords, make_directory


def test_coregister():
    """
    Tests the coregistration process for neuroimaging data, ensuring that the output is properly
    formatted and correctly written to the simulation output files.

    This test performs several key operations:
    1. Retrieves fiducial coordinates necessary for coregistration from a participant's metadata.
    2. Prepares a specific MEG data file for simulation based on test data paths and session
       identifiers.
    3. Copies necessary data files to a designated output directory for processing.
    4. Executes the coregistration function using specified MRI files and mesh data for the forward
       model.
    5. Validates the presence of specific data structures in the output files to verify successful
       coregistration.

    Specific checks include:
    - Verifying that the 'inv' (inverse solution) field does not exist in the original data file's
      structure to ensure it's unprocessed.
    - Confirming that the 'inv' field is present in the new file after coregistration, indicating
      successful processing and data integrity.

    Methods used:
    - get_fiducial_coords to fetch fiducial points.
    - make_directory to create a directory for output data.
    - shutil.copy to duplicate necessary files to the working directory.
    - coregister function to align MEG data with an anatomical MRI using the fiducial points and
      the specified mesh.
    - h5py.File to interact with HDF5 file format for assertions on data structure.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'
    ses_id = 'ses-01'

    # Fiducial coil coordinates
    nas, lpa, rpa = get_fiducial_coords(subj_id, os.path.join(test_data_path, 'participants.tsv'))

    # Data file to base simulations on
    data_file = os.path.join(
        test_data_path,
        subj_id,
        'meg',
        ses_id,
        'spm/spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    )

    data_path, data_file_name = os.path.split(data_file)
    data_base = os.path.splitext(data_file_name)[0]

    # Where to put simulated data
    out_dir = make_directory('./', ['output'])

    # Copy data files to tmp directory
    shutil.copy(
        os.path.join(data_path, f'{data_base}.mat'),
        out_dir.joinpath(f'{data_base}.mat')
    )
    shutil.copy(
        os.path.join(data_path, f'{data_base}.dat'),
        out_dir.joinpath(f'{data_base}.dat')
    )

    # Construct base file name for simulations
    base_fname = os.path.join(out_dir, f'{data_base}.mat')

    # Native space MRI to use for coregistration
    mri_fname = os.path.join(
        test_data_path,
        subj_id,
        'mri',
        's2023-02-28_13-33-133958-00001-00224-1.nii'
    )

    # Mesh to use for forward model in the simulations
    multilayer_mesh_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'multilayer.11.ds.link_vector.fixed.gii'
    )

    # Coregister data to multilayer mesh
    # pylint: disable=duplicate-code
    coregister(
        nas,
        lpa,
        rpa,
        mri_fname,
        multilayer_mesh_fname,
        base_fname
    )

    with h5py.File(data_file, 'r') as old_file:
        assert 'inv' not in old_file['D']['other']

    with h5py.File(base_fname, 'r') as new_file:
        assert 'inv' in new_file['D']['other']
