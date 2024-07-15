"""
This module contains the unit tests for the `surf` module from the `lameg` package.
"""
import os

import numpy as np
import pytest
from scipy.io import loadmat

from lameg.surf import split_fv, smoothmesh_multilayer_mm  # pylint: disable=no-name-in-module


def test_split_fv():
    """
    Test the split_fv function to ensure it correctly splits a set of faces and vertices into
    distinct patches based on connectivity.

    This unit test performs the following:
    - Initializes two numpy arrays, `faces` and `vertices`
    - Invokes the `split_fv` function to partition the input mesh into separate patches.
    - Verifies that the number of returned patches matches the expected count.
    - Checks that both the faces and vertices of each patch accurately match the predefined target
      arrays using numpy operations to confirm zero discrepancies.

    The test cases include:
    - A mesh where two patches are expected.
    - A larger mesh configuration that results in a single patch, ensuring robustness to changes in
      input size.

    Raises:
        AssertionError: If any of the conditions are not met, indicating a failure in the mesh
        splitting logic.
    """
    faces = np.array([[1, 2, 3], [1, 3, 4], [5, 6, 1], [7, 8, 9], [11, 10, 4]])
    vertices = np.array([[2, 4],
                  [2, 8],
                  [8, 4],
                  [8, 0],
                  [0, 4],
                  [2, 6],
                  [2, 2],
                  [4, 2],
                  [4, 0],
                  [5, 2],
                  [5, 0]])
    split_patches = split_fv(faces, vertices)

    assert len(split_patches) == 2

    patch_1_faces = split_patches[0]['faces']
    target = np.array([[0, 1, 2],
                       [0, 2, 3],
                       [4, 5, 0],
                       [7, 6, 3]])
    assert np.allclose(patch_1_faces, target)

    patch_1_vertices = split_patches[0]['vertices']
    target = np.array([[2, 4],
                       [2, 8],
                       [8, 4],
                       [8, 0],
                       [0, 4],
                       [2, 6],
                       [5, 2],
                       [5, 0]])
    assert np.allclose(patch_1_vertices, target)

    patch_2_faces = split_patches[1]['faces']
    target = np.array([[0, 1, 2]])
    assert np.allclose(patch_2_faces, target)

    patch_2_vertices = split_patches[1]['vertices']
    target = np.array([[2, 2],
                       [4, 2],
                       [4, 0]])
    assert np.allclose(patch_2_vertices, target)

    faces = np.array([[1, 2, 3], [1, 3, 4], [5, 6, 1], [7, 8, 9], [11, 10, 4], [1, 6, 7]])
    vertices = np.array([[2, 4],
                  [2, 8],
                  [8, 4],
                  [8, 0],
                  [0, 4],
                  [2, 6],
                  [2, 2],
                  [4, 2],
                  [4, 0],
                  [5, 2],
                  [5, 0]])
    split_patches = split_fv(faces, vertices)

    assert len(split_patches) == 1

    patch_faces = split_patches[0]['faces']
    target = np.array([[ 0,  1,  2],
                       [ 0,  2,  3],
                       [ 4,  5,  0],
                       [ 6,  7,  8],
                       [10,  9,  3],
                       [ 0,  5,  6]])
    assert np.allclose(patch_faces, target)

    patch_vertices = split_patches[0]['vertices']
    target = np.array([[2, 4],
                       [2, 8],
                       [8, 4],
                       [8, 0],
                       [0, 4],
                       [2, 6],
                       [2, 2],
                       [4, 2],
                       [4, 0],
                       [5, 2],
                       [5, 0]])
    assert np.allclose(patch_vertices, target)


@pytest.mark.dependency()
def test_smoothmesh_multilayer_mm():
    """
    Tests the `smoothmesh_multilayer_mm` function to ensure it correctly smooths the surface mesh
    data and saves the output as expected. The test verifies the function's ability to handle file
    operations correctly and produce a matrix of smoothed values that match a predefined target.

    This test primarily checks:
    1. The function's capability to execute smoothing operations on multilayer mesh data.
    2. The creation of an output file following the smoothing process.
    3. The correctness of the output data by comparing the contents of the newly created file
       against expected values.

    Steps performed:
    - Run the `smoothmesh_multilayer_mm` function with specified parameters to perform the
      smoothing.
    - Check for the existence of the output file post-function execution to ensure file operations
      are handled correctly.
    - Load the output file and validate the content against a target matrix to ensure the smoothing
      process is accurate.

    Assertions:
    - Assert that the output file exists after function execution.
    - Assert that the numerical content of the output file closely matches the predefined target
      values, confirming both the smoothing accuracy and the correctness of file writing
      operations.
    """

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')
    subj_id = 'sub-104'
    # pylint: disable=duplicate-code
    mesh_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'multilayer.2.ds.link_vector.fixed.gii'
    )

    out_fname = os.path.join(
        test_data_path,
        subj_id,
        'surf',
        'FWHM5.00_multilayer.2.ds.link_vector.fixed.mat'
    )

    smoothmesh_multilayer_mm(mesh_fname, 5, 2)

    assert os.path.exists(out_fname)

    results = loadmat(out_fname)
    target = np.array(
        [[0.17707503, 0.12816778, 0.04624505, 0.01718994, 0., 0., 0., 0., 0.13324403 ,0.],
         [0.16848713, 0.23277974, 0.1643145,  0.09286189, 0., 0., 0., 0., 0.06913673, 0.],
         [0.05768403, 0.15591159, 0.22087557, 0.19283595, 0., 0., 0., 0., 0., 0.],
         [0.01843762, 0.07576701, 0.16581663, 0.18992747, 0., 0., 0.,0., 0., 0.],
         [0., 0., 0., 0., 0.24208555, 0.20217255, 0.12082949, 0., 0., 0.]]
    )
    assert np.allclose(results['QG'][:5,:10].todense(), target)
