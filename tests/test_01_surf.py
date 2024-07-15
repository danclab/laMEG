"""
This module contains the unit tests for the `surf` module from the `lameg` package.
"""
import numpy as np

from lameg.surf import split_fv  # pylint: disable=no-name-in-module


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
