"""
This module contains the unit tests for the `surf` module from the `lameg` package.
"""
import copy
import os
import shutil
import subprocess
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import nibabel as nib
from scipy.sparse import csr_matrix

from lameg.surf import split_fv, mesh_adjacency, fix_non_manifold_edges, \
    find_non_manifold_edges, create_surf_gifti, _normit, mesh_normals, \
    remove_vertices, remove_unconnected_vertices, downsample_single_surface, \
    iterative_downsample_single_surface, downsample_multiple_surfaces, \
    combine_surfaces, interpolate_data, compute_dipole_orientations, \
    create_layer_mesh, postprocess_freesurfer_surfaces  # pylint: disable=no-name-in-module
from lameg.util import make_directory
# pylint: disable=C0302

def assert_sparse_equal(actual, expected):
    """
    Helper function to assert that two sparse matrices are equal.
    """
    assert (actual != expected).nnz == 0, "Sparse matrices are not equal"


@pytest.mark.parametrize("vectors, expected", [
    # Case with regular vectors
    (np.array([[3, 4, 0], [1, 1, 1]]),
     np.array([[0.6, 0.8, 0], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])),

    # Case with a zero vector
    (np.array([[0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0]])),

    # Case with very small values near the limit of floating point precision
    (np.array([[1e-20, 1e-20, 1e-20], [1e-20, 0, 0]]),
     np.array([[1e-20, 1e-20, 1e-20], [1e-20, 0, 0]])),

    # Single vector normal case
    (np.array([[9, 12, 0]]), np.array([[0.6, 0.8, 0]])),

    # Empty array case
    (np.array([]).reshape(0, 3), np.array([]).reshape(0, 3))
])
def test_normit(vectors, expected):
    """
    Tests the _normit function to ensure vectors are normalized correctly across various cases.
    This includes vectors of different magnitudes and handling of zero or near-zero magnitude
    vectors.
    """
    normalized_vectors = _normit(vectors)
    np.testing.assert_almost_equal(
        normalized_vectors,
        expected,
        decimal=5,
        err_msg="Vectors are not normalized correctly"
    )


@pytest.mark.parametrize("unit", [True, False])
def test_mesh_normals(unit):
    """
    Tests the mesh_normals function to ensure it correctly computes normals for vertices and faces
    from a mesh.This test checks both cases where normals are required to be unit length and where
    they are not.

    Parameters:
    unit (bool): Indicates whether the normals should be normalized to unit length.

    This function uses a simple pyramid mesh as the test case and verifies that the computed
    normals match expected values, which are predetermined for simplicity in this test scenario.
    """
    # Define vertices and faces for a simple pyramid mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])

    # Expected outcomes (mocked for demonstration)
    expected_vertex_normals = np.array([[-1., 1., -1.],
                                        [-0.57735026, 0.42264974, -1.5773503],
                                        [-1.5773503, -0.57735026, -1.5773503],
                                        [-1.5773503, 0.42264974, -0.57735026]])
    expected_face_normals = np.array([[-0., -0., -1],
                                      [-0., 1., -0.],
                                      [-0.57735026, -0.57735026, -0.57735026],
                                      [-1., -0., -0.]])

    if unit:
        expected_vertex_normals = _normit(expected_vertex_normals)
        expected_face_normals = _normit(expected_face_normals)

    # Call the function under test
    vertex_normals, face_normals = mesh_normals(vertices, faces, unit=unit)

    # Assert conditions
    np.testing.assert_array_almost_equal(vertex_normals, expected_vertex_normals)
    np.testing.assert_array_almost_equal(face_normals, expected_face_normals)


@pytest.mark.parametrize("vertices, faces, normals, expect_normals", [
    # Test case with normals
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
     np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
     np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
     True),

    # Test case without normals
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
     np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
     None,
     False)
])
def test_create_surf_gifti(vertices, faces, normals, expect_normals):
    """
    Tests the create_surf_gifti function to ensure that it correctly creates a GiftiImage object
    with the correct data arrays for vertices, faces, and optionally normals.
    """
    gifti_img = create_surf_gifti(vertices, faces, normals)

    # Verify that the created GiftiImage has the correct number of data arrays
    expected_num_arrays = 2 if normals is None else 3
    assert len(gifti_img.darrays) == expected_num_arrays,\
        "Incorrect number of data arrays in GiftiImage"

    # Check vertices
    np.testing.assert_array_equal(
        gifti_img.darrays[0].data,
        vertices,
        "Vertices data array not matched"
    )

    # Check faces
    np.testing.assert_array_equal(
        gifti_img.darrays[1].data,
        faces,
        "Faces data array not matched"
    )

    # Check normals if provided
    if expect_normals:
        np.testing.assert_array_equal(
            gifti_img.darrays[2].data,
            normals,
            "Normals data array not matched"
        )


@pytest.fixture
def connected_gifti():
    """
    Provides a GiftiImage fixture with all vertices connected to faces.

    This fixture represents a typical scenario where every vertex is part of at least one face,
    ensuring that no vertices should be removed during the processing of connected vertices.

    Returns:
    nibabel.gifti.GiftiImage: A GiftiImage object where all vertices are connected.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return create_surf_gifti(vertices, faces)


@pytest.fixture
def unconnected_gifti():
    """
    Provides a GiftiImage fixture with some vertices unconnected to any faces.

    This fixture is designed to test the removal functionality for unconnected vertices,
    simulating a scenario where one vertex out of several does not contribute to any face.

    Returns:
    nibabel.gifti.GiftiImage: A GiftiImage object with some unconnected vertices.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 2, 2]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return create_surf_gifti(vertices, faces)


# pylint: disable=W0621
def test_remove_unconnected_vertices_with_connected_gifti(connected_gifti):
    """
    Test the function with a Gifti surface where all vertices are connected.
    Ensures that no vertices are removed when all are connected to faces.
    """
    result = remove_unconnected_vertices(connected_gifti)
    assert result.darrays[0].data.shape[0] == 4


# pylint: disable=W0621
def test_remove_unconnected_vertices_with_unconnected_gifti(unconnected_gifti):
    """
    Test the function with a Gifti surface containing unconnected vertices.
    Verifies that unconnected vertices are correctly removed.
    """
    result = remove_unconnected_vertices(unconnected_gifti)
    assert len(result.darrays[0].data) == 4


def test_remove_unconnected_vertices_no_vertices():
    """
    Test the function on a Gifti surface with no vertices.
    Checks the function's handling of empty vertex data.
    """
    gifti = create_surf_gifti(np.zeros((0,3)), np.zeros((0,3)))
    result = remove_unconnected_vertices(gifti)
    assert result.darrays[0].data.shape[0] == 0


def test_remove_unconnected_vertices_no_faces():
    """
    Test the function on a Gifti surface with vertices but no faces.
    Ensures that all vertices are removed as none are connected.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0]])
    gifti = create_surf_gifti(vertices, np.zeros((0,3)))
    result = remove_unconnected_vertices(gifti)
    assert result.darrays[0].data.shape[0] == 0


def test_remove_unconnected_vertices_edge_case():
    """
    Test the function with a Gifti surface where all vertices are exactly connected.
    Confirms no unnecessary removal of connected vertices.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2]])
    gifti = create_surf_gifti(vertices, faces)
    result = remove_unconnected_vertices(gifti)
    assert len(result.darrays[0].data) == 3


@pytest.fixture
def sample_gifti():
    """
    Pytest fixture that creates a simple GiftiImage object suitable for testing.

    This fixture constructs a GiftiImage with a minimal set of vertices and faces that form
    a quadrilateral, using two triangles. It's used to provide a consistent test surface for
    functions that operate on GiftiImage objects.

    Returns:
        nibabel.gifti.GiftiImage: A GiftiImage object with predefined vertices and faces.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    return create_surf_gifti(vertices, faces, normals=normals)


# pylint: disable=W0621
def test_remove_no_vertices(sample_gifti):
    """
    Test `remove_vertices` with no vertices specified for removal.
    Verifies that the Gifti surface remains unchanged when the vertices_to_remove array is empty.
    """
    new_gifti = remove_vertices(sample_gifti, np.array([]))
    assert np.array_equal(new_gifti.darrays[0].data, sample_gifti.darrays[0].data)
    assert np.array_equal(new_gifti.darrays[1].data, sample_gifti.darrays[1].data)
    assert np.array_equal(new_gifti.darrays[2].data, sample_gifti.darrays[2].data)


# pylint: disable=W0621
def test_remove_specific_vertices(sample_gifti):
    """
    Test `remove_vertices` with a specific vertex removed.
    Checks that the correct vertex is removed and that the faces are correctly updated.
    """
    vertices_to_remove = np.array([1])
    new_gifti = remove_vertices(sample_gifti, vertices_to_remove)
    assert len(new_gifti.darrays[0].data) == 3  # Check vertices are reduced
    assert np.array_equal(
        new_gifti.darrays[0].data,
        np.array([[0., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    )
    assert np.array_equal(new_gifti.darrays[1].data, np.array([[0, 1, 2]]))
    assert np.array_equal(
        new_gifti.darrays[2].data,
        np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    )


# pylint: disable=W0621
def test_remove_all_vertices(sample_gifti):
    """
    Test `remove_vertices` with all vertices specified for removal.
    Confirms that the new Gifti surface contains no vertices or faces.
    """
    vertices_to_remove = np.arange(4)
    new_gifti = remove_vertices(sample_gifti, vertices_to_remove)
    assert len(new_gifti.darrays[0].data) == 0
    assert len(new_gifti.darrays[1].data) == 0
    assert len(new_gifti.darrays[2].data) == 0


# pylint: disable=W0621
def test_edge_case_removal(sample_gifti):
    """
    Test `remove_vertices` removing vertices that result in the removal of all faces.
    Verifies that faces are correctly deemed invalid and removed when their vertices are no longer
    present.
    """
    vertices_to_remove = np.array([0, 2])
    new_gifti = remove_vertices(sample_gifti, vertices_to_remove)
    assert len(new_gifti.darrays[1].data) == 0  # Assuming it removes one face


@pytest.mark.parametrize("faces, expected_result", [
    # No non-manifold edges: Simple triangle
    (np.array([[0, 1, 2]]), {}),

    # Complex case with multiple overlapping edges
    (np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 1, 2]]), {(1, 2): [0, 1, 3]}),
])
def test_find_non_manifold_edges(faces, expected_result):
    """
    Tests the find_non_manifold_edges function to ensure it correctly identifies non-manifold edges
    in various mesh configurations.

    The function should accurately map edges shared by more than two faces, crucial for detecting
    complex topological errors in 3D meshes.
    """
    result = find_non_manifold_edges(faces)
    assert result == expected_result, f"Expected {expected_result}, got {result}"


@pytest.mark.parametrize("vertices, faces, expected_faces", [
    # Case 1: Single triangle, no non-manifold edges
    (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
     np.array([[0, 1, 2]]),
     np.array([[0, 1, 2]])),

    # Case 3: Complex mesh with one non-manifold edge
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]]),
     np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 1, 2]]),
     np.array([[2, 3, 4]])),

    # Case 4: Mesh without any faces (should handle gracefully)
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
     np.array([]),
     np.array([]))
])
def test_fix_non_manifold_edges(vertices, faces, expected_faces):
    """
    Tests the fix_non_manifold_edges function to ensure it correctly removes non-manifold edges
    from a mesh.

    Each test case verifies that the function can handle various mesh configurations, including
    simple and complex geometries, and meshes without any non-manifold conditions, to assert the
    correct behavior of face removal or retention based on non-manifold edge detection.

    Parameters:
    vertices (np.ndarray): The array of vertices of the mesh.
    faces (np.ndarray): The array of faces of the mesh.
    expected_faces (np.ndarray): The expected array of faces after removing non-manifold edges.
    """
    _, actual_faces = fix_non_manifold_edges(vertices, faces)
    np.testing.assert_array_equal(actual_faces, expected_faces)


@pytest.fixture
def large_gifti():
    """
    Provides a large GiftiImage fixture with a dense mesh of vertices and faces.

    This fixture is designed for testing functions that operate on complex, high-resolution
    neuroimaging surface data. It simulates a realistic scenario where a mesh might include
    thousands of vertices and faces, typical of high-resolution brain scans.

    Returns:
    nibabel.gifti.GiftiImage: A GiftiImage object representing a dense mesh.
    """
    # Create a large number of vertices arranged in a grid
    num_vertices_side = 100  # Creates a 100x100 grid of vertices
    vertices = np.array([[x_coord, y_coord, np.sin(x_coord) * np.cos(y_coord)]
                         for x_coord in range(num_vertices_side)
                         for y_coord in range(num_vertices_side)])
    normals = np.array([[x_coord, y_coord, -1*np.sin(x_coord) + 3*np.cos(y_coord)]
                         for x_coord in range(num_vertices_side)
                         for y_coord in range(num_vertices_side)])

    # Create faces using the grid of vertices
    faces = []
    for x_idx in range(num_vertices_side - 1):
        for y_idx in range(num_vertices_side - 1):
            # Each square in the grid is divided into two triangles
            top_left = x_idx * num_vertices_side + y_idx
            top_right = top_left + 1
            bottom_left = top_left + num_vertices_side
            bottom_right = bottom_left + 1

            faces.append([top_left, bottom_left, top_right])  # First triangle
            faces.append([bottom_left, bottom_right, top_right])  # Second triangle

    # Convert lists to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)
    normals = np.array(normals)

    # Create the GiftiImage
    gii = create_surf_gifti(vertices, faces, normals=normals)

    return gii


# pylint: disable=W0621
def test_downsample_single_surface(large_gifti):
    """
    Test the downsampling of a single surface represented as a Gifti file.

    This function tests whether the `downsample_single_surface` function accurately reduces
    the number of vertices and faces in a Gifti surface mesh according to a specified downsampling
    factor. It verifies the new size of the vertex and face arrays and checks the first few entries
    for correctness against predefined targets.

    Parameters:
    - large_gifti (GiftiImage): A GiftiImage object that represents a 3D surface mesh.

    Asserts:
    - The number of vertices in the first data array after downsampling is exactly 3224.
    - The number of vertices in the second data array after downsampling is exactly 1960.
    - The content of the first ten entries of the vertex and face arrays match the predefined
      target arrays, confirming that the downsampling and data integrity are preserved.

    This function will raise an AssertionError if the conditions are not met, indicating a
    potential issue in the `downsample_single_surface` function or the input data.
    """
    ds_gifti = downsample_single_surface(large_gifti, 0.1)

    assert ds_gifti.darrays[0].data.shape[0] == 3224
    assert ds_gifti.darrays[1].data.shape[0] == 1960
    assert ds_gifti.darrays[2].data.shape[0] == 3224
    target = np.array([[ 0.,  1.,  0.], [ 0.,  6.,  0.], [ 0.,  9., -0.], [ 0., 21., -0.],
                       [ 0., 31.,  0.], [ 0., 41., -0.], [ 0., 43.,  0.], [ 0., 53., -0.],
                       [ 0., 68.,  0.], [ 0., 75.,  0.]])
    assert np.allclose(ds_gifti.darrays[0].data[:10,:], target)
    target = np.array([[  12,   13,    0], [1834,    1, 1717], [  16,   27, 1004],
                       [1628,    3, 1627], [2851,   60, 2818], [  60, 1805, 2818],
                       [1361, 2618, 2387], [1001,   17,    4], [2074,   37,  998],
                       [  37, 1620,  998]])
    assert np.allclose(ds_gifti.darrays[1].data[:10,:], target)
    target = np.array([[ 0., 1., 1.620907 ], [ 0., 6., 2.8805108], [ 0., 9., -2.7333908],
                       [ 0., 21., -1.6431878], [ 0., 31., 2.7442272]])
    assert np.allclose(ds_gifti.darrays[2].data[:5, :], target)


# pylint: disable=W0621
def test_iterative_downsample_single_surface(large_gifti):
    """
    Test the iterative downsampling process on a single surface represented as a Gifti file.

    This function tests the `iterative_downsample_single_surface` function to ensure it correctly
    applies an iterative downsampling algorithm to reduce the complexity of a 3D surface mesh
    while maintaining the integrity of the data structure. The test checks if the output dimensions
    and specific data points of the vertex and face arrays conform to expected values after
    downsampling.

    Parameters:
    - large_gifti (GiftiImage): A GiftiImage object that represents a 3D surface mesh to be
      downsampled.

    Asserts:
    - The number of vertices in the first data array after downsampling should be exactly 3224.
    - The number of vertices in the second data array after downsampling should be exactly 1960.
    - The first ten entries of the vertex and face arrays are checked against predefined target
      arrays, ensuring that the iterative downsampling process preserves data fidelity and
      accuracy.

    Raises:
    - AssertionError: If any of the conditions are not met, an AssertionError is raised, indicating
      a potential issue in the iterative downsampling process or the initial data.
    """
    ds_gifti = iterative_downsample_single_surface(large_gifti, 0.1)

    assert ds_gifti.darrays[0].data.shape[0] == 1202
    assert ds_gifti.darrays[1].data.shape[0] == 759
    target = np.array([[ 0., 9., -0.], [ 0., 21., -0.], [ 0., 43., 0.], [ 0., 68., 0.],
                       [ 0., 75., 0.], [ 1., 0., 0.84147096], [ 1., 3., -.83305],
                       [ 1., 12., 0.71008], [1., 65., -0.47329], [1., 94., 0.81577]])
    assert np.allclose(ds_gifti.darrays[0].data[:10,:], target)
    target = np.array([[ 603,    1,  602], [  17,  656, 1015], [ 501,  941,  865],
                       [ 596, 1153,    8], [ 610,   11,    4], [ 771, 1117, 1118],
                       [ 945,  946,  947], [ 750,  951,  907], [ 972,  670,    0],
                       [ 752,  678,    7]])
    assert np.allclose(ds_gifti.darrays[1].data[:10,:], target)


# pylint: disable=W0621
def test_downsample_multiple_surfaces(large_gifti):
    """
    Test the downsampling of multiple surface meshes represented as Gifti files.

    This function performs a test to ensure the functionality of the `downsample_multiple_surfaces`
    function, which is intended to downsample a list of surface meshes. The test involves modifying
    the z-coordinates of the vertices of a copy of the input mesh to create a distinct but related
    surface mesh. Both the original and modified meshes are then downsampled, and various
    assertions are performed to check data integrity and consistency between the two downsampled
    surfaces.

    Parameters:
    - large_gifti (GiftiImage): A GiftiImage object that represents a 3D surface mesh.

    Processes:
    - Copies the vertex data from the original surface and modifies the z-coordinates.
    - Creates a new GiftiImage with the modified vertices to simulate a different but related
      surface.
    - Applies downsampling to both the original and modified surfaces.
    - Checks the vertex data and face data of the resulting downsampled surfaces against expected
      targets.

    Asserts:
    - The first five vertices of the downsampled surfaces match the expected target values.
    - The first five faces of the downsampled original surface match expected target values.
    - The number of vertices in the downsampled surfaces are equal, ensuring consistent
      downsampling.
    - The face data of both downsampled surfaces are identical, asserting data integrity
      post-downsampling.

    Raises:
    - AssertionError: If any of the assertions fail, indicating discrepancies in the downsampling
      process or input modifications.
    """
    verts2 = copy.copy(large_gifti.darrays[0].data)
    verts2[:,2] = verts2[:,2]+5
    large_gifti2 = create_surf_gifti(
        verts2,
        large_gifti.darrays[1].data,
        normals=large_gifti.darrays[2].data
    )

    ds_surfs = downsample_multiple_surfaces([large_gifti, large_gifti2], 0.1)

    target = np.array([[ 0.,  9., -0.], [ 0., 21., -0.], [ 0., 43.,  0.], [ 0., 68.,  0.],
                       [ 0., 75.,  0.]])
    assert np.allclose(ds_surfs[0].darrays[0].data[:5,:], target)

    target = np.array([[ 0.,  9.,  5.], [ 0., 21.,  5.], [ 0., 43.,  5.], [ 0., 68.,  5.],
                       [ 0., 75.,  5.]])
    assert np.allclose(ds_surfs[1].darrays[0].data[:5, :], target)

    target = np.array([[ 603,    1,  602], [  17,  656, 1015], [ 501,  941,  865],
                       [ 596, 1153,    8], [ 610,   11,    4]])
    assert np.allclose(ds_surfs[0].darrays[1].data[:5,:], target)

    target = np.array([[ 0., 9., -2.7333908], [ 0., 21., -1.6431878], [ 0., 43., 1.66534  ],
                       [ 0., 68., 1.3204291], [ 0., 75., 2.7652538]])
    assert np.allclose(ds_surfs[0].darrays[2].data[:5, :], target)

    target = np.array([[0., 9., -2.7333908], [0., 21., -1.6431878], [0., 43., 1.66534],
                       [0., 68., 1.3204291], [0., 75., 2.7652538]])
    assert np.allclose(ds_surfs[1].darrays[2].data[:5, :], target)

    assert ds_surfs[0].darrays[0].data.shape[0] == ds_surfs[1].darrays[0].data.shape[0]
    assert np.allclose(ds_surfs[0].darrays[1].data, ds_surfs[1].darrays[1].data)


# pylint: disable=W0621
def test_combine_surfaces(large_gifti):
    """
    Test the combination of multiple surface meshes into a single surface mesh.

    This function evaluates the functionality of the `combine_surfaces` function, which is designed
    to merge multiple Gifti surface meshes into a single composite surface. It involves creating a
    second surface by modifying the z-coordinates of the vertex data from the original mesh, then
    combining this modified mesh with the original mesh. The function tests if the vertices and
    faces of the resulting combined surface correctly concatenate the data from the original and
    modified surfaces.

    Parameters:
    - large_gifti (GiftiImage): A GiftiImage object representing a 3D surface mesh.

    Processes:
    - Copies the vertex data from the original surface and modifies the z-coordinates to create a
      variation.
    - Constructs a new GiftiImage with these modified vertices.
    - Combines the original and modified meshes into a single composite mesh.
    - Validates that the vertex and face data in the combined surface mesh matches the concatenated
      data of the two individual meshes.

    Asserts:
    - The vertex data of the combined surface exactly matches the concatenated vertex data of the
      original and modified surfaces.
    - The face data of the combined surface exactly matches the concatenated face data of the
      original and modified surfaces.

    Raises:
    - AssertionError: If any of the assertions fail, it indicates an error in the surface
      combination process or an inconsistency in data manipulation prior to combining.
    """
    verts2 = copy.copy(large_gifti.darrays[0].data)
    verts2[:, 2] = verts2[:, 2] + 5
    large_gifti2 = create_surf_gifti(
        verts2,
        large_gifti.darrays[1].data,
        normals=large_gifti.darrays[2].data
    )

    combined_surf = combine_surfaces([large_gifti, large_gifti2])

    target = np.vstack([
        large_gifti.darrays[0].data,
        large_gifti2.darrays[0].data
    ])
    assert np.allclose(combined_surf.darrays[0].data, target)

    target = np.vstack([
        large_gifti.darrays[1].data,
        large_gifti2.darrays[1].data+large_gifti.darrays[0].data.shape[0]
    ])
    assert np.allclose(combined_surf.darrays[1].data, target)

    target = np.vstack([
        large_gifti.darrays[2].data,
        large_gifti2.darrays[2].data
    ])
    assert np.allclose(combined_surf.darrays[2].data, target)


# pylint: disable=W0621
def test_compute_dipole_orientations(large_gifti):
    """
    Test the compute_dipole_orientations function for different methods to ensure correct
    calculation of dipole orientations across various scenarios.

    Steps:
    1. Create two slightly different surface files by modifying vertex coordinates.
    2. Downsample both surfaces and save them.
    3. Compute dipole orientations for the 'link_vector', 'ds_surf_norm', 'orig_surf_norm',
       and 'cps' methods.
    4. Verify the orientations against predefined expected outcomes for each method.

    The test checks:
    - That the function handles the creation of link vectors correctly by comparing the
      computed vectors against expected ones for identical configurations (fixed=True).
    - That the surface normal calculations are correct for downsampled and original surfaces
      under the conditions of a fixed orientation across layers.
    - That the orientation results are close to the expected values, indicating correct
      functionality of the method computations and the handling of inputs and surface
      manipulations.

    This test helps confirm the robustness of the orientation computation under a variety of
    typical usage scenarios and ensures that any changes to the underlying implementation
    maintain the expected behavior.
    """
    _ = make_directory('./', ['output'])

    nib.save(large_gifti, os.path.join('./output/white.gii'))

    verts2 = copy.copy(large_gifti.darrays[0].data)
    verts2[:, 2] = verts2[:, 2] + 5
    large_gifti2 = create_surf_gifti(verts2, large_gifti.darrays[1].data)
    nib.save(large_gifti2, os.path.join('./output/pial.gii'))

    ds_surfs = downsample_multiple_surfaces([large_gifti2, large_gifti], 0.1)
    nib.save(ds_surfs[0], './output/pial.ds.gii')
    nib.save(ds_surfs[1], './output/white.ds.gii')

    normals = compute_dipole_orientations(
        'link_vector',
        ['pial','white'],
        './output',
        fixed=True
    )
    target = np.array([[ 0.,  0., -1.], [ 0.,  0., -1.], [ 0.,  0. ,-1.], [ 0.,  0., -1.],
                       [ 0.,  0., -1.], [ 0.,  0., -1.], [ 0.,  0., -1.], [ 0.,  0., -1.],
                       [ 0.,  0., -1.], [ 0.,  0., -1.]])
    assert np.allclose(normals[0,:10,:], target)
    assert np.allclose(normals[1,:10,:], target)

    normals = compute_dipole_orientations(
        'ds_surf_norm',
        ['pial', 'white'],
        './output',
        fixed=True
    )

    target = np.array([[-0.04698259,  0.1540498,  -0.98694545],
                       [-0.07900448,  0.12416247, -0.9891117 ],
                       [ 0.05659738,  0.10748,    -0.99259496],
                       [-0.11056121, -0.07688656, -0.99089086],
                       [-0.04565653,  0.14132655, -0.9889096 ]])
    assert np.allclose(normals[0,:5,:], target)
    assert np.allclose(normals[1, :5, :], target)

    normals = compute_dipole_orientations(
        'orig_surf_norm',
        ['pial', 'white'],
        './output',
        fixed=True
    )

    target = np.array([[ 0.4929678,  -0.1130688, -0.8626692 ],
                       [-0.4485411,  -0.16056265, -0.87922156],
                       [-0.16060773, -0.21938397 ,-0.9623283 ],
                       [ 0.16595066,  0.21885467, -0.961542  ],
                       [ 0.08390685,  0.22491537, -0.97075886]])
    assert np.allclose(normals[0,:5,:], target)
    assert np.allclose(normals[1, :5, :], target)

    normals = compute_dipole_orientations(
        'cps',
        ['pial', 'white'],
        './output',
        fixed=True
    )

    target = np.array([[ 1.8139431e-01, -2.7962229e-01 ,-9.4281888e-01],
                       [-1.0353364e-02, -7.2763674e-02, -9.9729550e-01],
                       [ 3.1780198e-04, -1.1128817e-01 ,-9.9378812e-01],
                       [-1.8438324e-02, -1.2312002e-02, -9.9975419e-01],
                       [ 7.1880430e-02,  1.5277593e-01, -9.8564333e-01]])
    assert np.allclose(normals[0,:5,:], target)
    assert np.allclose(normals[1,:5,:], target)


def test_create_layer_mesh():
    """
    Test the `create_layer_mesh` function for correct mesh file creation or retrieval.

    This test simulates different cortical layer specifications and hemisphere selections
    within a mocked FreeSurfer environment. It evaluates whether the function correctly:
    - Returns proper mesh identifiers for boundary cortical layers (0 for 'white' and 1 for
      'pial').
    - Formats and returns the name for intermediate layers accurately.
    - Triggers the creation of mesh files using `mris_expand` when required.

    Mocks:
        os.path.exists: Simulated to always return False to enforce the condition that mesh files
                        do not exist, which should trigger the `mris_expand` command.
        subprocess.run: Mocked to prevent actual execution of the `mris_expand` command, allowing
                        us to test that it is called correctly without performing any real file
                        operations.

    The function is tested under the following conditions:
    - Intermediate layer values (e.g., 0.5), to check for correct formatted string return and
      subprocess invocation.
    - Boundary layer values (0 and 1), to ensure correct identifiers are returned without
      subprocess invocation.
    - Invalid layer values (e.g., -0.1, 1.1), to confirm the function returns None as expected.

    Assertions:
        - Assert that the result for intermediate layers matches the formatted layer name.
        - Confirm that subprocess.run is called the correct number of times with the expected
          arguments.
        - Validate that boundary layers return 'white' or 'pial', and invalid inputs return None.
    """
    # Setup test parameters
    layer = 0.5
    hemispheres = ['lh', 'rh']
    fs_subject_dir = '/fake/freesurfer/subjects/subject1'

    # Mocking os.path.exists to always return False
    # This implies that the mesh files do not exist and will trigger mris_expand
    with patch('os.path.exists', return_value=False):
        # Mocking subprocess.run to simulate mris_expand without actually calling it
        with patch('subprocess.run') as mock_run:
            result = create_layer_mesh(layer, hemispheres, fs_subject_dir)

            # Asserting correct behavior
            assert result == '0.500', "Layer name should be formatted to three decimal places"
            # Ensure subprocess.run was called twice (once for each hemisphere)
            assert mock_run.call_count == 2
            # Check if subprocess.run was called with the expected command
            for hemi in hemispheres:
                expected_call = [
                    'mris_expand',
                    '-thickness',
                    f'/fake/freesurfer/subjects/subject1/surf/{hemi}.white',
                    f'{layer}',
                    f'/fake/freesurfer/subjects/subject1/surf/{hemi}.0.500'
                ]
                mock_run.assert_any_call(expected_call, check=True)

    # Test edge cases for layers 0 and 1
    assert create_layer_mesh(1, hemispheres, fs_subject_dir) == 'pial'
    assert create_layer_mesh(0, hemispheres, fs_subject_dir) == 'white'
    assert create_layer_mesh(-0.1, hemispheres, fs_subject_dir) is None
    assert create_layer_mesh(1.1, hemispheres, fs_subject_dir) is None


def test_postprocess_freesurfer_surfaces():
    """
    Test the `postprocess_freesurfer_surfaces` function to ensure it processes surface meshes
    correctly.

    This test checks that the function handles:
    - Environment setup and surface mesh creation through the `create_layer_mesh` function.
    - Proper handling of subprocess calls for `mris_info` and other commands.
    - Interaction with the filesystem for reading and writing files.

    Mocks:
    - `os.getenv` to provide a fake SUBJECTS_DIR environment variable.
    - `subprocess.Popen` to simulate the output of the `mris_info` command.
    - `create_layer_mesh` to avoid actual mesh creation and subprocess calls.
    - File operations like `nibabel.load` and `nibabel.save` to handle read/write without actual
      files.

    The test is run with a typical set of input parameters and uses a fixture to set up a temporary
    directory structure.
    """
    subj_id = 'sub-104'
    out_dir = './output'
    shutil.copy('./test_data/sub-104/surf/lh.pial.gii', './output/lh.pial.gii')
    shutil.copy('./test_data/sub-104/surf/lh.white.gii', './output/lh.white.gii')
    shutil.copy('./test_data/sub-104/surf/lh.white.gii', './output/lh.inflated.gii')
    shutil.copy('./test_data/sub-104/surf/rh.pial.gii', './output/rh.pial.gii')
    shutil.copy('./test_data/sub-104/surf/rh.white.gii', './output/rh.white.gii')
    shutil.copy('./test_data/sub-104/surf/rh.white.gii', './output/rh.inflated.gii')
    out_fname = 'processed_surface.gii'

    # Setup the Popen mock
    mock_process = MagicMock()
    mock_process.communicate.return_value = (b'0.1 0.2 0.3', b'')

    # Mock the Popen context manager
    mock_popen = MagicMock()
    mock_popen.return_value.__enter__.return_value = mock_process
    mock_popen.return_value.__exit__.return_value = None

    # Mock for subprocess.run()
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0

    # Define side_effect function
    # pylint: disable=W0613
    def create_layer_mesh_side_effect(layer, hemispheres, fs_subject_dir):
        if layer == 1:
            return 'pial'
        if 0 < layer < 1:
            return f'{layer:.3f}'
        if layer == 0:
            return 'white'
        return None

    # Mock environment and external functions
    with patch('os.getenv', return_value='./test_data/fs'), \
            patch('subprocess.Popen', mock_popen), \
            patch('subprocess.run', mock_run), \
            patch('lameg.surf.create_layer_mesh', side_effect=create_layer_mesh_side_effect):

        # Call function
        postprocess_freesurfer_surfaces(subj_id, out_dir, out_fname, n_surfaces=2)

        # Ensure `create_layer_mesh` is called correctly
        assert mock_popen.call_count == 1, \
            "Expected subprocess.Popen to be called once for mris_info"
        mock_popen.assert_called_with(
            f"mri_info --cras {os.path.join('./test_data/fs', subj_id, 'mri', 'orig.mgz')}",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        final_surf = nib.load(os.path.join(out_dir, out_fname))
        target = np.array([[ -4.945391,  -57.815586,   28.967335 ],
                           [ -5.680048,  -57.90388,    27.428965 ],
                           [ -6.0047097, -57.829296,   25.688501 ],
                           [ -6.5609765, -57.508076,   24.787767 ],
                           [-11.259755,  -57.880123,   20.450663 ]])
        assert np.allclose(final_surf.darrays[0].data[:5,:], target)

        target = np.array([[    0,    10,    28],
                           [    0,    28,     8],
                           [   11,     0,     8],
                           [    0,    11, 46271],
                           [   10,    13, 46894]])
        assert np.allclose(final_surf.darrays[1].data[:5, :], target)

        target = np.array([[ 0.14771064,  0.9553547,  -0.25588843],
                           [ 0.22574118,  0.97246087, -0.05797116],
                           [ 0.28987604,  0.95356506,  0.08176449],
                           [ 0.4109391,   0.89564127,  0.17016393],
                           [-0.06476931,  0.996889,   -0.04491389]])
        assert np.allclose(final_surf.darrays[2].data[:5, :], target)



@pytest.mark.parametrize("faces, expected", [
    (np.array([[0, 1, 2]]), csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))),
    (np.array([[0, 1, 2], [2, 3, 4]]), csr_matrix(np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])))
])
def test_mesh_adjacency(faces, expected):
    """
    Tests the fix_non_manifold_edges function to ensure it correctly removes non-manifold edges
    from a mesh.

    Each test case verifies that the function can handle various mesh configurations, including
    simple and complex geometries, and meshes without any non-manifold conditions, to assert the
    correct behavior of face removal or retention based on non-manifold edge detection.

    Parameters:
    vertices (np.ndarray): The array of vertices of the mesh.
    faces (np.ndarray): The array of faces of the mesh.
    expected_faces (np.ndarray): The expected array of faces after removing non-manifold edges.
    """
    result = mesh_adjacency(faces)
    assert_sparse_equal(result, expected)


# pylint: disable=W0621
def test_interpolate_data(large_gifti):
    """
    Tests the interpolate_data function by comparing the interpolated results
    from a downsampled GIFTI surface with known target values.

    Parameters:
    large_gifti (nibabel.gifti.GiftiImage): The original high-resolution GIFTI surface image.

    Steps:
    1. Downsamples the large_gifti surface to 10% of its vertices.
    2. Creates random data for the downsampled surface.
    3. Interpolates the data back onto the large_gifti surface.
    4. Asserts that the interpolated data closely matches a predefined target array
       for the first 10 vertices.
    """
    smaller_gifti = downsample_single_surface(large_gifti, 0.1)
    n_verts = smaller_gifti.darrays[0].data.shape[0]
    ds_data = np.linspace(0,100, n_verts)
    interp_data = interpolate_data(large_gifti, smaller_gifti, ds_data)
    target = np.array([3.03420237e+01, 5.32733478e+01, 5.32733478e+01, 6.74284478e+01,
                       7.44034286e+01, 3.72172278e+01, 3.10269935e-02, 3.10269935e-02,
                       4.65404902e-02, 6.20539870e-02])
    assert np.allclose(interp_data[:10], target)


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
    faces = np.array([[0, 1, 2], [2, 4, 5], [6, 7, 2], [8, 9, 10]])
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
    target = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 2]])
    assert np.allclose(patch_1_faces, target)

    patch_1_vertices = split_patches[0]['vertices']
    target = np.array([[2, 4], [2, 8], [8, 4], [0, 4], [2, 6], [2, 2], [4, 2]])
    assert np.allclose(patch_1_vertices, target)

    patch_2_faces = split_patches[1]['faces']
    target = np.array([[0, 1, 2]])
    assert np.allclose(patch_2_faces, target)

    patch_2_vertices = split_patches[1]['vertices']
    target = np.array([[4, 0], [5, 2], [5, 0]])
    assert np.allclose(patch_2_vertices, target)

    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 0], [6, 7, 8], [10, 9, 3], [0, 5, 6]])
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
