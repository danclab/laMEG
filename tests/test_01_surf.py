"""
This module contains the unit tests for the `surf` module from the `lameg` package.
"""
import copy
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from lameg.surf import _split_connected_components, _mesh_adjacency, _fix_non_manifold_edges, \
    _find_non_manifold_edges, _create_surf_gifti, _normit, _vertex_normal_vectors, \
    _remove_vertices, _remove_unconnected_vertices, _downsample_single_surface, \
    _iterative_downsample_single_surface, _concatenate_surfaces, interpolate_data, \
    convert_fsaverage_to_native, convert_native_to_fsaverage, \
    LayerSurfaceSet  # pylint: disable=no-name-in-module


# pylint: disable=C0302


def assert_sparse_equal(actual, expected):
    """
    Helper function to assert that two sparse matrices are equal.
    """
    assert (actual != expected).nnz == 0, "Sparse matrices are not equal"


@pytest.mark.parametrize("vectors, expected", [
    # Case with regular vectors
    (np.array([[3, 4, 0], [1, 1, 1]]),
     np.array([[0.6, 0.8, 0], [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]])),

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
def test_vertex_normal_vectors(unit):
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
    vertex_normals, face_normals = _vertex_normal_vectors(vertices, faces, unit=unit)

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
    gifti_img = _create_surf_gifti(vertices, faces, normals)

    # Verify that the created GiftiImage has the correct number of data arrays
    expected_num_arrays = 2 if normals is None else 3
    assert len(gifti_img.darrays) == expected_num_arrays, \
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
    return _create_surf_gifti(vertices, faces)


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
    return _create_surf_gifti(vertices, faces)


# pylint: disable=W0621
def test_remove_unconnected_vertices_with_connected_gifti(connected_gifti):
    """
    Test the function with a Gifti surface where all vertices are connected.
    Ensures that no vertices are removed when all are connected to faces.
    """
    result = _remove_unconnected_vertices(connected_gifti)
    assert result.darrays[0].data.shape[0] == 4


# pylint: disable=W0621
def test_remove_unconnected_vertices_with_unconnected_gifti(unconnected_gifti):
    """
    Test the function with a Gifti surface containing unconnected vertices.
    Verifies that unconnected vertices are correctly removed.
    """
    result = _remove_unconnected_vertices(unconnected_gifti)
    assert len(result.darrays[0].data) == 4


def test_remove_unconnected_vertices_no_vertices():
    """
    Test the function on a Gifti surface with no vertices.
    Checks the function's handling of empty vertex data.
    """
    gifti = _create_surf_gifti(np.zeros((0, 3)), np.zeros((0, 3)))
    result = _remove_unconnected_vertices(gifti)
    assert result.darrays[0].data.shape[0] == 0


def test_remove_unconnected_vertices_no_faces():
    """
    Test the function on a Gifti surface with vertices but no faces.
    Ensures that all vertices are removed as none are connected.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0]])
    gifti = _create_surf_gifti(vertices, np.zeros((0, 3)))
    result = _remove_unconnected_vertices(gifti)
    assert result.darrays[0].data.shape[0] == 0


def test_remove_unconnected_vertices_edge_case():
    """
    Test the function with a Gifti surface where all vertices are exactly connected.
    Confirms no unnecessary removal of connected vertices.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2]])
    gifti = _create_surf_gifti(vertices, faces)
    result = _remove_unconnected_vertices(gifti)
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
    return _create_surf_gifti(vertices, faces, normals=normals)


# pylint: disable=W0621
def test_remove_no_vertices(sample_gifti):
    """
    Test `remove_vertices` with no vertices specified for removal.
    Verifies that the Gifti surface remains unchanged when the vertices_to_remove array is empty.
    """
    new_gifti = _remove_vertices(sample_gifti, np.array([]))
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
    new_gifti = _remove_vertices(sample_gifti, vertices_to_remove)
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
    new_gifti = _remove_vertices(sample_gifti, vertices_to_remove)
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
    new_gifti = _remove_vertices(sample_gifti, vertices_to_remove)
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
    result = _find_non_manifold_edges(faces)
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
    _, actual_faces = _fix_non_manifold_edges(vertices, faces)
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
    normals = np.array([[x_coord, y_coord, -1 * np.sin(x_coord) + 3 * np.cos(y_coord)]
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
    gii = _create_surf_gifti(vertices, faces, normals=normals)

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
    ds_gifti = _downsample_single_surface(large_gifti, 0.1)

    assert ds_gifti.darrays[0].data.shape[0] == 3224
    assert ds_gifti.darrays[1].data.shape[0] == 1960
    assert ds_gifti.darrays[2].data.shape[0] == 3224
    target = np.array([[0., 1., 0.], [0., 6., 0.], [0., 9., -0.], [0., 21., -0.],
                       [0., 31., 0.], [0., 41., -0.], [0., 43., 0.], [0., 53., -0.],
                       [0., 68., 0.], [0., 75., 0.]])
    assert np.allclose(ds_gifti.darrays[0].data[:10, :], target)
    target = np.array([[12, 13, 0], [1834, 1, 1717], [16, 27, 1004],
                       [1628, 3, 1627], [2851, 60, 2818], [60, 1805, 2818],
                       [1361, 2618, 2387], [1001, 17, 4], [2074, 37, 998],
                       [37, 1620, 998]])
    assert np.allclose(ds_gifti.darrays[1].data[:10, :], target)
    target = np.array([[0., 1., 1.620907], [0., 6., 2.8805108], [0., 9., -2.7333908],
                       [0., 21., -1.6431878], [0., 31., 2.7442272]])
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
    ds_gifti = _iterative_downsample_single_surface(large_gifti, 0.1)

    assert ds_gifti.darrays[0].data.shape[0] == 1202
    assert ds_gifti.darrays[1].data.shape[0] == 759
    target = np.array([[0., 9., -0.], [0., 21., -0.], [0., 43., 0.], [0., 68., 0.],
                       [0., 75., 0.], [1., 0., 0.84147096], [1., 3., -.83305],
                       [1., 12., 0.71008], [1., 65., -0.47329], [1., 94., 0.81577]])
    assert np.allclose(ds_gifti.darrays[0].data[:10, :], target)
    target = np.array([[603, 1, 602], [17, 656, 1015], [501, 941, 865],
                       [596, 1153, 8], [610, 11, 4], [771, 1117, 1118],
                       [945, 946, 947], [750, 951, 907], [972, 670, 0],
                       [752, 678, 7]])
    assert np.allclose(ds_gifti.darrays[1].data[:10, :], target)


# pylint: disable=W0621
def test_concatenate_surfaces(large_gifti):
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
    large_gifti2 = _create_surf_gifti(
        verts2,
        large_gifti.darrays[1].data,
        normals=large_gifti.darrays[2].data
    )

    combined_surf = _concatenate_surfaces([large_gifti, large_gifti2])

    target = np.vstack([
        large_gifti.darrays[0].data,
        large_gifti2.darrays[0].data
    ])
    assert np.allclose(combined_surf.darrays[0].data, target)

    target = np.vstack([
        large_gifti.darrays[1].data,
        large_gifti2.darrays[1].data + large_gifti.darrays[0].data.shape[0]
    ])
    assert np.allclose(combined_surf.darrays[1].data, target)

    target = np.vstack([
        large_gifti.darrays[2].data,
        large_gifti2.darrays[2].data
    ])
    assert np.allclose(combined_surf.darrays[2].data, target)


@pytest.mark.dependency()
def test_create_layer_surface_set():
    """
    Test the `create` function of `LayerSurfaceSet` to ensure it processes surface meshes
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

    # Helper to mock mri_info matrices
    def _mri_info_side_effect(args, *_, **__):
        # args is like ['mri_info', '--vox2ras-tkr', '<.../orig.mgz>']
        flag = args[1]
        if flag in ('--vox2ras-tkr', '--vox2ras'):
            if flag == '--vox2ras-tkr':
                mat = np.array([
                    [-0.8, 0.0, 0.0, 128.0],
                    [0.0, 0.0, 0.8, -128.0],
                    [0.0, -0.8, 0.0, 128.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]).reshape(-1)
            else:
                mat = np.array([
                    [-0.8, 0.0, 0.0, 130.02538],
                    [0.0, 0.0, 0.8, -91.08466],
                    [0.0, -0.8, 0.0, 144.88278],
                    [0.0, 0.0, 0.0, 1.0]
                ]).reshape(-1)
            return (' '.join(str(x) for x in mat)).encode('utf-8')
        raise RuntimeError(f'unexpected mri_info flag: {flag}')

    # Mock for subprocess.run()
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0

    error_raised = False
    try:
        surf_set = LayerSurfaceSet(subj_id, 2)
        surf_set.create()
    except EnvironmentError:
        error_raised = True
    assert error_raised

    def mock_which(cmd):
        if cmd in ["mris_convert", "mris_inflate", "mri_info"]:
            return f"/usr/local/freesurfer/bin/{cmd}"
        return None

    # Mock environment and external functions
    with patch("shutil.which", side_effect=mock_which), \
            patch('os.getenv', return_value='./test_data/fs'), \
            patch('subprocess.check_output', side_effect=_mri_info_side_effect) as mock_chk, \
            patch('subprocess.run', mock_run):

        surf_set = LayerSurfaceSet(subj_id, 2)
        surf_set.create()

        # Assert we asked for the two matrices
        assert mock_chk.call_count == 2
        flags = [ca[0][0][1] for ca in mock_chk.call_args_list]
        paths = [ca[0][0][2] for ca in mock_chk.call_args_list]
        assert '--vox2ras-tkr' in flags and '--vox2ras' in flags
        assert all(p == os.path.join('./test_data/fs', subj_id, 'mri', 'orig.mgz') for p in paths)

        final_surf = surf_set.load(stage='ds', orientation='link_vector', fixed=True)

        assert final_surf.darrays[0].data.shape[0] == 99418
        assert final_surf.darrays[1].data.shape[0] == 156346
        assert final_surf.darrays[2].data.shape[0] == 99418

        target = np.array([[-5.045371, -58.015545, 28.667315],
                           [-5.780028, -58.10384, 27.128946],
                           [-6.1046896, -58.029255, 25.388483],
                           [-6.6609564, -57.708035, 24.487747],
                           [-11.3597355, -58.080082, 20.150642]])
        assert np.allclose(final_surf.darrays[0].data[:5, :], target)

        target = np.array([[0, 10, 28],
                           [0, 28, 8],
                           [11, 0, 8],
                           [0, 11, 46254],
                           [10, 13, 46877]])
        assert np.allclose(final_surf.darrays[1].data[:5, :], target)

        target = np.array([[0.14771064, 0.9553547, -0.25588843],
                           [0.22574118, 0.97246087, -0.05797191],
                           [0.28987604, 0.95356506, 0.08176449],
                           [0.4109391, 0.89564127, 0.17016393],
                           [-0.06476931, 0.996889, -0.04491389]])
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
    result = _mesh_adjacency(faces)
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
    smaller_gifti = _downsample_single_surface(large_gifti, 0.1)
    n_verts = smaller_gifti.darrays[0].data.shape[0]
    ds_data = np.linspace(0, 100, n_verts)
    interp_data = interpolate_data(large_gifti, smaller_gifti, ds_data)
    target = np.array([[15.25175214, 27.25540839, 49.9735954, 56.24878169, 49.07015577,
                        33.66244014, 31.40144587, 28.00277272, 20.85717175, 27.23136023]])
    assert np.allclose(interp_data[:10], target)


def test_split_connected_components():
    """
    Test the _split_connected_components function to ensure it correctly splits a set of faces and
    vertices into distinct patches based on connectivity.

    This unit test performs the following:
    - Initializes two numpy arrays, `faces` and `vertices`
    - Invokes the `_split_connected_components` function to partition the input mesh into separate
      patches.
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
    split_patches = _split_connected_components(faces, vertices)

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
    split_patches = _split_connected_components(faces, vertices)

    assert len(split_patches) == 1

    patch_faces = split_patches[0]['faces']
    target = np.array([[0, 1, 2],
                       [0, 2, 3],
                       [4, 5, 0],
                       [6, 7, 8],
                       [10, 9, 3],
                       [0, 5, 6]])
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


# pylint: disable=W0621
@pytest.mark.dependency(depends=["test_create_layer_surface_set"])
def test_compute_dipole_orientations():
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
    with patch('os.getenv', return_value='./test_data/fs'):
        surf_set = LayerSurfaceSet('sub-104', 2)
        normals = surf_set.compute_dipole_orientations('link_vector', fixed=True)
        target = np.array([[0.14771064, 0.9553547, -0.25588843],
                           [0.22574118, 0.97246087, -0.05797153],
                           [0.28987604, 0.95356506, 0.08176449],
                           [0.4109391, 0.89564127, 0.17016393],
                           [-0.06476931, 0.99688905, -0.04491353]])
        assert np.allclose(normals[0, :5, :], target)
        assert np.allclose(normals[1, :5, :], target)

        normals = surf_set.compute_dipole_orientations('ds_surf_norm', fixed=True)
        target = np.array([[0.11921045, 0.9877371, -0.10081787],
                           [-0.20836607, 0.97758627, 0.03014262],
                           [-0.00309545, 0.98333, 0.18180308],
                           [0.4079666, 0.9053587, 0.11785118],
                           [-0.05669601, 0.9797419, 0.1920715]])
        assert np.allclose(normals[0, :5, :], target)
        assert np.allclose(normals[1, :5, :], target)

        normals = surf_set.compute_dipole_orientations('orig_surf_norm', fixed=True)
        target = np.array([[0.10781617, 0.9787314, -0.17452957],
                           [0.23795962, 0.9670099, -0.09092305],
                           [0.22581099, 0.9701262, 0.08868194],
                           [0.405921, 0.87620986, 0.25977778],
                           [-0.0196353, 0.994927, 0.09866542]])
        assert np.allclose(normals[0, :5, :], target)
        assert np.allclose(normals[1, :5, :], target)

        normals = surf_set.compute_dipole_orientations('cps', fixed=True)
        target = np.array([[0.05076333, 0.9803699, -0.19052005],
                           [0.17128408, 0.98397416, -0.04956321],
                           [0.07817034, 0.9902022, 0.11571109],
                           [0.44305757, 0.8352342, 0.325705],
                           [-0.02427451, 0.99666697, 0.0778831]])
        assert np.allclose(normals[0, :5, :], target)
        assert np.allclose(normals[1, :5, :], target)


@pytest.mark.dependency(depends=["test_create_layer_surface_set"])
def test_convert_fsaverage_to_native():
    """
    Tests the `convert_fsaverage_to_native` function to ensure it accurately converts
    FreeSurfer average coordinates to native brain space coordinates and handles errors
    appropriately when provided with invalid inputs.

    This test performs the following checks:
    1. Verify that the function returns the correct native vertex index for given valid inputs.
    2. Test the function's error handling by passing an out-of-range vertex index to ensure
       an IndexError is raised.
    3. Test the function's error handling by passing a non-existent subject ID to ensure
       a FileNotFoundError is raised.

    Procedures:
    - Calls `convert_fsaverage_to_native` with a valid subject ID, hemisphere, and vertex index.
      Checks if the returned value matches the expected native vertex index.
    - Attempts to call the function with an invalid vertex index to verify that an IndexError
      is properly triggered.
    - Attempts to call the function with a non-existent subject ID to verify that a
      FileNotFoundError is properly triggered.

    Assertions:
    - Asserts that the native vertex index returned matches the expected value of 166759 for
      known valid inputs.
    - Asserts that an IndexError is raised when a vertex index is used that exceeds the valid range.
    - Asserts that a FileNotFoundError is raised when an invalid subject ID is used.

    Raises:
        AssertionError: If any of the conditions checked by the assertions are not met, indicating
        an issue with the function's accuracy or error handling capabilities.
    """
    surf_set = LayerSurfaceSet('sub-104', 2)
    native_vtx = convert_fsaverage_to_native(surf_set, 'pial', 'lh', vert_idx=1000)
    target = 11341
    assert native_vtx == target

    native_vtx = convert_fsaverage_to_native(surf_set, 'pial', 'rh', vert_idx=1000)
    target = 33360
    assert native_vtx == target

    error_raise = False
    try:
        convert_fsaverage_to_native(surf_set, 'pial', 'lh', vert_idx=100000000)
    except IndexError:
        error_raise = True
    assert error_raise

    native_vtx = convert_fsaverage_to_native(surf_set, 'pial', 'lh')
    target = 163842
    assert native_vtx.shape[0] == target
    target = np.array([9343, 2602, 10319, 17133, 12151, 5355, 39024, 39271, 45703, 15326])
    assert np.allclose(native_vtx[:10], target)


@pytest.mark.dependency(depends=["test_create_layer_surface_set"])
def test_convert_native_to_fsaverage():
    """
    Tests the `convert_native_to_fsaverage` function to ensure it accurately maps native brain
    space coordinates to FreeSurfer average brain coordinates and handles errors appropriately.

    This function performs multiple checks:
    1. Validate that the function correctly returns the hemisphere and fsaverage vertex index
       for a given set of native space coordinates.
    2. Test the function's error handling by passing an invalid subject ID, expecting a
       FileNotFoundError to be raised.
    3. Test the function's error handling by passing an invalid surface path, also expecting a
       FileNotFoundError to be raised.

    Procedures:
    - Calls `convert_native_to_fsaverage` with a valid subject ID and surface path, checking
      if the hemisphere and vertex index returned match expected values.
    - Attempts to invoke the function with a non-existent subject ID to check for proper
      exception handling.
    - Attempts to invoke the function with a non-existent surface path to check for proper
      exception handling.

    Assertions:
    - Asserts that the hemisphere and fsaverage vertex index are as expected for valid inputs,
      confirming the function's accuracy in coordinate transformation.
    - Asserts that a FileNotFoundError is raised for an invalid subject ID and surface path,
      indicating robust error handling.

    Raises:
        AssertionError: If any of the conditions checked by the assertions are not met, indicating
        an issue with the function's accuracy or error handling capabilities.
    """
    with patch('os.getenv', return_value='./test_data/fs'):
        surf_set = LayerSurfaceSet('sub-104', 2)

        hemi, fs_vert = convert_native_to_fsaverage(
            surf_set,
            'pial',
            subj_coord=[-5.045391, -58.015587, 28.667336]
        )
        assert hemi == 'lh'
        assert fs_vert == 87729

        hemi, fs_vert = convert_native_to_fsaverage(
            surf_set,
            'pial',
            subj_coord=[25.045391, -58.015587, 28.667336]
        )
        assert hemi == 'rh'
        assert fs_vert == 86092

        hemis, verts = convert_native_to_fsaverage(
            surf_set,
            'pial'
        )
        assert len(hemis) == len(verts) == 49709
        target = ['lh', 'lh', 'lh', 'lh', 'lh', 'lh', 'lh', 'lh', 'lh', 'lh']
        assert np.all(hemis[:10] == target)
        target = [87729, 112541, 112542, 52282, 87824, 158647, 39230, 135879, 6903, 126602]
        assert np.all(verts[:10] == target)


# pylint: disable=R0915
@pytest.mark.dependency(depends=["test_create_layer_surface_set"])
def test_layersurfaceset():
    """
    Test the LayerSurfaceSet class and its core functionality.

    This unit test verifies initialization, metadata handling, mesh path generation,
    surface saving/loading, and interpolation functions of the LayerSurfaceSet class.
    It also ensures that appropriate exceptions are raised under error conditions
    (e.g., missing SUBJECTS_DIR environment variable, nonexistent subject directory,
    invalid hemisphere or surface name).

    Specifically, the test performs the following checks:
    - Raises EnvironmentError when SUBJECTS_DIR is unset.
    - Raises FileNotFoundError when the subject directory is missing.
    - Confirms correct string representation (__repr__) and mesh path formatting.
    - Verifies metadata creation, update, and retrieval via JSON sidecars.
    - Ensures correct layer naming for multi-layer configurations.
    - Tests validation error handling for missing hemispheres or surfaces.
    - Checks surface saving and metadata sidecar generation.
    - Validates interpolation consistency between downsampled and original meshes.

    Dependencies
    ------------
    - Depends on successful execution of `test_postprocess_freesurfer_surfaces`.

    Notes
    -----
    Temporary files (e.g., JSON metadata sidecars) are created under `./test_data/fs/sub-104`
    and removed after testing.
    """
    with patch('os.getenv', return_value=None):
        error_raise = False
        try:
            _ = LayerSurfaceSet('sub-104', 2)
        except EnvironmentError:
            error_raise = True
        assert error_raise

    error_raise = False
    try:
        _ = LayerSurfaceSet('sub-100', 2)
    except FileNotFoundError:
        error_raise = True
    assert error_raise

    with patch('os.getenv', return_value='./test_data/fs'):
        surf_set = LayerSurfaceSet('sub-104', 2)
        surf_str = f'{surf_set}'
        target = '<LayerSurfaceSet subj_id=\'sub-104\', n_layers=2, ' \
                 'dir=\'./test_data/fs/sub-104/surf/laminar\'>'
        assert surf_str == target

        mesh_path = surf_set.get_mesh_path('pial', stage='ds', orientation='link_vector',
                                           fixed=True)
        target = './test_data/fs/sub-104/surf/laminar/pial.ds.link_vector.fixed.gii'
        assert mesh_path == target

        meta = surf_set.load_meta('pial', stage='raw', hemi='lh')
        assert len(list(meta.keys())) == 0

        surf_set.update_meta('pial', stage='raw', hemi='lh')
        meta = surf_set.load_meta('pial', stage='raw', hemi='lh')
        assert len(list(meta.keys())) == 0

        surf_set.update_meta('pial', stage='raw', hemi='lh', updates={'test': 'abc'})
        meta = surf_set.load_meta('pial', stage='raw', hemi='lh')
        assert meta['test'] == 'abc'
        os.remove('./test_data/fs/sub-104/surf/laminar/lh.pial.raw.json')

        surf_set_3 = LayerSurfaceSet('sub-104', 3)
        layer_names = surf_set_3.get_layer_names()
        assert len(layer_names) == 3
        assert layer_names[0] == 'pial'
        assert layer_names[1] == '0.500'
        assert layer_names[2] == 'white'

        error_raise = False
        try:
            _ = surf_set.load('test')
        except FileNotFoundError:
            error_raise = True
        assert error_raise

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        test_surf = _create_surf_gifti(vertices, faces)
        surf_set.save(test_surf, 'test_layer')
        meta = surf_set.load_meta('test_layer')
        assert len(list(meta.keys())) == 1

        error_raise = False
        try:
            surf_set.validate(['raw'], hemis=['test'])
        except FileNotFoundError:
            error_raise = True
        assert error_raise

        pial_ds = surf_set.load('pial', stage='ds')
        np.random.seed(42)
        ds_data = np.random.random(pial_ds.darrays[0].data.shape[0])
        interp_data = surf_set.interpolate_layer_data('pial', ds_data)
        target = np.array([0.42281239, 0.46960404, 0.50846158, 0.57337757, 0.38641278])
        assert np.allclose(interp_data[:5], target)


@pytest.mark.dependency(depends=["test_create_layer_surface_set"])
def test_map_between_stages():
    """
    Tests LayerSurfaceSet.map_between_stages without modifying any files on disk.

    This test verifies:
    - Geometric KD-tree mappings ('combined' -> 'ds')
    - Metadata-based vertex removal mapping ('converted' -> 'nodeep')
    - Hemisphere concatenation/splitting logic ('nodeep' -> 'combined')
    - Proper ValueError handling for invalid hemisphere/stage combinations
    """
    with patch('os.getenv', return_value='./test_data/fs'):
        surf_set = LayerSurfaceSet('sub-104', 2)

        # --- geometric mapping between combined -> ds ---
        idx_comb_to_ds = surf_set.map_between_stages('pial', from_stage='combined', to_stage='ds')
        idx_ds_to_comb = surf_set.map_between_stages('pial', from_stage='ds', to_stage='combined')

        assert isinstance(idx_comb_to_ds, np.ndarray)
        assert isinstance(idx_ds_to_comb, np.ndarray)
        assert idx_comb_to_ds.ndim == 1
        assert idx_ds_to_comb.ndim == 1
        assert np.all(idx_comb_to_ds >= 0)
        assert np.all(idx_ds_to_comb >= 0)

        # --- converted -> nodeep mapping (mocking load_meta) ---
        with patch.object(LayerSurfaceSet, 'load_meta',
                          return_value={'deep_vertices_removed': [0, 2, 4]}):
            idx_conv_to_nodeep = surf_set.map_between_stages(
                'pial', from_stage='converted', to_stage='nodeep', from_hemi='lh',
                to_hemi='lh')
        assert isinstance(idx_conv_to_nodeep, np.ndarray)
        assert np.all(np.diff(idx_conv_to_nodeep) > 0)
        assert not np.any(np.isin([0, 2, 4], idx_conv_to_nodeep))

        # --- nodeep -> combined mapping (hemisphere offset) ---
        idx_lh = surf_set.map_between_stages('pial', from_stage='nodeep', from_hemi='lh',
                                             to_stage='combined')
        idx_rh = surf_set.map_between_stages('pial', from_stage='nodeep', from_hemi='rh',
                                             to_stage='combined')
        assert np.all(idx_lh == np.arange(len(idx_lh)))
        assert np.all(idx_rh >= len(idx_lh))

        # --- combined -> nodeep mapping (splitting) ---
        idx_lh_back = surf_set.map_between_stages('pial', from_stage='combined',
                                                  to_stage='nodeep', to_hemi='lh')
        idx_rh_back = surf_set.map_between_stages('pial', from_stage='combined',
                                                  to_stage='nodeep', to_hemi='rh')
        assert np.all(idx_lh_back == np.arange(len(idx_lh_back)))
        assert np.all(idx_rh_back == np.arange(idx_rh_back[0], idx_rh_back[-1] + 1))

        # --- error handling ---
        with pytest.raises(ValueError):
            surf_set.map_between_stages('pial', from_stage='converted', from_hemi='lh',
                                        to_stage='nodeep', to_hemi="rh")
        with pytest.raises(ValueError):
            surf_set.map_between_stages('pial', from_stage='combined', from_hemi='lh',
                                        to_stage='ds')
        with pytest.raises(ValueError):
            surf_set.map_between_stages('pial', from_stage='ds', to_stage='combined', to_hemi='lh')
        with pytest.raises(ValueError):
            surf_set.map_between_stages('pial', from_stage='converted', to_stage='combined')
        with pytest.raises(ValueError):
            surf_set.map_between_stages('pial', from_stage='nodeep', to_stage='combined')
        with pytest.raises(ValueError):
            surf_set.map_between_stages('pial', from_stage='combined', to_stage='nodeep')


def test_get_bigbrain_layer_boundaries():
    """
    Tests the `get_bigbrain_layer_boundaries` function to ensure it accurately retrieves
    the proportional boundaries of brain layers from the BigBrain model.

    This function performs several checks:
    1. Verifies that the function correctly returns the expected array of proportional
       boundaries for a given subject and coordinates using known correct inputs.
    2. Confirms that the function raises a FileNotFoundError when provided with an
       invalid subject identifier, ensuring robust error handling.
    3. Ensures that the function also raises a FileNotFoundError when given an incorrect
       path to the surface files, which tests the function's dependency on file paths.

    The tests use:
    - A path to test data structured in a typical project directory format.
    - Hardcoded coordinates which are representative of typical inputs.
    - Assertions to verify both the data accuracy and the error handling mechanisms.
    """

    subj_id = 'sub-104'
    surf_set = LayerSurfaceSet(subj_id, 2)
    vert_bb_prop = surf_set.get_bigbrain_layer_boundaries([-5.045391, -58.015587, 28.667336])
    expected = np.array([0.18065107, 0.2555629 , 0.4672846 , 0.7949229 , 0.90064305, 1.])
    assert np.allclose(vert_bb_prop, expected)

    vert_bb_prop = surf_set.get_bigbrain_layer_boundaries()
    assert vert_bb_prop.shape[0] == 6
    assert vert_bb_prop.shape[1] == 49709
    expected = np.array([0.18065107, 0.17599325, 0.11645006, 0.16804233, 0.12389062, 0.15487793,
                         0.15541628, 0.11214042, 0.18332304, 0.16658252])
    assert np.allclose(vert_bb_prop[0,:10], expected)


def test_get_cortical_thickness():
    """
    Test the get_cortical_thickness function of LayerSurfaceSet
    """

    subj_id = 'sub-104'
    surf_set = LayerSurfaceSet(subj_id, 2)
    thickness = surf_set.get_cortical_thickness(stage='ds')
    expected = np.array([1.9739379, 2.545998,  2.6078107, 2.5250378, 1.9865115])
    assert np.allclose(thickness[:5], expected)

    thickness = surf_set.get_cortical_thickness(stage='combined')
    expected = np.array([2.2318888, 2.1544316, 2.0110805, 1.9122171, 2.3491132])
    assert np.allclose(thickness[:5], expected)

    thickness = surf_set.get_cortical_thickness(stage='converted', hemi='rh')
    expected = np.array([2.5199146, 2.3940508, 2.3622077, 2.4827237, 2.5154493])
    assert np.allclose(thickness[:5], expected)

def test_get_distance_to_scalp():
    """
    Test the get_cortical_thickness function of LayerSurfaceSet
    """

    subj_id = 'sub-104'
    surf_set = LayerSurfaceSet(subj_id, 2)
    distance = surf_set.get_distance_to_scalp(stage='ds')
    expected = np.array([11.655455,   11.32269334, 11.4108027,  11.85575421, 11.56102818])
    assert np.allclose(distance[:5], expected)

    distance = surf_set.get_distance_to_scalp(layer_name='white', stage='combined')
    expected = np.array([13.5630841,  13.47294449, 13.47622333, 13.47952796, 13.7609574 ])
    assert np.allclose(distance[:5], expected)

    distance = surf_set.get_distance_to_scalp(stage='converted', hemi='rh')
    expected = np.array([12.27074489, 12.23813509, 12.3212802,  12.62445071, 12.87414463])
    assert np.allclose(distance[:5], expected)

def test_get_radiality_to_scalp():
    """
    Test the get_radiality_to_scalp function of LayerSurfaceSet
    """
    subj_id = 'sub-104'
    surf_set = LayerSurfaceSet(subj_id, 2)
    radiality = surf_set.get_radiality_to_scalp(layer_name='white')
    expected = np.array([0.98478,    0.98240334, 0.9504422,  0.8885453,  0.9843805 ])
    assert np.allclose(radiality[:5], expected)

def test_get_vertices_per_layer():
    """
    Test the get_vertices_per_layer function of LayerSurfaceSet
    """
    subj_id = 'sub-104'
    surf_set = LayerSurfaceSet(subj_id, 2)
    pial_mesh = surf_set.load(layer_name='pial', stage='ds')
    assert surf_set.get_vertices_per_layer() == pial_mesh.darrays[0].data.shape[0]
