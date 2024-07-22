"""
This module contains the unit tests for the `surf` module from the `lameg` package.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from lameg.surf import split_fv, mesh_adjacency, fix_non_manifold_edges, \
    find_non_manifold_edges, create_surf_gifti, _normit, mesh_normals, \
    remove_vertices, remove_unconnected_vertices  # pylint: disable=no-name-in-module


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
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return create_surf_gifti(vertices, faces)


@pytest.fixture
def unconnected_gifti():
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
    return create_surf_gifti(vertices, faces)


# pylint: disable=W0621
def test_remove_no_vertices(sample_gifti):
    """
    Test `remove_vertices` with no vertices specified for removal.
    Verifies that the Gifti surface remains unchanged when the vertices_to_remove array is empty.
    """
    new_gifti = remove_vertices(sample_gifti, np.array([]))
    assert np.array_equal(new_gifti.darrays[0].data, sample_gifti.darrays[0].data)
    assert np.array_equal(new_gifti.darrays[1].data, sample_gifti.darrays[1].data)


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
