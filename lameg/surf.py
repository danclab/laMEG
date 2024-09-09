"""
This module provides a set of tools for handling and manipulating surface mesh data. The
functionalities include computing mesh normals, interpolating mesh data, handling non-manifold
edges, creating and manipulating GIFTI surface files, and downsampling meshes using the VTK
library.

Key functionalities include:
- Normalization of vectors to unit length.
- Calculation of normals for mesh surfaces using both Delaunay triangulation and custom methods.
- Creation of GIFTI images from mesh data.
- Removal of specified vertices from a mesh and updating the mesh topology accordingly.
- Identification and handling of non-manifold edges to ensure mesh manifoldness.
- Interpolation of data from a downsampled mesh back to its original high-resolution mesh.
- Downsampling of meshes using VTK's decimation algorithms.
- Combination and adjustment of multiple surface meshes into a single mesh.
"""

# pylint: disable=C0302
import os
import copy
import subprocess
from collections import defaultdict

from joblib import Parallel, delayed

import nibabel as nib
import numpy as np
from scipy.spatial import KDTree, cKDTree # pylint: disable=E0611
from scipy.spatial.distance import cdist
from scipy.sparse import find, csr_matrix

# pylint: disable=E0611
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkFiltersCore import vtkDecimatePro
from vtkmodules.util.numpy_support import vtk_to_numpy

def _normit(vectors):
    """
    Normalize a numpy array of vectors.

    This function normalizes each row in the array of vectors to have unit length. If the length of
    a vector is below a certain threshold (machine epsilon), it is set to 1 to avoid division by
    zero.

    Parameters
    ----------
    vectors : ndarray
        Array of vectors to be normalized. Each row represents a vector.

    Returns
    -------
    ndarray
        Normalized array of vectors where each row has unit length.
    """

    norm_n = np.sqrt(np.sum(vectors ** 2, axis=1))
    norm_n[norm_n < np.finfo(float).eps] = 1
    return vectors / norm_n[:, np.newaxis]


def mesh_normals(vertices, faces, unit=False):
    """
    Normalize a numpy array of vectors.

    This function normalizes each row in the array of vectors to have unit length. If the length of
    a vector is below a certain threshold (machine epsilon), it is set to 1 to avoid division by
    zero.

    Parameters
    ----------
    vectors : ndarray
        Array of vectors to be normalized. Each row represents a vector.

    Returns
    -------
    ndarray
        Normalized array of vectors where each row has unit length.
    """

    face_normal = np.cross(
        vertices[faces[:, 1], :] - vertices[faces[:, 0], :],
        vertices[faces[:, 2], :] - vertices[faces[:, 0], :]
    )
    face_normal = _normit(face_normal)

    vertex_normal = np.zeros_like(vertices)
    for i in range(len(faces)):
        for j in range(3):
            vertex_normal[faces[i, j], :] += face_normal[i, :]

    centered_vertices = vertices - np.mean(vertices, axis=0)
    if np.count_nonzero(np.sign(np.sum(centered_vertices * vertex_normal, axis=1))) > \
            len(centered_vertices) / 2:
        vertex_normal = -vertex_normal
        face_normal = -face_normal

    if unit:
        vertex_normal = _normit(vertex_normal)
        face_normal = _normit(face_normal)

    return vertex_normal, face_normal


def create_surf_gifti(vertices, faces, normals=None):
    """
    Create a Gifti image object from surface mesh data.

    This function creates a GiftiImage object from the provided vertices, faces, and optional
    normals. The vertices and faces are required, while normals are optional. If normals are
    provided, they are added to the Gifti image. The function returns the GiftiImage object.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of vertices. Each row represents a vertex with its x, y, z coordinates.
    faces : numpy.ndarray
        Array of faces. Each row represents a face with three integers corresponding to vertex
        indices.
    normals : numpy.ndarray, optional
        Array of vertex normals. Each row represents a normal vector corresponding to a vertex.

    Returns
    -------
    new_gifti : nibabel.gifti.GiftiImage
        The GiftiImage object created from the provided mesh data.

    Notes
    -----
    - Vertex, face, and normal arrays should be NumPy arrays.
    - Vertices and normals should be in float32 format, and faces should be in int32 format.

    Example
    -------
    >>> import numpy as np
    >>> import nibabel as nib
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    >>> faces = np.array([[0, 1, 2], [0, 2, 3]])
    >>> normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    >>> gifti_img = create_surf_gifti(vertices, faces, normals)
    """

    # Create new gifti object
    new_gifti = nib.gifti.GiftiImage()

    # Cast vertices and faces to the appropriate data types
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)

    # Add the vertices and faces to the gifti object
    new_gifti.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=vertices,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
        )
    )
    new_gifti.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=faces,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
        )
    )

    # If normals are provided and not empty, cast them to float32 and add them to the Gifti image
    if normals is not None:
        normals = np.array(normals).astype(np.float32)
        new_gifti.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data=normals,
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']
            ))

    return new_gifti


def remove_unconnected_vertices(gifti_surf):
    """
    Remove vertices that are not connected to any faces from a Gifti surface object.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        The Gifti surface object to be processed.

    Returns
    -------
    cleaned_gifti_surf : nibabel.gifti.GiftiImage
        A new GiftiImage object with unconnected vertices removed.
    """

    # Get the pointset (vertices) and triangle array (faces) from the Gifti surface
    vertices = gifti_surf.darrays[0].data
    faces = gifti_surf.darrays[1].data

    # Find all unique vertex indices that are used in faces
    connected_vertices = np.unique(faces.flatten())

    # Determine which vertices are not connected to any faces
    all_vertices = np.arange(vertices.shape[0])
    unconnected_vertices = np.setdiff1d(all_vertices, connected_vertices)

    # Remove unconnected vertices using the provided remove_vertices function
    cleaned_gifti_surf = remove_vertices(gifti_surf, unconnected_vertices)

    return cleaned_gifti_surf


def remove_vertices(gifti_surf, vertices_to_remove):
    """
    Remove specified vertices from a Gifti surface and update the faces accordingly.

    This function modifies a Gifti surface by removing the specified vertices. It also updates
    the faces of the surface so that they only reference the remaining vertices. If normals
    are present in the surface, they are also updated to correspond to the new set of vertices.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        The Gifti surface object from which vertices will be removed.
    vertices_to_remove : array_like
        An array of vertex indices to be removed from the surface.

    Returns
    -------
    new_gifti : nibabel.gifti.GiftiImage
        A new GiftiImage object with the specified vertices removed and faces updated.

    Notes
    -----
    - The function assumes that the GiftiImage object contains at least two data arrays: one for
      vertices and one for faces. If normals are present, they are also updated.
    - Vertex indices in `vertices_to_remove` should be zero-based (following Python's indexing
      convention).
    - The returned GiftiImage object is a new object; the original `gifti_surf` object is not
      modified in place.

    Example
    -------
    >>> import nibabel as nib
    >>> gifti_surf = nib.load('path_to_gifti_file.gii')
    >>> vertices_to_remove = np.array([0, 2, 5])  # Indices of vertices to remove
    >>> new_gifti_surf = remove_vertices(gifti_surf, vertices_to_remove)
    """

    # Extract vertices and faces from the gifti object
    vertices_data = [da for da in gifti_surf.darrays
                     if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']][0]
    faces_data = [da for da in gifti_surf.darrays
                  if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']][0]

    vertices = vertices_data.data
    faces = faces_data.data

    # Determine vertices to keep
    vertices_to_keep = np.setdiff1d(np.arange(vertices.shape[0]), vertices_to_remove)

    # Create new array of vertices
    new_vertices = vertices[vertices_to_keep, :]

    # Find which faces to keep - ones that point to kept vertices
    if faces.shape[0]>0:
        face_x = np.isin(faces[:, 0], vertices_to_keep)
        face_y = np.isin(faces[:, 1], vertices_to_keep)
        face_z = np.isin(faces[:, 2], vertices_to_keep)
        faces_to_keep = np.where(face_x & face_y & face_z)[0]

        # Re-index faces
        x_faces = faces[faces_to_keep, :].reshape(-1)
        idxs = np.searchsorted(vertices_to_keep, x_faces)
        new_faces = idxs.reshape(-1, 3)
    else:
        new_faces = faces

    # Create new gifti object
    normals = None
    normals_data = [da for da in gifti_surf.darrays
                    if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']]
    if normals_data:
        normals = normals_data[0].data[vertices_to_keep, :]
    new_gifti = create_surf_gifti(new_vertices, new_faces, normals=normals)

    return new_gifti


def find_non_manifold_edges(faces):
    """
    Identify non-manifold edges in a given mesh represented by its faces.

    A non-manifold edge is defined as an edge that is shared by more than two faces. This function
    processes an array of faces, each face represented by a tuple of vertex indices, and identifies
    edges that meet the non-manifold criteria.

    Parameters
    ----------
    faces : np.ndarray
        An array where each row represents a face as a tuple of three vertex indices.

    Returns
    -------
    non_manifold_edges : dict
        A dictionary where keys are tuples representing non-manifold edges (vertex indices are
        sorted), and values are lists of face indices that share the edge.

    Notes
    -----
    The function uses a `defaultdict` to collect face indices for each edge encountered in the mesh.
    It then filters out edges that are associated with more than two faces, identifying them as
    non-manifold.
    """

    edge_faces = defaultdict(list)

    for i, (vertex_1, vertex_2, vertex_3) in enumerate(faces):
        for edge in [(vertex_1, vertex_2), (vertex_2, vertex_3), (vertex_3, vertex_1)]:
            edge_faces[tuple(sorted(edge))].append(i)

    non_manifold_edges = {edge: fcs for edge, fcs in edge_faces.items() if len(fcs) > 2}
    return non_manifold_edges


def fix_non_manifold_edges(vertices, faces):
    """
    Remove faces associated with non-manifold edges from a mesh defined by vertices and faces.

    Non-manifold edges are edges that are shared by more than two faces, which can cause issues
    in various mesh processing tasks such as mesh simplification, smoothing, or 3D printing. This
    function identifies such edges and removes all faces associated with them to ensure
    manifoldness of the mesh.

    Parameters
    ----------
    vertices : np.ndarray
        An array of vertices, where each row represents a vertex as [x, y, z] coordinates.
    faces : np.ndarray
        An array of faces, where each row represents a face as indices into the vertices array.

    Returns
    -------
    vertices : np.ndarray
        The unchanged array of vertices.
    new_faces : np.ndarray
        The modified array of faces, with faces associated with non-manifold edges removed.

    Examples
    --------
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    >>> faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]])
    >>> new_vertices, new_faces = fix_non_manifold_edges(vertices, faces)
    >>> new_faces
    array([[0, 1, 2], [0, 2, 3]])  # Assuming face [1, 2, 3] was associated with a non-manifold
    edge.
    """

    non_manifold_edges = find_non_manifold_edges(faces)
    conflicting_faces = set()
    for faces_list in non_manifold_edges.values():
        conflicting_faces.update(faces_list)

    # Create a new face list excluding the conflicting faces
    new_faces = np.array(
        [face for i, face in enumerate(faces) if i not in conflicting_faces],
        dtype=np.int32
    )
    return vertices, new_faces


def downsample_single_surface(gifti_surf, ds_factor=0.1):
    """
    Downsample a Gifti surface using the VTK library.

    This function takes a Gifti surface defined by its vertices and faces, and downsamples it using
    VTK's `vtkDecimatePro` algorithm. The reduction ratio determines the degree of downsampling.
    The function returns the downsampled Gifti surface.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        The Gifti surface object to be downsampled.
    reduction_ratio : float
        The proportion of the mesh to remove. For example, a reduction ratio of 0.1 retains 90% of
        the original mesh.

    Returns
    -------
    new_gifti_surf : nibabel.gifti.GiftiImage
        A new GiftiImage object with the downsampled surface.

    Notes
    -----
    - The input faces array should be triangulated, i.e., each face should consist of exactly three
      vertex indices.
    - The VTK library is used for mesh decimation, which must be installed and properly configured.
    - The returned GiftiImage object is a new object; the original `gifti_surf` object is not
      modified in place.

    Example
    -------
    >>> import numpy as np
    >>> gifti_surf = nib.load('path_to_gifti_file.gii')
    >>> new_gifti_surf = downsample_single_surface(gifti_surf, 0.1)
    """

    vertices = gifti_surf.darrays[0].data
    faces = gifti_surf.darrays[1].data

    # Convert vertices and faces to a VTK PolyData object
    points = vtkPoints()
    for point in vertices:
        points.InsertNextPoint(point)

    cells = vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for vertex in face:
            cells.InsertCellPoint(vertex)

    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    # Apply vtkDecimatePro for decimation
    decimate = vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(1 - ds_factor)
    decimate.Update()

    # Extract the decimated mesh
    decimated_polydata = decimate.GetOutput()

    # Convert back to numpy arrays
    reduced_vertices = vtk_to_numpy(decimated_polydata.GetPoints().GetData())

    # Extract and reshape the face data
    face_data = vtk_to_numpy(decimated_polydata.GetPolys().GetData())
    # Assuming the mesh is triangulated, every fourth item is the size (3), followed by three
    # vertex indices
    reduced_faces = face_data.reshape(-1, 4)[:, 1:4]

    # Find the original vertices closest to the downsampled vertices
    kdtree = KDTree(gifti_surf.darrays[0].data)
    _, orig_vert_idx = kdtree.query(reduced_vertices, k=1)

    reduced_normals = None
    if len(gifti_surf.darrays) > 2 and \
            gifti_surf.darrays[2].intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR'] and \
            gifti_surf.darrays[2].data.shape[0]==gifti_surf.darrays[0].data.shape[0]:
        reduced_normals = gifti_surf.darrays[2].data[orig_vert_idx]

    new_gifti_surf = create_surf_gifti(reduced_vertices, reduced_faces, normals=reduced_normals)

    return new_gifti_surf


def iterative_downsample_single_surface(gifti_surf, ds_factor=0.1):
    """
    Iteratively downsample a single surface mesh to a target number of vertices.

    This function reduces the number of vertices in a surface mesh (in GIFTI format) to a specified
    fraction of its original size. Downsampling is performed iteratively until the target number of
    vertices is reached or closely approximated.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        The surface mesh to be downsampled, provided as a GIFTI image object.
    ds_factor : float, optional
        The downsampling factor representing the target fraction of the original number of vertices.
        Default is 0.1.

    Returns
    -------
    current_surf : nibabel.gifti.GiftiImage
        The downsampled surface mesh as a GIFTI image object.

    Notes
    -----
    - The downsampling process is iterative. In each iteration, the mesh is downsampled by a factor
      calculated to approach the target number of vertices.
    - If the calculated downsampling factor in an iteration equals or exceeds 1, the process is
      terminated to prevent upsampling or infinite loops.
    """

    current_surf = gifti_surf
    current_vertices = gifti_surf.darrays[0].data.shape[0]
    target_vertices = int(current_vertices * ds_factor)
    current_ds_factor = target_vertices / current_vertices

    while current_vertices > target_vertices:
        # Downsample the mesh
        current_surf = downsample_single_surface(current_surf, ds_factor=current_ds_factor)

        # Update the current vertices
        current_vertices = current_surf.darrays[0].data.shape[0]

        current_ds_factor = (target_vertices / current_vertices) * 1.25
        if current_ds_factor >= 1:
            break

    # Remove non-manifold edges
    ds_vertices = current_surf.darrays[0].data
    ds_faces = current_surf.darrays[1].data
    nonmani_vertices, nonmani_faces = fix_non_manifold_edges(ds_vertices, ds_faces)

    normals = None
    if len(current_surf.darrays) > 2 and \
            current_surf.darrays[2].intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR'] and \
            current_surf.darrays[2].data.shape[0]==current_surf.darrays[0].data.shape[0]:
        normals = current_surf.darrays[2].data

    current_surf = create_surf_gifti(nonmani_vertices, nonmani_faces, normals=normals)

    # Remove unconnected vertices
    current_surf = remove_unconnected_vertices(current_surf)

    return current_surf


def downsample_multiple_surfaces(in_surfs, ds_factor):
    """
    Downsample multiple surface meshes using the VTK decimation algorithm.

    This function takes a list of input surface meshes (in Gifti format) and applies a downsampling
    process to each surface. The downsampling is performed using VTK's `vtkDecimatePro` algorithm.
    The first surface in the list is downsampled, and its vertex mapping is then applied to all
    other surfaces in the list. The function returns a list of downsampled surface meshes.

    Parameters
    ----------
    in_surfs : list of nibabel.gifti.GiftiImage
        Input Gifti surface meshes to be downsampled.
    ratio : float
        The reduction ratio for the downsampling process. For example, a ratio of 0.1
        implies that the mesh will be reduced to 90% of its original size.

    Returns
    -------
    out_surfs : list of nibabel.gifti.GiftiImage
        List of downsampled Gifti surface meshes.

    Notes
    -----
    - The function prints the percentage of vertices retained in the first surface after
      downsampling.
    - If normals are present in the input surfaces, they are also downsampled and mapped to the
      new surfaces.
    - The resulting surfaces maintain the original topology and are suitable for visualization and
      further processing.

    Example
    -------
    >>> import nibabel as nib
    >>> in_surfs = [nib.load('path/to/input_surf1.gii'), nib.load('path/to/input_surf2.gii')]
    >>> ratio = 0.1
    >>> out_surfs = downsample_multiple_surfaces(in_surfs, ratio)
    >>> for i, ds_surf in enumerate(out_surfs):
    ...     nib.save(ds_surf, f'path/to/output_surf{i+1}.gii')
    """

    out_surfs = []

    primary_surf = in_surfs[0]
    ds_primary_surf = iterative_downsample_single_surface(primary_surf, ds_factor=ds_factor)
    reduced_vertices = ds_primary_surf.darrays[0].data
    reduced_faces = ds_primary_surf.darrays[1].data

    # Find the original vertices closest to the downsampled vertices
    kdtree = KDTree(primary_surf.darrays[0].data)
    # Calculate the percentage of vertices retained
    decim_orig_dist, orig_vert_idx = kdtree.query(reduced_vertices, k=1)
    print(
        f"{(1 - np.mean(decim_orig_dist > 0)) * 100}% of the vertices in the decimated surface "
        f"belong to the original surface."
    )

    # Save the downsampled primary surface with normals
    out_surfs.append(ds_primary_surf)

    # Process other surfaces
    for i in range(1, len(in_surfs)):
        surf = in_surfs[i]

        reduced_normals = None
        if len(surf.darrays) > 2 and \
                surf.darrays[2].intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR'] and \
                surf.darrays[2].data.shape[0] == surf.darrays[0].data.shape[0]:
            reduced_normals = surf.darrays[2].data[orig_vert_idx]

        surf_verts=surf.darrays[0].data[orig_vert_idx, :]

        nonmani_vertices, nonmani_faces = fix_non_manifold_edges(surf_verts, reduced_faces)

        ds_surf = create_surf_gifti(nonmani_vertices, nonmani_faces, normals=reduced_normals)

        out_surfs.append(ds_surf)
    return out_surfs


def combine_surfaces(surfaces):
    """
    Combine multiple surface meshes into a single surface mesh.

    This function takes a list of Gifti surface meshes and combines them into a single surface
    mesh. It concatenates the vertices, faces, and normals (if present) from each surface. The
    faces are re-indexed appropriately to maintain the correct references to the combined vertex
    array.

    Parameters
    ----------
    surfaces : list of nibabel.gifti.GiftiImage
        List of Gifti surface meshes to be combined.

    Returns
    -------
    combined_surf : nibabel.gifti.GiftiImage
        A single combined Gifti surface mesh.

    Notes
    -----
    - The vertices, faces, and normals (if present) from each surface are concatenated.
    - The faces are re-indexed to reference the correct vertices in the combined vertex array.
    - If normals are present in any of the input surfaces, they are also combined.

    Raises
    ------
    ValueError
        If the vertex or face arrays do not have the expected dimensions.

    Example
    -------
    >>> import nibabel as nib
    >>> surfaces = [nib.load('path/to/surface1.gii'), nib.load('path/to/surface2.gii')]
    >>> combined_surf = combine_surfaces(surfaces)
    >>> nib.save(combined_surf, 'path/to/combined_surface.gii')
    """

    combined_vertices = []
    combined_faces = []
    combined_normals = []

    face_offset = 0

    for mesh in surfaces:

        # Extract vertices and faces
        vertices = np.concatenate(
            [da.data for da in mesh.darrays
             if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']])
        faces = np.concatenate(
            [da.data for da in mesh.darrays
             if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']])

        # Check for normals
        normal_arrays = [da.data for da in mesh.darrays
                         if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']]
        normals = np.concatenate(normal_arrays) if normal_arrays else np.array([])

        combined_vertices.append(vertices)
        combined_faces.append(faces + face_offset)
        if normals.size:
            combined_normals.append(normals)

        face_offset += vertices.shape[0]

    # Combine the arrays
    combined_vertices = np.vstack(combined_vertices).astype(np.float32)
    combined_faces = np.vstack(combined_faces).astype(np.int32)
    if combined_normals:
        combined_normals = np.vstack(combined_normals).astype(np.float32)

    combined_surf = create_surf_gifti(
        combined_vertices,
        combined_faces,
        normals=combined_normals
    )

    return combined_surf


# pylint: disable=R0912
def compute_dipole_orientations(method, layer_names, surf_dir, fixed=True):
    """
    Compute dipole orientations for cortical layers using different methods.

    Parameters
    ----------
    method : str
        Method for computing dipole orientations ('link_vector', 'ds_surf_norm', 'orig_surf_norm',
        or 'cps').
        link_vector: Vectors connecting pial vertices to corresponding white matter vertices.
        ds_surf_norm: Surface normal vectors computed from the downsampled surface.
        orig_surf_norm: Surface normal vectors computed from the original (non-downsampled) surface.
        cps: Cortical patch statistics - mean surface normal vectors from connected vertices in the
        original (non-downsampled) surface.
    layer_names : list
        Names of the cortical layers.
    surf_dir : str
        Directory where the surface files are stored.
    fixed : bool, optional
        Flag to ensure that orientation of corresponding vertices across layers is the same (True
        by default). If True, for 'ds_surf_norm', 'orig_surf_norm', and 'cps', orientations
        computed from the pial surface are used for all layers.

    Returns
    -------
    orientations : np.ndarray
        An array of dipole orientations for each vertex in each layer.

    Raises
    ------
    ValueError
        If the number of vertices in pial and white matter surfaces do not match.
    """

    orientations = None

    if method == 'link_vector':
        # Method: Use link vectors between pial and white surfaces as dipole orientations
        # Load downsampled pial and white surfaces
        pial_surf = nib.load(os.path.join(surf_dir, 'pial.ds.gii'))
        white_surf = nib.load(os.path.join(surf_dir, 'white.ds.gii'))

        # Extract vertices
        pial_vertices = pial_surf.darrays[0].data
        white_vertices = white_surf.darrays[0].data

        # Ensure same number of vertices in pial and white surfaces
        if pial_vertices.shape[0] != white_vertices.shape[0]:
            raise ValueError("Pial and white surfaces must have the same number of vertices")

        # Compute link vectors
        link_vectors = white_vertices - pial_vertices
        link_vectors = _normit(link_vectors)

        # Replicate link vectors for each layer
        orientations = np.tile(link_vectors, (len(layer_names), 1, 1))

    elif method == 'ds_surf_norm':
        # Method: Use normals of the downsampled surfaces
        orientations = []
        for l_idx, layer_name in enumerate(layer_names):
            if l_idx == 0 or not fixed:
                in_surf_path = os.path.join(surf_dir, f'{layer_name}.ds.gii')
                surf = nib.load(in_surf_path)
                vtx_norms, _ = mesh_normals(
                    surf.darrays[0].data,
                    surf.darrays[1].data,
                    unit=True
                )
            orientations.append(vtx_norms)
        orientations = np.array(orientations)

    elif method == 'orig_surf_norm':
        # Method: Use normals of the original surfaces, mapped to downsampled surfaces
        orientations = []
        for l_idx, layer_name in enumerate(layer_names):
            if l_idx == 0 or not fixed:
                in_surf_path = os.path.join(surf_dir, f'{layer_name}.gii')
                orig_surf = nib.load(in_surf_path)
                ds_surf_path = os.path.join(surf_dir, f'{layer_name}.ds.gii')
                ds_surf = nib.load(ds_surf_path)
                kdtree = KDTree(orig_surf.darrays[0].data)
                _, orig_vert_idx = kdtree.query(ds_surf.darrays[0].data, k=1)
                vtx_norms, _ = mesh_normals(
                    orig_surf.darrays[0].data,
                    orig_surf.darrays[1].data,
                    unit=True
                )
            orientations.append(vtx_norms[orig_vert_idx, :])
        orientations = np.array(orientations)

    elif method == 'cps':
        # Method: Use cortical patch statistics for normals
        orientations = []
        for l_idx, layer_name in enumerate(layer_names):
            if l_idx == 0 or not fixed:
                in_surf_path = os.path.join(surf_dir, f'{layer_name}.gii')
                orig_surf = nib.load(in_surf_path)
                ds_surf_path = os.path.join(surf_dir, f'{layer_name}.ds.gii')
                ds_surf = nib.load(ds_surf_path)
                kdtree = KDTree(ds_surf.darrays[0].data)
                _, ds_vert_idx = kdtree.query(orig_surf.darrays[0].data, k=1)
                orig_vtx_norms, _ = mesh_normals(
                    orig_surf.darrays[0].data,
                    orig_surf.darrays[1].data,
                    unit=True
                )
                vtx_norms, _ = mesh_normals(
                    ds_surf.darrays[0].data,
                    ds_surf.darrays[1].data,
                    unit=True
                )
                for v_idx in range(vtx_norms.shape[0]):
                    orig_idxs = np.where(ds_vert_idx == v_idx)[0]
                    if len(orig_idxs):
                        vtx_norms[v_idx, :] = np.mean(orig_vtx_norms[orig_idxs, :], axis=0)
                vtx_norms = _normit(vtx_norms)
            orientations.append(vtx_norms)
        orientations = np.array(orientations)

    return orientations


def create_layer_mesh(layer, hemispheres, fs_subject_dir) -> None:
    """
    Create or retrieve a specified cortical layer mesh file name or path based on the provided
    layer proportional thickness or identifier.

    Parameters
    ----------
    layer : float or int
        Specifies the cortical layer. The value 1 corresponds to the 'pial' surface, values between
        0 and 1 (exclusive) correspond to intermediate layers (specified as a decimal), and the
        value 0 corresponds to the 'white matter' surface.
    hemispheres : list of str
        A list of hemisphere identifiers (e.g., ['lh', 'rh']) for which meshes should be created or
        retrieved.
    fs_subject_dir : str
        Path to the subject directory within the FreeSurfer environment. This directory should
        include a 'surf' directory where mesh files are stored.

    Returns
    -------
    layer_name : str or None
        Returns a string representing the mesh layer ('pial', 'white', or a specific intermediate
        layer as a formatted string). Returns None if the input layer does not match any recognized
        pattern (e.g., a negative number or a number greater than 1).

    Notes
    -----
    For intermediate layers (0 < layer < 1), the function will check for the existence of the mesh
    file corresponding to each hemisphere. If it does not exist, it uses 'mris_expand' to generate
    it using the white matter surface file. If the layer exactly matches 0 or 1, it returns the
    corresponding standard FreeSurfer mesh identifier ('white' or 'pial').
    """

    if layer == 1:
        return 'pial'
    if 0 < layer < 1:
        layer_name = f'{layer:.3f}'
        for hemi in hemispheres:
            wm_file = os.path.join(fs_subject_dir, 'surf', f'{hemi}.white')
            out_file = os.path.join(fs_subject_dir, 'surf', f'{hemi}.{layer_name}')
            if not os.path.exists(out_file):
                cmd = ['mris_expand', '-thickness', wm_file, f'{layer}', out_file]
                subprocess.run(cmd, check=True)
        return layer_name
    if layer == 0:
        return 'white'
    return None


# pylint: disable=R0912,R0915
def postprocess_freesurfer_surfaces(subj_id,
                                    out_dir,
                                    out_fname,
                                    n_surfaces=11,
                                    ds_factor=0.1,
                                    orientation='link_vector',
                                    fix_orientation=True,
                                    remove_deep=True,
                                    n_jobs=-1):
    """
    Process and combine FreeSurfer surface meshes for a subject.

    This function processes FreeSurfer surface meshes for a given subject by creating intermediate
    surfaces, adjusting for RAS offset, removing deep vertices, combining hemispheres,
    downsampling, and computing link vectors. The resulting surfaces are combined and saved to a
    specified output file.

    Parameters
    ----------
    subj_id : str
        Subject ID corresponding to the FreeSurfer subject directory.
    out_dir : str
        Output directory where the processed files will be saved.
    out_fname : str
        Filename for the final combined surface mesh.
    n_surfaces : int, optional
        Number of intermediate surfaces to create between white and pial surfaces.
    ds_factor : float, optional
        Downsampling factor for surface decimation.
    orientation : str, optional
        Method to compute orientation vectors ('link_vector' for pial-white link, 'ds_surf_norm'
        for downsampled surface normals, 'orig_surf_norm' for original surface normals, and 'cps'
        for cortical patch statistics).
    fix_orientation : bool, optional
        Flag to ensure that orientation of corresponding vertices across layers is the same (True
        by default).
    remove_deep : bool, optional
        Flag to remove vertices located in deep regions (labeled as 'unknown').
    n_jobs : int, optional
        Number of parallel processes to run. -1 for all available cores (default is -1).

    Notes
    -----
    - This function assumes the FreeSurfer 'SUBJECTS_DIR' environment variable is set.
    - Surfaces are processed in Gifti format and combined into a single surface mesh.

    Example
    -------
    >>> postprocess_freesurfer_surfaces('subject1', '/path/to/output', 'combined_surface.gii')
    """

    hemispheres = ['lh', 'rh']
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')

    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    layers = np.linspace(1, 0, n_surfaces)

    ## Create intermediate surfaces if needed
    layer_names = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(create_layer_mesh)(layer, hemispheres, fs_subject_dir) for layer in layers
    )

    ## Compute RAS offset
    # Define the path to the MRI file
    ras_off_file = os.path.join(fs_subject_dir, 'mri', 'orig.mgz')

    # Execute the shell command to get RAS offset
    command = f"mri_info --cras {ras_off_file}"
    with subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
    ) as process:
        out, _ = process.communicate()

    # Parse the output
    cols = out.decode().split()
    ras_offset = np.array([float(cols[0]), float(cols[1]), float(cols[2])])

    # Print the result
    print(ras_offset)

    ## Convert to gifti, adjust for RAS offset, and remove deep vertices
    surfaces_to_process = copy.copy(layer_names)
    surfaces_to_process.append('inflated')

    # pylint: disable=R1702
    for surface_name in surfaces_to_process:
        for hemi in hemispheres:
            # Construct the original and new file names
            orig_name = os.path.join(fs_subject_dir, 'surf', f'{hemi}.{surface_name}')
            new_name = os.path.join(out_dir, f'{hemi}.{surface_name}.gii')
            rm_deep_name = os.path.join(out_dir, f'{hemi}.{surface_name}.nodeep.gii')

            # Convert the surface file to Gifti format
            subprocess.run(['mris_convert', orig_name, new_name], check=True)

            # Load the Gifti file
            surf_g = nib.load(new_name)

            # Set transformation matrix to identity
            surf_g.affine = np.eye(4)

            # Adjust for RAS offset
            n_vertices = 0
            for data_array in surf_g.darrays:
                if data_array.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
                    data_array.data += ras_offset
                    n_vertices = data_array.data.shape[0]
            nib.save(surf_g, new_name)

            annotation = os.path.join(fs_subject_dir, 'label', f'{hemi}.aparc.annot')
            label, _, names = nib.freesurfer.read_annot(annotation)

            # Remove vertices created by cutting the hemispheres
            if remove_deep:
                vertices_to_remove = []
                for vtx in range(n_vertices):
                    if label[vtx] > 0:
                        region = names[label[vtx]]
                        if region == 'unknown':
                            vertices_to_remove.append(vtx)
                    else:
                        vertices_to_remove.append(vtx)
                surf_g = remove_vertices(surf_g, np.array(vertices_to_remove))

            # Save the modified Gifti file
            nib.save(surf_g, rm_deep_name)

    ## Combine hemispheres
    for surface_name in surfaces_to_process:
        # Load left and right hemisphere surfaces
        lh_fname = os.path.join(out_dir, f'lh.{surface_name}.nodeep.gii')
        l_hemi = nib.load(lh_fname)
        rh_fname = os.path.join(out_dir, f'rh.{surface_name}.nodeep.gii')
        r_hemi = nib.load(rh_fname)
        if surface_name == 'inflated':
            lh_width = np.max(l_hemi.darrays[0].data[:, 0])-np.min(l_hemi.darrays[0].data[:, 0])
            r_hemi.darrays[0].data[:, 0] = r_hemi.darrays[0].data[:, 0] + \
                                           np.max(l_hemi.darrays[0].data[:, 0]) + \
                                           (.5*lh_width)

        # Combine the surfaces
        combined = combine_surfaces([l_hemi, r_hemi])
        combined_fname = os.path.join(out_dir, f'{surface_name}.gii')
        nib.save(combined, combined_fname)

    ## Downsample surfaces at the same time
    # Get list of surfaces
    in_surfs = []
    for surface_name in surfaces_to_process:
        in_surf_fname = os.path.join(out_dir, f'{surface_name}.gii')
        in_surf = nib.load(in_surf_fname)
        in_surfs.append(in_surf)

    # Downsample multiple surfaces
    out_surfs = downsample_multiple_surfaces(in_surfs, ds_factor)
    for surface_name, out_surf in zip(surfaces_to_process, out_surfs):
        out_surf_path = os.path.join(out_dir, f'{surface_name}.ds.gii')
        nib.save(out_surf, out_surf_path)

    ## Compute dipole orientations
    orientations = compute_dipole_orientations(
        orientation,
        layer_names,
        out_dir,
        fixed=fix_orientation
    )

    base_fname = f'ds.{orientation}'
    if fix_orientation:
        base_fname = f'{base_fname}.fixed'
    for l_idx, layer_name in enumerate(layer_names):
        in_surf_path = os.path.join(out_dir, f'{layer_name}.ds.gii')
        surf = nib.load(in_surf_path)

        # Set these link vectors as the normals for the downsampled surface
        ori_array=nib.gifti.GiftiDataArray(data=orientations[l_idx, :, :],
                                           intent=nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR'])
        surf.add_gifti_data_array(ori_array)

        # Save the modified downsampled surface with link vectors as normals
        out_surf_path = os.path.join(out_dir, f'{layer_name}.{base_fname}.gii')
        nib.save(surf, out_surf_path)

    ## Combine layers
    all_surfs = []
    for layer_name in layer_names:
        surf_path = os.path.join(out_dir, f'{layer_name}.{base_fname}.gii')
        surf = nib.load(surf_path)
        all_surfs.append(surf)

    combined = combine_surfaces(all_surfs)
    nib.save(combined, os.path.join(out_dir, out_fname))


def mesh_adjacency(faces):
    """
    Compute the adjacency matrix of a triangle mesh.

    Parameters
    ----------
    faces : np.array, shape (f, 3)
        The mesh faces, where `f` is the number of faces, and each face is represented by a tuple of
        three vertex indices.

    Returns
    -------
    adjacency : np.array, shape (v, v)
        The adjacency matrix of the mesh, where `v` is the number of vertices. Each entry (i, j)
        indicates whether vertices i and j are connected by an edge.
    """

    faces = np.asarray(faces, dtype=int)
    n_vertices = np.max(faces)+1  # Assuming max vertex index represents the number of vertices

    # Flatten the indices to create row and column indices for the adjacency matrix
    row_indices = np.hstack([faces[:, 0], faces[:, 0], faces[:, 1],
                             faces[:, 1], faces[:, 2], faces[:, 2]])
    col_indices = np.hstack([faces[:, 1], faces[:, 2], faces[:, 0],
                             faces[:, 2], faces[:, 0], faces[:, 1]])

    # Create a sparse matrix from row and column indices
    adjacency = csr_matrix(
        (np.ones_like(row_indices), (row_indices, col_indices)),
        shape=(n_vertices, n_vertices)
    )

    # Ensure the adjacency matrix is binary
    adjacency = (adjacency > 0).astype(int)

    return adjacency


def interpolate_data(original_mesh, downsampled_mesh, downsampled_data, adjacency_matrix=None,
                     max_iterations=10):
    """
    Interpolate vertex data from a downsampled mesh back to the original mesh using nearest
    neighbor matching and optional smoothing based on an adjacency matrix. Both meshes are
    expected to be nibabel Gifti objects.

    Parameters
    ----------
    original_mesh : nibabel.gifti.GiftiImage
        The original high-resolution mesh as a nibabel Gifti object from which 'downsampled_mesh'
        is derived.
    downsampled_mesh : nibabel.gifti.GiftiImage
        The downsampled version of the original mesh as a nibabel Gifti object.
    downsampled_data : array
        Data associated with the vertices of 'downsampled_mesh'.
    adjacency_matrix : sparse matrix, optional
        A vertex-by-vertex adjacency matrix of the original mesh. If None, it will be computed from
        the 'original_mesh'.
    max_iterations : int, optional
        The maximum number of iterations to perform for smoothing the interpolated data.

    Returns
    -------
    vertex_data : np.ndarray
        An array of interpolated data for each vertex in the 'original_mesh'. The data is initially
        interpolated using nearest neighbors and can be further refined through iterative smoothing.

    Notes
    -----
    The function first finds the nearest vertex in the 'downsampled_mesh' for each vertex in the
    'original_mesh' using a KD-tree. It directly assigns corresponding data values where a close
    match is found. The function iteratively adjusts data values at vertices without direct matches
    by averaging over neighbors.
    """

    if adjacency_matrix is None:
        adjacency_matrix = mesh_adjacency(original_mesh.darrays[1].data)

    original_vertices = original_mesh.darrays[0].data
    downsampled_vertices = downsampled_mesh.darrays[0].data

    # Build a KD-tree for the downsampled vertices
    tree = cKDTree(downsampled_vertices) # pylint: disable=not-callable

    # Preallocate the vertex data array
    vertex_data = np.full(len(original_vertices), np.nan)

    # Find the nearest neighbor in the downsampled mesh for each vertex in the original mesh
    distances, indices = tree.query(original_vertices, distance_upper_bound=1e-5)

    # Set the vertex data for vertices that match (distance is zero or very close)
    for i, (distance, index) in enumerate(zip(distances, indices)):
        if distance < 1e-6:  # Adjust this threshold as needed
            vertex_data[i] = downsampled_data[index]

    iteration = 0
    while np.isnan(vertex_data).any() and iteration < max_iterations:
        for i in np.where(np.isnan(vertex_data))[0]:
            _, neighbors, _ = find(adjacency_matrix[i, :])
            valid_neighbors = neighbors[~np.isnan(vertex_data[neighbors])]

            if valid_neighbors.size > 0:
                distances = cdist([original_vertices[i]], original_vertices[valid_neighbors])[0]
                weights = 1 / distances
                normalized_weights = weights / weights.sum()
                vertex_data[i] = np.dot(normalized_weights, vertex_data[valid_neighbors])
        iteration += 1

    return vertex_data


def split_fv(faces, vertices):
    """
    Split faces and vertices into connected pieces based on the connectivity of the faces.

    Parameters
    ----------
    faces : np.ndarray
        A 2D numpy array of faces, where each row represents a face and each element is an index to
        a vertex in `vertices`.
    vertices : np.ndarray
        A 2D numpy array of vertices, where each row represents a vertex.

    Returns
    -------
    fv_out : list of dict
        A list where each element is a dictionary with keys 'faces' and 'vertices'. Each dictionary
        represents a separately connected patch of the mesh.

    Examples
    --------
    >>> faces = np.array([[1, 2, 3], [1, 3, 4], [5, 6, 1], [7, 8, 9], [11, 10, 4]])
    >>> vertices = np.array([[2, 4], [2, 8], [8, 4], [8, 0], [0, 4], [2, 6], [2, 2], [4, 2],
    >>>                      [4, 0], [5, 2], [5, 0]])
    >>> split_patches = split_fv(faces, vertices)

    Notes
    -----
    Faces and vertices should be defined such that faces sharing a vertex reference the same vertex
    number. This function does not explicitly test for duplicate vertices at the same location.
    """

    num_faces = faces.shape[0]
    f_sets = np.zeros(num_faces, dtype=np.uint32)
    current_set = 0

    while np.any(f_sets == 0):
        current_set += 1
        next_avail_face = np.where(f_sets == 0)[0][0]
        open_vertices = faces[next_avail_face]

        while open_vertices.size:
            avail_face_inds = np.where(f_sets == 0)[0]
            is_member = np.isin(faces[avail_face_inds], open_vertices)
            avail_face_sub = np.where(np.any(is_member, axis=1))[0]
            f_sets[avail_face_inds[avail_face_sub]] = current_set
            open_vertices = np.unique(faces[avail_face_inds[avail_face_sub]])

    fv_out = []
    for set_num in range(1, current_set + 1):
        set_f = faces[f_sets == set_num]
        unique_vertices, new_vertex_indices = np.unique(set_f, return_inverse=True)
        fv_out.append({
            'faces': new_vertex_indices.reshape(set_f.shape),
            'vertices': vertices[unique_vertices]
        })

    return fv_out
