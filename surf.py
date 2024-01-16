import os
import copy
import subprocess
from joblib import Parallel, delayed

import numpy as np
import nibabel as nib
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkFiltersCore import vtkDecimatePro
from vtkmodules.util.numpy_support import vtk_to_numpy

from scipy.spatial import Delaunay
from csurf import compute_geodesic_distances
import scipy.io as sio


def _normit(N):
    """
    Normalize a numpy array of vectors.

    This function normalizes each row in the array N to have a unit length. If the length of a vector is below a certain
    threshold (machine epsilon), it is set to 1 to avoid division by zero.

    Parameters:
    N (ndarray): Array of vectors to be normalized. Each row represents a vector.

    Returns:
    ndarray: Normalized array of vectors where each row has unit length.
    """
    normN = np.sqrt(np.sum(N ** 2, axis=1))
    normN[normN < np.finfo(float).eps] = 1
    return N / normN[:, np.newaxis]


def _mesh_normal(vertices, faces):
    """
    Compute the normals for a mesh defined by vertices and faces.

    This function calculates the normal vectors for each vertex and face in a mesh. The normals are computed based on
    the cross product of vectors defined by the mesh faces.

    Parameters:
    vertices (ndarray): Array of vertices of the mesh. Each row represents a vertex.
    faces (ndarray): Array of faces of the mesh. Each row represents a face with indices to the vertices array.

    Returns:
    tuple: A tuple containing two ndarrays:
           - Nv: Normal vectors for each vertex.
           - Nf: Normal vectors for each face.
    """
    Nf = np.cross(
        vertices[faces[:, 1], :] - vertices[faces[:, 0], :],
        vertices[faces[:, 2], :] - vertices[faces[:, 0], :])
    Nf = _normit(Nf)

    Nv = np.zeros_like(vertices)
    for i in range(len(faces)):
        for j in range(3):
            Nv[faces[i, j], :] += Nf[i, :]

    C = vertices - np.mean(vertices, axis=0)
    if np.count_nonzero(np.sign(np.sum(C * Nv, axis=1))) > len(C) / 2:
        Nv = -Nv
        Nf = -Nf

    return Nv, Nf


def mesh_normals(vertices, faces, unit=False):
    """
    Calculate normals for a mesh using Delaunay triangulation or a fallback method.

    This function attempts to compute the normals of a mesh using Delaunay triangulation. If this fails (e.g., due to
    non-manifold geometry), it falls back to a custom method for normal calculation.

    Parameters:
    vertices (ndarray): Array of vertices of the mesh. Each row represents a vertex.
    faces (ndarray): Array of faces of the mesh. Each row represents a face with indices to the vertices array.
    unit (bool, optional): If True, the normals are normalized to unit length. Default is False.

    Returns:
    tuple: A tuple containing two ndarrays:
           - Nv: Normal vectors for each vertex.
           - Nf: Normal vectors for each face.
    """
    try:
        t = Delaunay(vertices)
        Nv = -t.vertex_normal
        Nf = -t.face_normal
    except:
        Nv, Nf = _mesh_normal(vertices, faces)

    if unit:
        Nv = _normit(Nv)
        Nf = _normit(Nf)

    return Nv, Nf


def create_surf_gifti(vertices, faces, normals=None):
    """
    Create a Gifti image object from surface mesh data.

    This function creates a GiftiImage object from the provided vertices, faces, and optional normals.
    The vertices and faces are required, while normals are optional. If normals are provided, they are
    added to the Gifti image. The function returns the GiftiImage object.

    Parameters:
    vertices (numpy.ndarray): Array of vertices. Each row represents a vertex with its x, y, z coordinates.
    faces (numpy.ndarray): Array of faces. Each row represents a face with three integers corresponding to vertex indices.
    normals (numpy.ndarray, optional): Array of vertex normals. Each row represents a normal vector corresponding to a vertex.

    Returns:
    nibabel.gifti.GiftiImage: The GiftiImage object created from the provided mesh data.

    Notes:
    - Vertex, face, and normal arrays should be NumPy arrays.
    - Vertices and normals should be in float32 format, and faces should be in int32 format.

    Example:
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
        nib.gifti.GiftiDataArray(data=vertices, intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']))
    new_gifti.add_gifti_data_array(
        nib.gifti.GiftiDataArray(data=faces, intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']))

    # If normals are provided and not empty, cast them to float32 and add them to the Gifti image
    if normals is not None and len(normals) > 0:
        normals = normals.astype(np.float32)
        new_gifti.add_gifti_data_array(
            nib.gifti.GiftiDataArray(data=normals, intent=nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']))

    return new_gifti


def remove_vertices(gifti_surf, vertices_to_remove):
    """
    Remove specified vertices from a Gifti surface and update the faces accordingly.

    This function modifies a Gifti surface by removing the specified vertices. It also updates
    the faces of the surface so that they only reference the remaining vertices. If normals
    are present in the surface, they are also updated to correspond to the new set of vertices.

    Parameters:
    gifti_surf (nibabel.gifti.GiftiImage): The Gifti surface object from which vertices will be removed.
    vertices_to_remove (array_like): An array of vertex indices to be removed from the surface.

    Returns:
    nibabel.gifti.GiftiImage: A new GiftiImage object with the specified vertices removed and faces updated.

    Notes:
    - The function assumes that the GiftiImage object contains at least two data arrays: one for vertices
      and one for faces. If normals are present, they are also updated.
    - Vertex indices in `vertices_to_remove` should be zero-based (following Python's indexing convention).
    - The returned GiftiImage object is a new object; the original `gifti_surf` object is not modified in place.

    Example:
    >>> import nibabel as nib
    >>> gifti_surf = nib.load('path_to_gifti_file.gii')
    >>> vertices_to_remove = np.array([0, 2, 5])  # Indices of vertices to remove
    >>> new_gifti_surf = remove_vertices(gifti_surf, vertices_to_remove)
    """

    # Extract vertices and faces from the gifti object
    vertices_data = [da for da in gifti_surf.darrays if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']][
        0]
    faces_data = [da for da in gifti_surf.darrays if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']][0]

    vertices = vertices_data.data
    faces = faces_data.data

    # Determine vertices to keep
    vertices_to_keep = np.setdiff1d(np.arange(vertices.shape[0]), vertices_to_remove)

    # Create new array of vertices
    new_vertices = vertices[vertices_to_keep, :]

    # Find which faces to keep - ones that point to kept vertices
    face_x = np.isin(faces[:, 0], vertices_to_keep)
    face_y = np.isin(faces[:, 1], vertices_to_keep)
    face_z = np.isin(faces[:, 2], vertices_to_keep)
    faces_to_keep = np.where(face_x & face_y & face_z)[0]

    # Re-index faces
    x_faces = faces[faces_to_keep, :].reshape(-1)
    idxs = np.searchsorted(vertices_to_keep, x_faces)
    new_faces = idxs.reshape(-1, 3)

    # Create new gifti object
    normals = None
    normals_data = [da for da in gifti_surf.darrays if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']]
    if normals_data:
        normals = normals_data[0].data[vertices_to_keep, :]
    new_gifti = create_surf_gifti(new_vertices, new_faces, normals=normals)

    return new_gifti


def downsample_single_surface(gifti_surf, ds_factor=0.1):
    """
    Downsample a Gifti surface using the VTK library.

    This function takes a Gifti surface defined by its vertices and faces, and downsamples it using
    VTK's vtkDecimatePro algorithm. The reduction ratio determines the degree of downsampling.
    The function returns the downsampled Gifti surface.

    Parameters:
    gifti_surf (nibabel.gifti.GiftiImage): The Gifti surface object from which vertices will be removed.
    reduction_ratio (float): The proportion of the mesh to remove. For example, a reduction ratio of 0.1
                             retains 90% of the original mesh.

    Returns:
    nibabel.gifti.GiftiImage: A new GiftiImage object with the downsampled surface.

    Notes:
    - The input faces array should be triangulated, i.e., each face should consist of exactly three vertex indices.
    - The VTK library is used for mesh decimation, which must be installed and properly configured.
    - The returned GiftiImage object is a new object; the original `gifti_surf` object is not modified in place.

    Example:
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
    # Assuming the mesh is triangulated, every fourth item is the size (3), followed by three vertex indices
    reduced_faces = face_data.reshape(-1, 4)[:, 1:4]

    new_gifti_surf = create_surf_gifti(reduced_vertices, reduced_faces)

    return new_gifti_surf


def iterative_downsample_single_surface(gifti_surf, ds_factor=0.1):
    """
   Iteratively downsample a single surface mesh to a target number of vertices.

   This function reduces the number of vertices in a surface mesh (in GIFTI format) to a specified fraction of its
   original size. Downsampling is performed iteratively until the target number of vertices is reached or closely
   approximated.

   Parameters:
   gifti_surf (nibabel.gifti.GiftiImage): The surface mesh to be downsampled, provided as a GIFTI image object.
   ds_factor (float, optional): The downsampling factor representing the target fraction of the original number of
                                vertices. Default is 0.1.

   Returns:
   nibabel.gifti.GiftiImage: The downsampled surface mesh as a GIFTI image object.

   Notes:
   - The downsampling process is iterative. In each iteration, the mesh is downsampled by a factor calculated to
     approach the target number of vertices.
   - If the calculated downsampling factor in an iteration equals or exceeds 1, the process is terminated to prevent
     upsampling or infinite loops.
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

    return current_surf


def downsample_multiple_surfaces(in_surfs, ds_factor):
    """
    Downampled multiple surface meshes using the VTK decimation algorithm.

    This function takes a list of input surface meshes (in Gifti format) and applies a dowsampling
    process to each surface. The downsampling is performed using VTK's vtkDecimatePro algorithm. The
    first surface in the list is downsampled, and its vertex mapping is then applied to all other
    surfaces in the list. The function returns a list of downsampled surface meshes.

    Parameters:
    in_surfs (list of nibabel.gifti.GiftiImage): Input Gifti surface meshes to be downsampled.
    ratio (float): The reduction ratio for the downsampling process. For example, a ratio of 0.1
                   implies that the mesh will be reduced to 90% of its original size.

    Returns:
    list of nibabel.gifti.GiftiImage: List of downsampled Gifti surface meshes.

    Notes:
    - The function prints the percentage of vertices retained in the first surface after downsampling.
    - If normals are present in the input surfaces, they are also downsampled and mapped to the new surfaces.
    - The resulting surfaces maintain the original topology and are suitable for visualization and further processing.

    Example:
    >>> import nibabel as nib
    >>> in_surfs = [nib.load('path/to/input_surf1.gii'), nib.load('path/to/input_surf2.gii')]
    >>> ratio = 0.1
    >>> out_surfs = downsample_multiple_surfaces(in_surfs, ds_factor)
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
        f"{(1 - np.mean(decim_orig_dist > 0)) * 100}% of the vertices in the decimated surface belong to the original surface.")

    reduced_normals = None
    if len(primary_surf.darrays) > 2 and primary_surf.darrays[2].intent == nib.nifti1.intent_codes[
        'NIFTI_INTENT_VECTOR']:
        reduced_normals = primary_surf.darrays[2].data[orig_vert_idx]

    # Save the downsampled primary surface with normals
    ds_surf = create_surf_gifti(reduced_vertices, reduced_faces, normals=reduced_normals)
    out_surfs.append(ds_surf)

    # Process other surfaces
    for i in range(1, len(in_surfs)):
        surf = in_surfs[i]

        reduced_normals = None
        if len(surf.darrays) > 2 and surf.darrays[2].intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']:
            reduced_normals = surf.darrays[2].data[orig_vert_idx]

        ds_surf = create_surf_gifti(surf.darrays[0].data[orig_vert_idx, :], reduced_faces, normals=reduced_normals)
        out_surfs.append(ds_surf)
    return out_surfs


def combine_surfaces(surfaces):
    """
    Combine multiple surface meshes into a single surface mesh.

    This function takes a list of Gifti surface meshes and combines them into a single surface mesh.
    It concatenates the vertices, faces, and normals (if present) from each surface. The faces are
    re-indexed appropriately to maintain the correct references to the combined vertex array.

    Parameters:
    surfaces (list of nibabel.gifti.GiftiImage): List of Gifti surface meshes to be combined.

    Returns:
    nibabel.gifti.GiftiImage: A single combined Gifti surface mesh.

    Notes:
    - The vertices, faces, and normals (if present) from each surface are concatenated.
    - The faces are re-indexed to reference the correct vertices in the combined vertex array.
    - If normals are present in any of the input surfaces, they are also combined.

    Raises:
    ValueError: If the vertex or face arrays do not have the expected dimensions.

    Example:
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
            [da.data for da in mesh.darrays if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']])
        faces = np.concatenate(
            [da.data for da in mesh.darrays if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']])

        # Check for normals
        normal_arrays = [da.data for da in mesh.darrays if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']]
        normals = np.concatenate(normal_arrays) if normal_arrays else np.array([])

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Vertices array should have shape [n, 3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Faces array should have shape [n, 3]")

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

    combined_surf = create_surf_gifti(combined_vertices, combined_faces, normals=combined_normals)
    return combined_surf


def compute_dipole_orientations(method, layer_names, subject_out_dir, fixed=True):
    """
    Compute dipole orientations for cortical layers using different methods.

    Parameters:
    method (str): Method for computing dipole orientations ('link_vector', 'ds_surf_norm', 'orig_surf_norm', or 'cps').
    layer_names (list): Names of the cortical layers.
    subject_out_dir (str): Directory where the surface files are stored.
    fixed (bool, optional): Flag to ensure that orientation of corresponding vertices across layers is the same (True
                            by default)

    Returns:
    numpy.ndarray: An array of dipole orientations for each vertex in each layer.

    Raises:
    ValueError: If the number of vertices in pial and white surfaces do not match.
    """

    if method == 'link_vector':
        # Method: Use link vectors between pial and white surfaces as dipole orientations
        # Load downsampled pial and white surfaces
        pial_surf = nib.load(os.path.join(subject_out_dir, 'pial.ds.gii'))
        white_surf = nib.load(os.path.join(subject_out_dir, 'white.ds.gii'))

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
                in_surf_path = os.path.join(subject_out_dir, f'{layer_name}.ds.gii')
                surf = nib.load(in_surf_path)
                vtx_norms, _ = mesh_normals(surf.darrays[0].data, surf.darrays[1].data, unit=True)
            orientations.append(vtx_norms)
        orientations = np.array(orientations)

    elif method == 'orig_surf_norm':
        # Method: Use normals of the original surfaces, mapped to downsampled surfaces
        orientations = []
        for l_idx, layer_name in enumerate(layer_names):
            if l_idx == 0 or not fixed:
                in_surf_path = os.path.join(subject_out_dir, f'{layer_name}.gii')
                orig_surf = nib.load(in_surf_path)
                ds_surf_path = os.path.join(subject_out_dir, f'{layer_name}.ds.gii')
                ds_surf = nib.load(ds_surf_path)
                kdtree = KDTree(orig_surf.darrays[0].data)
                _, orig_vert_idx = kdtree.query(ds_surf.darrays[0].data, k=1)
                vtx_norms, _ = mesh_normals(orig_surf.darrays[0].data, orig_surf.darrays[1].data, unit=True)
            orientations.append(vtx_norms[orig_vert_idx, :])
        orientations = np.array(orientations)

    elif method == 'cps':
        # Method: Use cortical patch statistics for normals
        orientations = []
        for l_idx, layer_name in enumerate(layer_names):
            if l_idx == 0 or not fixed:
                in_surf_path = os.path.join(subject_out_dir, f'{layer_name}.gii')
                orig_surf = nib.load(in_surf_path)
                ds_surf_path = os.path.join(subject_out_dir, f'{layer_name}.ds.gii')
                ds_surf = nib.load(ds_surf_path)
                kdtree = KDTree(ds_surf.darrays[0].data)
                _, ds_vert_idx = kdtree.query(orig_surf.darrays[0].data, k=1)
                orig_vtx_norms, _ = mesh_normals(orig_surf.darrays[0].data, orig_surf.darrays[1].data, unit=True)
                vtx_norms, _ = mesh_normals(ds_surf.darrays[0].data, ds_surf.darrays[1].data, unit=True)
                for v_idx in range(vtx_norms.shape[0]):
                    orig_idxs = np.where(ds_vert_idx == v_idx)[0]
                    if len(orig_idxs):
                        vtx_norms[v_idx, :] = np.mean(orig_vtx_norms[orig_idxs, :], axis=0)
                vtx_norms = _normit(vtx_norms)
            orientations.append(vtx_norms)
        orientations = np.array(orientations)

    return orientations


def postprocess_freesurfer_surfaces(subj_id,
                                    out_dir,
                                    out_fname,
                                    n_surfaces=11,
                                    ds_factor=0.1,
                                    orientation='link_vector',
                                    fix_orientation=True,
                                    remove_deep=True):
    """
    Process and combine FreeSurfer surface meshes for a subject.

    This function processes FreeSurfer surface meshes for a given subject by creating intermediate surfaces,
    adjusting for RAS offset, removing deep vertices, combining hemispheres, downsampling, and computing link vectors.
    The resulting surfaces are combined and saved to a specified output file.

    Parameters:
    subj_id (str): Subject ID corresponding to the FreeSurfer subject directory.
    out_dir (str): Output directory where the processed files will be saved.
    out_fname (str): Filename for the final combined surface mesh.
    n_surfaces (int, optional): Number of intermediate surfaces to create between white and pial surfaces.
    ds_factor (float, optional): Downsampling factor for surface decimation.
    orientation (str, optional): Method to compute orientation vectors ('link_vector' for pial-white link,
                                 'ds_surf_norm' for downsampled surface normals, 'orig_surf_norm' for original
                                 surface normals, and 'cps' for cortical patch statistics).
    fix_orientation (bool, optional): Flag to ensure that orientation of corresponding vertices across layers is the
                                      same (True by default)
    remove_deep (bool, optional): Flag to remove vertices located in deep regions (labeled as 'unknown').

    Notes:
    - This function assumes the FreeSurfer 'SUBJECTS_DIR' environment variable is set.
    - Surfaces are processed in Gifti format and combined into a single surface mesh.

    Example:
    >>> postprocess_freesurfer_surfaces('subject1', '/path/to/output', 'combined_surface.gii')
    """

    hemispheres = ['lh', 'rh']
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')

    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    subject_out_dir = os.path.join(out_dir, subj_id)
    layers = np.linspace(1, 0, n_surfaces)

    ## Create intermediate surfaces if needed

    def process_layer(layer, hemispheres, fs_subject_dir):
        if layer == 1:
            return 'pial'
        elif layer > 0 and layer < 1:
            layer_name = '{:.3f}'.format(layer)
            for hemi in hemispheres:
                wm_file = os.path.join(fs_subject_dir, 'surf', '{}.white'.format(hemi))
                out_file = os.path.join(fs_subject_dir, 'surf', '{}.{}'.format(hemi, layer_name))
                if not os.path.exists(out_file):
                    cmd = ['mris_expand', '-thickness', wm_file, '{}'.format(layer), out_file]
                    print(' '.join(cmd))
                    subprocess.run(cmd)
            return layer_name
        elif layer == 0:
            return 'white'

    layer_names = Parallel(n_jobs=-1)(delayed(process_layer)(layer, hemispheres, fs_subject_dir) for layer in layers)

    ## Compute RAS offset
    # Define the path to the MRI file
    ras_off_file = os.path.join(fs_subject_dir, 'mri', 'orig.mgz')

    # Execute the shell command to get RAS offset
    command = f"mri_info --cras {ras_off_file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    # Parse the output
    cols = out.decode().split()
    ras_offset = np.array([float(cols[0]), float(cols[1]), float(cols[2])])

    # Print the result
    print(ras_offset)

    ## Convert to gifti, adjust for RAS offset, and remove deep vertices
    surfaces_to_process = copy.copy(layer_names)
    surfaces_to_process.append('inflated')
    for surface_name in surfaces_to_process:
        for hemi in hemispheres:
            # Construct the original and new file names
            orig_name = os.path.join(fs_subject_dir, 'surf', f'{hemi}.{surface_name}')
            new_name = os.path.join(subject_out_dir, f'{hemi}.{surface_name}.gii')

            # Convert the surface file to Gifti format
            subprocess.run(['mris_convert', orig_name, new_name])

            # Load the Gifti file
            g = nib.load(new_name)

            # Set transformation matrix to identity
            g.affine = np.eye(4)

            # Adjust for RAS offset
            n_vertices = 0
            for da in g.darrays:
                if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
                    da.data += ras_offset
                    n_vertices = da.data.shape[0]

            annotation = os.path.join(fs_subject_dir, 'label', f'{hemi}.aparc.annot')
            label, ctab, names = nib.freesurfer.read_annot(annotation)

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
                g = remove_vertices(g, np.array(vertices_to_remove))

            # Save the modified Gifti file
            nib.save(g, new_name)

    ## Combine hemispheres
    for surface_name in surfaces_to_process:
        # Load left and right hemisphere surfaces
        lh_fname = os.path.join(subject_out_dir, f'lh.{surface_name}.gii')
        lh = nib.load(lh_fname)
        rh_fname = os.path.join(subject_out_dir, f'rh.{surface_name}.gii')
        rh = nib.load(rh_fname)
        if surface_name == 'inflated':
            lh_width = np.max(lh.darrays[0].data[:, 0])-np.min(lh.darrays[0].data[:, 0])
            rh.darrays[0].data[:, 0] = rh.darrays[0].data[:, 0] + np.max(lh.darrays[0].data[:, 0]) + (.5*lh_width)

        # Combine the surfaces
        combined = combine_surfaces([lh, rh])
        combined_fname = os.path.join(subject_out_dir, f'{surface_name}.gii')
        nib.save(combined, combined_fname)

    ## Downsample surfaces at the same time
    # Get list of surfaces
    in_surfs = []
    for surface_name in surfaces_to_process:
        in_surf_fname = os.path.join(subject_out_dir, f'{surface_name}.gii')
        in_surf = nib.load(in_surf_fname)
        in_surfs.append(in_surf)

    # Downsample multiple surfaces
    out_surfs = downsample_multiple_surfaces(in_surfs, ds_factor)
    for surface_name, out_surf in zip(surfaces_to_process, out_surfs):
        out_surf_path = os.path.join(subject_out_dir, f'{surface_name}.ds.gii')
        nib.save(out_surf, out_surf_path)

    ## Compute dipole orientations
    orientations = compute_dipole_orientations(orientation, layer_names, subject_out_dir, fixed=fix_orientation)

    base_fname = f'ds.{orientation}'
    if fix_orientation:
        base_fname = f'{base_fname}.fixed'
    for l_idx, layer_name in enumerate(layer_names):
        in_surf_path = os.path.join(subject_out_dir, f'{layer_name}.ds.gii')
        surf = nib.load(in_surf_path)

        # Set these link vectors as the normals for the downsampled surface
        surf.add_gifti_data_array(nib.gifti.GiftiDataArray(data=orientations[l_idx, :, :],
                                                           intent=nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']))

        # Save the modified downsampled surface with link vectors as normals
        out_surf_path = os.path.join(subject_out_dir, f'{layer_name}.{base_fname}.gii')
        nib.save(surf, out_surf_path)

    ## Combine layers
    all_surfs = []
    for layer_name in layer_names:
        surf_path = os.path.join(subject_out_dir, f'{layer_name}.{base_fname}.gii')
        surf = nib.load(surf_path)
        all_surfs.append(surf)

    combined = combine_surfaces(all_surfs)
    nib.save(combined, os.path.join(subject_out_dir, out_fname))


def compute_mesh_area(gifti_surf, PF=False):
    """
    Compute the surface area of a triangle mesh.

    Parameters:
    M (dict or np.ndarray): A dictionary with 'vertices' (mx3) and 'faces' (nx3) as keys
                            or a 3xm array of edge distances.
    PF (bool): If True, return the surface area per face. Default is False.

    Returns:
    float or np.ndarray: Total surface area or an array of areas per face.
    """
    # Extract vertices and faces
    vertices, faces = gifti_surf.darrays[0].data, gifti_surf.darrays[1].data

    # Compute edge lengths of each triangle
    A = np.linalg.norm(vertices[faces[:, 1], :] - vertices[faces[:, 0], :], axis=1)
    B = np.linalg.norm(vertices[faces[:, 2], :] - vertices[faces[:, 1], :], axis=1)
    C = np.linalg.norm(vertices[faces[:, 2], :] - vertices[faces[:, 0], :], axis=1)

    # Heron's formula for area
    s = (A + B + C) / 2
    area = np.sqrt(s * (s - A) * (s - B) * (s - C))

    return np.sum(area) if not PF else area


def compute_mesh_distances(vertices, faces):
    """
    Compute the pairwise Euclidean distances between connected vertices in a mesh.

    This function calculates the distances between each pair of connected vertices in a mesh defined by its vertices and
    faces. The result is a sparse matrix where each entry (i, j) represents the distance between vertices i and j.

    Parameters:
    vertices (ndarray): Array of vertices of the mesh. Each row represents a vertex as a 3D point.
    faces (ndarray): Array of faces of the mesh. Each row represents a face with indices to the vertices array.

    Returns:
    csr_matrix: A sparse matrix in Compressed Sparse Row (CSR) format containing the pairwise distances between
    connected vertices.

    Notes:
    - The function calculates distances only for directly connected vertices (i.e., vertices that share an edge).
    - The resulting distance matrix is symmetric, as the distance from vertex i to j is the same as from j to i.
    - The matrix is sparse, containing non-zero entries only for pairs of connected vertices.
    """
    # Compute the differences
    d0 = vertices[faces[:, 0], :] - vertices[faces[:, 1], :]
    d1 = vertices[faces[:, 1], :] - vertices[faces[:, 2], :]
    d2 = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]

    # Flatten the arrays for creating a COO matrix
    rows = np.hstack([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.hstack([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.sqrt(np.sum(np.vstack([d0 ** 2, d1 ** 2, d2 ** 2]), axis=1))

    # Create the sparse matrix
    D_coo = coo_matrix((data, (rows, cols)), shape=(vertices.shape[0], vertices.shape[0]))

    # Convert to CSR and symmetrize
    D_csr = (D_coo.tocsr() + D_coo.transpose().tocsr()) / 2
    return D_csr


def smoothmesh_multilayer_mm(meshname, fwhm, n_layers, redo=False, n_jobs=-1):
    """
    Compute smoothed matrices for a multilayer mesh.

    Parameters:
    meshname (str): Filename of the mesh (GIFTI format).
    fwhm (float): Full width at half maximum for smoothing.
    n_layers (int): Number of layers in the mesh.
    redo (bool): Recompute matrices if already exist.

    Returns:
    str: Filename of the saved matrix.
    """
    gifti_surf = nib.load(meshname)
    vertices = gifti_surf.darrays[0].data
    faces = gifti_surf.darrays[1].data
    Ns = vertices.shape[0]
    Ns_per_layer = Ns // n_layers
    vertspace = np.mean(np.sqrt(compute_mesh_area(gifti_surf, PF=True)))
    spacing = fwhm / vertspace

    print(f'FWHM of {fwhm:3.2f} is approx {spacing:3.2f} times vertex spacing')

    # Extract the directory and the base name without extension
    mesh_dir = os.path.dirname(meshname)
    mesh_base_name, _ = os.path.splitext(os.path.basename(meshname))

    smoothmeshname = os.path.join(mesh_dir, f'FWHM{fwhm:3.2f}_{mesh_base_name}.mat')
    if os.path.exists(smoothmeshname) and not redo:
        return smoothmeshname

    sigma2 = (fwhm / 2.355) ** 2

    distance_matrix = compute_mesh_distances(vertices.astype(np.float64), faces)

    def process_vertex(j):
        rows_vertex, cols_vertex, data_vertex = [], [], []
        for l in range(n_layers):
            D_layer = distance_matrix[l * Ns_per_layer:(l + 1) * Ns_per_layer, l * Ns_per_layer:(l + 1) * Ns_per_layer]
            source_indices = [j % Ns_per_layer]
            dist = compute_geodesic_distances(D_layer, source_indices, max_dist=fwhm)

            mask = dist <= fwhm
            q = np.exp(-(dist[mask] ** 2) / (2 * sigma2))
            q = q * (q > np.exp(-8))
            q /= np.sum(q)

            rows_vertex.extend(np.full(sum(mask), l * Ns_per_layer + j % Ns_per_layer))
            cols_vertex.extend(np.where(mask)[0] + l * Ns_per_layer)
            data_vertex.extend(q)
        return rows_vertex, cols_vertex, data_vertex

    # Parallel computation for each vertex using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_vertex)(j) for j in range(Ns_per_layer)
    )

    # Aggregate results from all vertices
    rows, cols, data = zip(*results)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    QG = coo_matrix((data, (rows, cols)), shape=(Ns, Ns)).tocsr()

    # Add 1 for matlab
    faces = faces + 1

    sio.savemat(smoothmeshname, {'QG': QG, 'faces': faces}, do_compression=True)

    return smoothmeshname
