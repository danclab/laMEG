"""
Visualization and color mapping utilities for laminar and cortical surface data.

This module provides a set of functions for visualizing laminar MEG and cortical data, including
color mapping utilities, 3D surface rendering, and Current Source Density (CSD) plotting. It
integrates seamlessly with FreeSurfer surfaces, laMEG outputs, and matplotlib-based plotting
pipelines.

Main Features
--------------
- **Color Mapping**:
  - Convert numerical data into RGB, RGBA, or hexadecimal color representations.
  - Support for diverging, linear, and logarithmic normalization schemes.
  - Utilities for encoding RGB triplets as 32-bit integers for compact graphical representation.

- **Surface Visualization**:
  - Render 3D cortical surfaces using the `k3d` engine.
  - Overlay vertex-wise statistical maps, curvature-based shading, and anatomical markers.
  - Interactive camera controls and scene export support.

- **Laminar Data Visualization**:
  - Visualization of Current Source Density (CSD) profiles across cortical layers.
  - Optional display of laminar boundaries and custom color scaling.

Functions
---------
data_to_rgb(data, n_bins, cmap, vmin, vmax, vcenter=0.0, ret_map=False, norm='TS')
    Map numerical data to RGBA colors using matplotlib colormaps with specified normalization.

rgbtoint(rgb)
    Convert an RGB color triplet to a 32-bit integer color code.

color_map(data, cmap, vmin, vmax, n_bins=1000, vcenter=0, norm='TS')
    Map numerical data to hexadecimal color codes suitable for visualization.

show_surface(surf_set, layer_name='inflated', stage='combined', ...)
    Render a cortical or laminar surface with optional curvature, color, and marker overlays.

plot_csd(csd, times, axis, colorbar=True, cmap='RdBu_r', ...)
    Plot a 2D laminar Current Source Density (CSD) profile over time.

Notes
-----
- All functions are designed to integrate with laMEG's surface and layer representations.
- Surface visualization relies on `k3d` for GPU-accelerated rendering.
- The module assumes FreeSurfer-compatible surface files and curvature data.
"""


import os
import collections
import warnings

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree # pylint: disable=E0611
import nibabel as nib

from matplotlib import cm, colors
import matplotlib.pyplot as plt
import k3d

from lameg.invert import check_inversion_exists
from lameg.surf import _normit, _estimate_rigid_transform, _apply_affine

warnings.filterwarnings(
    "ignore", message="^.*A coerced copy has been created.*$"
)


def data_to_rgb(data, n_bins, cmap, vmin, vmax, vcenter=0.0, ret_map=False, norm="TS"):
    """
    Map numerical data values to RGB colors using a specified matplotlib colormap.

    This function normalizes input data according to a chosen scaling mode (two-slope, linear,
    or logarithmic), bins the data into a specified number of intervals, and maps each bin to
    an RGB color value. It optionally returns the colormap object for use in associated plots
    (e.g., colorbars).

    Parameters
    ----------
    data : array_like, shape (n,)
        One-dimensional array of numerical values to be color-mapped.
    n_bins : int
        Number of histogram bins used to discretize the data range.
    cmap : str
        Name of the matplotlib colormap (e.g., `'viridis'`, `'RdBu_r'`).
    vmin : float
        Minimum data value for normalization.
    vmax : float
        Maximum data value for normalization.
    vcenter : float, optional
        Center value for the colormap when using two-slope normalization (`norm="TS"`).
        Default is 0.0.
    ret_map : bool, optional
        If True, return the matplotlib `ScalarMappable` object along with the mapped RGB values.
        Default is False.
    norm : {'TS', 'N', 'LOG'}, optional
        Normalization type:
        - `'TS'`: Two-slope normalization (diverging colormap, zero-centered)
        - `'N'`: Linear normalization
        - `'LOG'`: Logarithmic normalization
        Default is `'TS'`.

    Returns
    -------
    color_mapped : ndarray, shape (n, 4)
        Array of RGBA color values corresponding to each input datum.
    scalar_map : matplotlib.cm.ScalarMappable, optional
        The colormap object used for mapping (returned only if `ret_map=True`).

    Raises
    ------
    ValueError
        If `norm` is not one of `'TS'`, `'N'`, or `'LOG'`.

    Notes
    -----
    - The function assigns each data point the color of its corresponding histogram bin.
    - The `'TS'` normalization is best suited for data centered around a neutral reference value
      (e.g., 0), producing balanced diverging colormaps.
    - The returned colors can be directly used for visualizing scalar data distributions or
      as vertex colors in surface plots.
    """

    if norm == "TS":
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    elif norm == "N":
        divnorm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == "LOG":
        divnorm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        raise ValueError("norm must be TS, N, or LOG")

    scalar_map = cm.ScalarMappable(divnorm, cmap=cmap)
    bins = np.histogram_bin_edges(data, bins=n_bins)
    bin_ranges = list(zip(bins[:-1], bins[1:]))
    color_mapped = np.zeros((data.shape[0], 4))
    for br_ix, bin_range in enumerate(bin_ranges):
        map_c = (data >= bin_range[0]) & (data <= bin_range[1])
        color_mapped[map_c, :] = scalar_map.to_rgba(bins[1:][br_ix])

    if not ret_map:
        return color_mapped
    return color_mapped, scalar_map


def rgbtoint(rgb):
    """
    Convert an RGB triplet to a 32-bit integer color representation.

    This function encodes red, green, and blue components (0-255) into a single 32-bit integer,
    enabling compact storage or low-level graphical interfacing. The resulting integer can be
    decoded back into RGB components via bitwise operations.

    Parameters
    ----------
    rgb : array_like of int, shape (3,)
        RGB color triplet with values in the range [0, 255].

    Returns
    -------
    color : int
        32-bit integer encoding of the input RGB color.

    Raises
    ------
    AssertionError
        If any RGB component is outside the valid range [0, 255].

    Notes
    -----
    - Values are packed in big-endian order: `R << 16 | G << 8 | B`.
    - Useful for transferring color information to graphics libraries that use integer encoding.
    """

    if isinstance(rgb, list):
        rgb = np.array(rgb)
    assert np.all((0 <= rgb) & (rgb <= 255)), "Requires integer RGB (values 0-255)"
    color = 0
    for rgb_val in rgb:
        color = (color << 8) + int(rgb_val)
    return color


def color_map(data, cmap, vmin, vmax, n_bins=1000, vcenter=0, norm="TS"):
    """
    Map numerical data to hexadecimal color codes using a specified matplotlib colormap.

    This function normalizes numerical data using a chosen normalization scheme, maps values to
    RGB colors via a matplotlib colormap, and converts them to hexadecimal format suitable for
    visualization (e.g., in surface plots or colorbar annotations).

    Parameters
    ----------
    data : array_like, shape (n,)
        One-dimensional array of numerical values to be color-mapped.
    cmap : str
        Name of the matplotlib colormap (e.g., `'viridis'`, `'RdBu_r'`).
    vmin : float
        Minimum data value for normalization.
    vmax : float
        Maximum data value for normalization.
    n_bins : int, optional
        Number of histogram bins used to discretize the data range. Default is 1000.
    vcenter : float, optional
        Center value for diverging normalization (`norm='TS'`). Default is 0.
    norm : {'TS', 'N', 'LOG'}, optional
        Type of normalization:
        - `'TS'`: Two-slope (diverging) normalization.
        - `'N'`: Linear normalization.
        - `'LOG'`: Logarithmic normalization.
        Default is `'TS'`.

    Returns
    -------
    map_colors : ndarray, shape (n,)
        Array of hexadecimal color codes corresponding to the input data values.
    cmap : matplotlib.colors.Colormap
        The matplotlib colormap object used for mapping.

    Notes
    -----
    - Internally uses `data_to_rgb()` to compute RGBA values, which are then scaled and converted
      to hexadecimal color codes.
    - The `'TS'` normalization is suited for data centered around a neutral reference (e.g., zero).
    - Useful for generating vertex-wise color encodings or colorbars consistent with matplotlib
      colormaps.
    """

    map_colors, c_map = data_to_rgb(
        data, n_bins, cmap, vmin, vmax,
        vcenter=vcenter, ret_map=True, norm=norm
    )
    map_colors = map_colors[:, :3] * 255
    map_colors = map_colors.astype(int)
    return map_colors, c_map


# pylint: disable=R0915, R0912
def show_surface(
        surf_set,
        layer_name='inflated',
        stage='ds',
        hemi=None,
        orientation=None,
        fixed=None,
        color=None,
        grid=False,
        menu=False,
        vertex_colors=None,
        info=False,
        camera_view=None,
        height=512,
        opacity=1.0,
        marker_coords=None,
        marker_vertices=None,
        marker_size=1,
        marker_color=None,
        plot_curvature=True):
    """
    Render a 3D cortical surface with optional curvature, vertex-wise data, and marker overlays.

    This function visualizes a laminar or pial surface loaded from a `LayerSurfaceSet` object
    using the `k3d` rendering backend. It supports curvature shading, vertex-specific color
    mapping, and visualization of fiducial or region markers, providing a flexible interactive
    3D view of laminar or anatomical surfaces.

    Parameters
    ----------
    surf_set : LayerSurfaceSet
        Layer surface container providing access to laminar or pial surfaces.
    layer_name : str, optional
        Name of the layer to render (e.g., `'pial'`, `'white'`, `'inflated'`). Default is
        `'inflated'`.
    stage : {'combined', 'ds'}, optional
        Rendering stage: `'combined'` for merged hemispheres, `'ds'` for downsampled single
        surfaces. Default is `'combined'`.
    hemi : {'lh', 'rh'}, optional
        Hemisphere to render. Required if `stage='ds'`. Default is None.
    orientation : str or None, optional
        Downsampling orientation identifier used when `stage='ds'`. Ignored otherwise.
    fixed : bool or None, optional
        Whether to load fixed or adaptive orientation surfaces when `stage='ds'`. Default is None.
    color : array_like of int, shape (3,), optional
        Base RGB color of the surface (0-255). Default is `[166, 166, 166]`.
    grid : bool, optional
        Display background grid. Default is False.
    menu : bool, optional
        Display interactive visualization menu. Default is False.
    vertex_colors : array_like, optional
        Vertex-wise colors specified as either RGB(A) values or packed 32-bit integers. Default is
        None.
    info : bool, optional
        If True, prints surface metadata (e.g., vertex count). Default is False.
    camera_view : array_like, optional
        Predefined camera position and orientation. If None, automatic camera fitting is used.
        Default is None.
    height : int, optional
        Plot height in pixels. Default is 512.
    opacity : float, optional
        Surface opacity (0 = transparent, 1 = opaque). Default is 1.0.
    marker_coords : array_like, optional
        Marker coordinates (n × 3) in the same space as the surface vertices.
    marker_vertices : array_like of int, optional
        Vertex indices to highlight as markers (used instead of explicit coordinates if provided).
    marker_size : float or sequence, optional
        Size (radius) of marker spheres. Default is 1.0.
    marker_color : array_like, shape (3,) or (n, 3), optional
        RGB color(s) of marker spheres (0-255). Default is `[255, 0, 0]`.
    plot_curvature : bool, optional
        If True, overlays sulcal curvature shading derived from FreeSurfer `.curv` files. Default
        is True.

    Returns
    -------
    plot : k3d.plot.Plot
        Interactive `k3d` plot object containing the rendered surface and optional overlays.

    Notes
    -----
    - Supports both per-vertex color overlays (e.g., statistical maps) and curvature-based shading.
    - Vertex colors with an alpha channel (RGBA) use transparency to blend with curvature shading.
    - Curvature values are mapped to grayscale for sulcal and gyral regions.
    - If both `marker_vertices` and `marker_coords` are provided, the function prioritizes
      `marker_vertices`.
    - Compatible with `fsaverage`-aligned surfaces and laminar surface sets generated by laMEG.
    """


    if color is None:
        color = [166, 166, 166]
    if marker_color is None:
        marker_color = [255, 0, 0]
    if marker_vertices is not None and marker_coords is not None:
        warnings.warn("Both `marker_vertices` and `marker_coords` specified - using "
                      "`marker_vertices` only.")
        marker_coords = None  # ignore manual coord
    if np.isscalar(marker_vertices):
        marker_vertices=[marker_vertices]

    base_color = rgbtoint(color)
    marker_color = np.array(marker_color).reshape(-1, 3)

    surface = surf_set.load(
        layer_name=layer_name,
        stage=stage,
        hemi=hemi,
        orientation=(orientation if stage == 'ds' else None),
        fixed=(fixed if stage == 'ds' else None)
    )
    try:
        vertices, faces, _ = surface.agg_data()
    except ValueError:
        vertices, faces = surface.agg_data()

    if marker_vertices is not None:
        marker_coords = vertices[np.asarray(marker_vertices, dtype=int)]

    mesh = k3d.mesh(vertices, faces, side="double", color=base_color, opacity=opacity)
    cam_autofit = camera_view is None
    plot = k3d.plot(
        grid_visible=grid, menu_visibility=menu, camera_auto_fit=cam_autofit, height=height
    )
    plot += mesh

    if plot_curvature:
        # Load curvature and full-resolution pial vertices
        lh_curv = nib.freesurfer.read_morph_data(os.path.join(surf_set.surf_dir, 'lh.curv'))
        rh_curv = nib.freesurfer.read_morph_data(os.path.join(surf_set.surf_dir, 'rh.curv'))
        lh_fr = surf_set.load(layer_name='pial', stage='converted', hemi='lh').darrays[0].data
        rh_fr = surf_set.load(layer_name='pial', stage='converted', hemi='rh').darrays[0].data
        full_verts = np.vstack([lh_fr, rh_fr])
        map_surf = surf_set.load(
            layer_name='pial',
            stage=stage,
            hemi=hemi,
            orientation=(orientation if stage == 'ds' else None),
            fixed=(fixed if stage == 'ds' else None), )
        map_vertices, _, *_ = map_surf.agg_data()
        map_vertices = np.asarray(map_vertices, dtype=np.float32)
        # Build mapping to current surface vertices
        tree = cKDTree(full_verts)
        _, nearest_idx = tree.query(map_vertices, k=1)
        combined_curv = np.concatenate([lh_curv, rh_curv])
        ds_curv = combined_curv[nearest_idx]
        sulc_rgb = np.zeros((len(ds_curv), 3), dtype=int)
        sulc_rgb[ds_curv <= 0, :] = 166
        sulc_rgb[ds_curv > 0, :] = 64

    if hasattr(vertex_colors, "__iter__"):
        vertex_colors = np.asarray(vertex_colors)

        # detect RGBA input (shape Nx4)
        if vertex_colors.ndim == 2 and vertex_colors.shape[1] == 4:
            rgb = vertex_colors[:, :3]  # * 255
            alpha = vertex_colors[:, 3] > 128  # binary mask
            rgb = rgb.astype(int)

            if plot_curvature:
                # replace transparent vertices
                rgb[~alpha, :] = sulc_rgb[~alpha, :]
            else:
                rgb[~alpha, :] = np.array(color)
        else:
            rgb = vertex_colors

        # convert to packed ints
        vertex_colors = [rgbtoint(c) for c in rgb]
    elif plot_curvature:
        vertex_colors = [rgbtoint(c) for c in sulc_rgb]

    if hasattr(vertex_colors, "__iter__"):
        mesh.colors = vertex_colors
    else:
        pass

    if marker_coords is not None:
        marker_coords = np.array(marker_coords).reshape(-1, 3)
        for c_idx in range(marker_coords.shape[0]):
            coord = marker_coords[c_idx, :]
            size = marker_size
            if isinstance(marker_size, collections.abc.Sequence):
                size = marker_size[c_idx]
            if marker_color.shape[0] > 1:
                color = marker_color[c_idx, :]
            else:
                color = marker_color[0, :]
            color = list(map(int, color))
            point = k3d.points(
                coord,
                point_size=size,
                color=rgbtoint(color)
            )
            plot += point

    if camera_view is not None:
        plot.camera = camera_view

    plot.display()
    if info:
        print(vertices.shape[0], "vertices")

    return plot


def _sensor_frame(normal, direction=None):
    """
    Construct an orthonormal in-plane frame for a sensor glyph.
    """
    normal = _normit(np.asarray(normal, dtype=float)[None, :])[0]

    if direction is not None and np.all(np.isfinite(direction)):
        direction = np.asarray(direction, dtype=float)
        direction = direction - np.dot(direction, normal) * normal
        if np.linalg.norm(direction) > np.finfo(float).eps:
            u_vec = direction / np.linalg.norm(direction)
            v_vec = np.cross(normal, u_vec)
            v_vec = v_vec / np.linalg.norm(v_vec)
            return u_vec, v_vec

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    u_vec = np.cross(normal, ref)
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    return u_vec, v_vec


def _make_circle_sensor_mesh(positions, orientations, radius=0.005, n_circle=24):
    """
    Build one batched triangular mesh containing one filled disc per sensor.
    """
    positions = np.asarray(positions, dtype=float)
    orientations = _normit(np.asarray(orientations, dtype=float))

    angles = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)

    vertices = []
    faces = []
    offset = 0

    for pos, normal in zip(positions, orientations):
        u_vec, v_vec = _sensor_frame(normal)

        center = pos
        ring = np.array([
            pos + radius * (np.cos(a) * u_vec + np.sin(a) * v_vec)
            for a in angles
        ])

        sensor_vertices = np.vstack([center[None, :], ring])
        vertices.append(sensor_vertices.astype(np.float32))

        # fan triangulation from center
        for i in range(n_circle):
            i1_idx = offset + 1 + i
            i2_idx = offset + 1 + ((i + 1) % n_circle)
            faces.append([offset, i1_idx, i2_idx])

        offset += sensor_vertices.shape[0]

    return np.vstack(vertices), np.asarray(faces, dtype=np.uint32)


def _make_square_sensor_mesh(positions, orientations, directions=None, edge_length=0.01):
    """
    Build one batched triangular mesh containing one filled square per sensor.
    """
    positions = np.asarray(positions, dtype=float)
    orientations = _normit(np.asarray(orientations, dtype=float))
    directions = None if directions is None else np.asarray(directions, dtype=float)

    half = edge_length / 2.0
    vertices = []
    faces = []
    offset = 0

    for idx, (pos, normal) in enumerate(zip(positions, orientations)):
        dir_vec = None if directions is None else directions[idx]
        u_vec, v_vec = _sensor_frame(normal, direction=dir_vec)

        corners = np.array([
            pos + (+half * u_vec) + (+half * v_vec),
            pos + (-half * u_vec) + (+half * v_vec),
            pos + (-half * u_vec) + (-half * v_vec),
            pos + (+half * u_vec) + (-half * v_vec),
        ], dtype=np.float32)

        vertices.append(corners)

        # two triangles, no diagonals rendered separately
        faces.append([offset + 0, offset + 1, offset + 2])
        faces.append([offset + 0, offset + 2, offset + 3])

        offset += 4

    return np.vstack(vertices), np.asarray(faces, dtype=np.uint32)


def _compute_planar_channel_directions(tra, coil_positions, planar_mask):
    """
    Compute planar gradiometer in-plane directions from the transfer matrix.

    This follows the FieldTrip logic used for square Neuromag glyphs:
    for channels with exactly one positive and one negative coil weight,
    the in-plane direction is:

        coilpos(poscoil) - coilpos(negcoil)

    Parameters
    ----------
    tra : ndarray, shape (n_channels, n_coils)
        Coil-to-channel transfer matrix.
    coil_positions : ndarray, shape (n_coils, 3)
        Coil positions.
    planar_mask : ndarray, shape (n_channels,)
        Boolean mask selecting planar channels.

    Returns
    -------
    chan_dir : ndarray, shape (n_planar, 3)
        In-plane direction vectors for planar channels. Rows are NaN if a
        simple positive/negative pair cannot be identified.
    """
    tra = np.asarray(tra, dtype=float)
    coil_positions = np.asarray(coil_positions, dtype=float)
    planar_idx = np.where(planar_mask)[0]

    chan_dir = np.full((len(planar_idx), 3), np.nan, dtype=float)

    for out_idx, chan_idx in enumerate(planar_idx):
        pos_coil = np.where(tra[chan_idx, :] > 0)[0]
        neg_coil = np.where(tra[chan_idx, :] < 0)[0]

        if len(pos_coil) == 1 and len(neg_coil) == 1:
            direction = coil_positions[pos_coil[0], :] - coil_positions[neg_coil[0], :]
            chan_dir[out_idx, :] = _normit(direction[None, :])[0]

    return chan_dir


def _read_matlab_string(file, ref):
    arr = file[ref][()]
    arr = np.asarray(arr).squeeze()
    return ''.join(chr(int(x)) for x in arr if int(x) != 0)


def _load_sensor_geometry(data_file, inversion_idx=0):
    """
    Load both forward sensor geometry
    """
    check_inversion_exists(data_file, inversion_idx=inversion_idx)

    try:
        with h5py.File(data_file, "r") as file:
            inv = file[file["D"]["other"]["inv"][inversion_idx][0]]

            forward = inv["forward"]

            forward_sens = {
                "chanpos": forward["sensors"]["chanpos"][()].T,
                "chanori": forward["sensors"]["chanori"][()].T,
                "coilpos": forward["sensors"]["coilpos"][()].T,
                "coilori": forward["sensors"]["coilori"][()].T,
                "tra": forward["sensors"]["tra"][()].T,
                "chantype": [
                    _read_matlab_string(file, x) for x in forward["sensors"]["chantype"][()][0]
                ],
                "label": [
                    _read_matlab_string(file, x) for x in forward["sensors"]["label"][()][0]
                ]
            }

            mesh_verts = np.array(forward["mesh"]["vert"][()], dtype=float).T
            mesh_faces = forward["mesh"]["face"][()].T - 1

    except OSError:
        mat = loadmat(data_file, simplify_cells=True)
        inv = mat["D"]["other"]["inv"][inversion_idx]

        forward_sens = {
            "chanpos": inv["forward"]["sensors"]["chanpos"],
            "chanori": inv["forward"]["sensors"]["chanori"],
            "coilpos": inv["forward"]["sensors"]["coilpos"],
            "coilori": inv["forward"]["sensors"]["coilori"],
            "tra": inv["forward"]["sensors"]["tra"],
            "chantype": inv["forward"]["sensors"]["chantype"],
            "label": inv["forward"]["sensors"]["label"]
        }

        mesh_verts = inv["forward"]["mesh"]["vert"]
        mesh_faces = inv["forward"]["mesh"]["face"]

    return {
        "forward_sensors": forward_sens,
        'mesh_verts': mesh_verts,
        'mesh_faces': mesh_faces
    }


def verify_coregistration(data_file, surf_set, layer_name=None, stage='ds',
                          orientation='link_vector', fixed=True,
                          inversion_idx=0, fid_coords=None, display_3d_scan=False,
                          camera_view=None):
    """
    Visualize MEG-MRI coregistration by plotting the scalp, skull, cortical surfaces,
    and sensors, optionally along with fiducial landmarks.

    This function provides a 3D visualization of sensors as well as the subject's head
    geometry and fiducial points (nasion, left and right preauricular, if provided) to
    verify whether MEG-MRI coregistration was successful. It loads the pial, scalp,
    inner skull, and outer skull surfaces from the subject's directory and displays
    them using a `k3d` interactive plot.

    Parameters
    ----------
    data_file : str
        Path to the MEG dataset (SPM-compatible .mat file) that coregistration has already
        been run on.
    surf_set : LayerSurfaceSet
        Subject's surface set containing paths and metadata for cortical and head meshes.
    layer_name : str, optional
        Name of the mesh layer used for coregistration.
        Default is None (multilayer mesh)
    stage : {'combined', 'ds'}, optional
        Processing stage of mesh used for coregistration : `'combined'` for merged
        hemispheres, `'ds'` for downsampled single surfaces.
        Default is `'ds'`.
    orientation : str or None, optional
        Downsampling orientation identifier used when `stage='ds'`. Ignored otherwise.
    fixed : bool or None, optional
        Whether to load fixed or adaptive orientation surfaces when `stage='ds'`. Default is None.
    inversion_idx: int, optional
        Index of the forward model to show within the SPM data object (default: 0).
    fid_coords : dict, optional
        Dictionary of fiducial coordinates in MEG headspace, e.g.:
        ``{'nas': [x, y, z], 'lpa': [x, y, z], 'rpa': [x, y, z]}``.
        Default is None.
    camera_view : array_like, optional
        Predefined camera position and orientation. If None, automatic camera fitting is used.
        Default is None.

    Returns
    -------
    plot : k3d.plot.Plot
        Interactive 3D plot displaying cortical and head surfaces with fiducial markers.

    Raises
    ------
    FileNotFoundError
        If one or more head surface meshes are missing, indicating that coregistration
        or segmentation has not yet been completed.

    Notes
    -----
    - Intended as a visual diagnostic tool; does not modify or compute transformations.
    """

    geom = _load_sensor_geometry(data_file, inversion_idx=inversion_idx)
    chan_positions = geom["forward_sensors"]["chanpos"]
    chan_orientations = geom["forward_sensors"]["chanori"]
    coil_positions = geom["forward_sensors"]["coilpos"]
    tra = geom["forward_sensors"]["tra"]
    channel_types = np.array(geom["forward_sensors"]["chantype"])
    inv_mesh = surf_set.load(
        layer_name=layer_name,
        stage=stage,
        orientation=orientation,
        fixed=fixed
    )
    inv_vertices, inv_faces, *_ = inv_mesh.agg_data()
    inv_mesh_path = surf_set.get_mesh_path(
        layer_name=layer_name,
        stage=stage,
        orientation=orientation,
        fixed=fixed
    )

    fwd_vertices = geom["mesh_verts"]
    fwd_faces = geom["mesh_faces"]

    mismatch_msg = (
        f"Coregistration was not done using {inv_mesh_path}. "
        f"Either run coregistration with this mesh, or change parameters "
        f"to match the mesh coregistration was done with."
    )

    if inv_vertices.shape[0] != fwd_vertices.shape[0]:
        raise ValueError(mismatch_msg)
    if inv_faces.shape != fwd_faces.shape:
        raise ValueError(mismatch_msg)
    if not np.array_equal(inv_faces, fwd_faces):
        raise ValueError(mismatch_msg)

    scaled_affine, rmse = _estimate_rigid_transform(
        inv_vertices,
        fwd_vertices,
        allow_scaling=True
    )
    print(f'RMSE of rigid transformation between forward model mesh and '
          f'surf_set mesh is {rmse}.')
    print('If this is high, coregistration may '
          f'not have been done using {inv_mesh_path}.')
    print('Either run coregistration with this mesh, or change parameters '
          'to match the mesh coregistration was done with.')

    pial_mesh = surf_set.load(layer_name='pial', stage='combined')
    pial_vertices, pial_faces, *_ = pial_mesh.agg_data()
    pial_vertices = _apply_affine(pial_vertices, scaled_affine)

    scalp_mesh = surf_set.load_head_mesh('scalp')
    scalp_vertices, scalp_faces, *_ = scalp_mesh.agg_data()
    scalp_vertices = _apply_affine(scalp_vertices, scaled_affine)

    iskull_mesh = surf_set.load_head_mesh('iskull')
    iskull_faces, iskull_vertices, *_ = iskull_mesh.agg_data()
    iskull_vertices = _apply_affine(iskull_vertices, scaled_affine)

    oskull_mesh = surf_set.load_head_mesh('oskull')
    oskull_faces, oskull_vertices, *_ = oskull_mesh.agg_data()
    oskull_vertices = _apply_affine(oskull_vertices, scaled_affine)

    cam_autofit = camera_view is None
    plot = k3d.plot(grid_visible=False, camera_auto_fit=cam_autofit)

    plot += k3d.mesh(
        pial_vertices, pial_faces,
        side="double",
        color=rgbtoint([166, 166, 166]),
        opacity=1.0,
        name="cortex"
    )

    plot += k3d.mesh(
        scalp_vertices, scalp_faces,
        side="double",
        color=rgbtoint([186, 160, 115]),
        opacity=0.35,
        name="scalp"
    )

    plot += k3d.mesh(
        iskull_vertices, iskull_faces,
        side="double",
        color=rgbtoint([255, 255, 255]),
        opacity=0.25,
        name="inner skull"
    )

    plot += k3d.mesh(
        oskull_vertices, oskull_faces,
        side="double",
        color=rgbtoint([255, 255, 255]),
        opacity=0.15,
        name="outer skull"
    )

    if fid_coords is not None:
        if "nas" in fid_coords:
            nas_vertex = _apply_affine(
                np.atleast_2d(fid_coords["nas"]).astype(np.float32),
                scaled_affine
            )
            plot += k3d.points(
                nas_vertex,
                point_size=0.004,
                color=rgbtoint([0, 0, 255]),
                name="nas"
            )
        if "lpa" in fid_coords:
            lpa_vertex = _apply_affine(
                np.atleast_2d(fid_coords["lpa"]).astype(np.float32),
                scaled_affine
            )
            plot += k3d.points(
                lpa_vertex,
                point_size=0.004,
                color=rgbtoint([255, 0, 0]),
                name="lpa"
            )
        if "rpa" in fid_coords:
            rpa_vertex = _apply_affine(
                np.atleast_2d(fid_coords["rpa"]).astype(np.float32),
                scaled_affine
            )
            plot += k3d.points(
                rpa_vertex,
                point_size=0.004,
                color=rgbtoint([0, 255, 0]),
                name="rpa"
            )

    sensor_color = [0, 255, 0]
    sensor_color_int = rgbtoint(sensor_color)

    mag_mask = channel_types == "megmag"
    if np.any(mag_mask):
        plot += k3d.points(
            chan_positions[mag_mask, :].astype(np.float32),
            point_size=0.003,
            color=sensor_color_int,
            name="magnetometers"
        )

    planar_mask = channel_types == "megplanar"
    if np.any(planar_mask):
        planar_dirs = _compute_planar_channel_directions(
            tra=tra,
            coil_positions=coil_positions,
            planar_mask=planar_mask
        )

        square_vertices, square_faces = _make_square_sensor_mesh(
            chan_positions[planar_mask, :],
            chan_orientations[planar_mask, :],
            directions=planar_dirs,
            edge_length=0.03
        )

        plot += k3d.mesh(
            square_vertices,
            square_faces,
            color=sensor_color_int,
            opacity=0.2,
            side="double",
            name="planar gradiometers"
        )

    meg_mask = (channel_types == "meg") | (channel_types == 'meggrad') | (channel_types == 'opm')
    if np.any(meg_mask):
        circle_vertices, circle_faces = _make_circle_sensor_mesh(
            chan_positions[meg_mask, :],
            chan_orientations[meg_mask, :],
            radius=0.01,
            n_circle=24
        )

        plot += k3d.mesh(
            circle_vertices,
            circle_faces,
            color=sensor_color_int,
            opacity=0.3,
            side="double",
            name="meg"
        )

    if display_3d_scan:
        scan_fname = os.path.join(surf_set.surf_dir, 'face_3d.gii')
        if not os.path.exists(scan_fname):
            raise ValueError(f'3D scan of face not found at {scan_fname}. Did you use '
                             f'lameg.util.coregister_3d_scan_mri?')
        face_surf = nib.load(scan_fname)
        face_verts, face_faces, *_ = face_surf.agg_data()
        face_verts = _apply_affine(face_verts, scaled_affine)
        plot += k3d.mesh(
            face_verts, face_faces,
            side="double",
            color=rgbtoint([0, 0, 255]),
            opacity=1.0,
            name="3dscan"
        )

    if camera_view is not None:
        plot.camera = camera_view

    plot.display()
    return plot


def plot_csd(csd, times, axis, colorbar=True, cmap="RdBu_r", vmin_vmax=None, n_layers=11,
             layer_boundaries=None):
    """
    Visualize a laminar Current Source Density (CSD) profile over time.

    This function plots a 2D representation of CSD data (layers × time) using a diverging
    colormap to indicate current sources and sinks. It supports automatic or user-defined color
    scaling, customizable colormaps, and optional display of laminar boundaries.

    Parameters
    ----------
    csd : np.ndarray, shape (n_layers, n_times)
        CSD matrix to plot, where each row corresponds to a cortical layer and each column to a
        time point.
    times : np.ndarray, shape (n_times,)
        Time vector corresponding to the CSD data columns (in ms).
    axis : matplotlib.axes.Axes
        Matplotlib axes on which to draw the CSD plot.
    colorbar : bool, optional
        If True, adds a colorbar indicating CSD amplitude (source/sink polarity). Default is True.
    cmap : str, optional
        Colormap name (e.g., `'RdBu_r'`, `'viridis'`). Default is `'RdBu_r'`.
    vmin_vmax : tuple of float or {'norm', None}, optional
        Color normalization limits:
        - (vmin, vmax): explicit color scale limits,
        - `'norm'`: symmetric normalization based on maximum absolute CSD value,
        - None: uses full data range. Default is None.
    n_layers : int, optional
        Number of cortical layers represented in the CSD matrix. Default is 11.
    layer_boundaries : array_like, optional
        Optional array of y-axis positions marking laminar boundaries. Default is None.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The plotted image object.
    cb : matplotlib.colorbar.Colorbar or None
        The colorbar object, if `colorbar=True`, otherwise None.

    Notes
    -----
    - Assumes CSD is in units of current per unit area (e.g., uA/mm²).
    - Negative values (blue) typically represent current sinks; positive values (red) represent
      sources.
    - If `layer_boundaries` is provided, horizontal lines are drawn at those positions.
    - Suitable for visualizing laminar profiles from simulations, hpMEG reconstructions, or
      LFP-derived CSD estimates.
    """

    max_smooth = np.max(np.abs(csd))
    if vmin_vmax is None:
        divnorm = colors.TwoSlopeNorm(vmin=-max_smooth, vcenter=0, vmax=max_smooth)
    elif vmin_vmax == "norm":
        divnorm = colors.Normalize()
    else:
        divnorm = colors.TwoSlopeNorm(vmin=vmin_vmax[0], vcenter=0, vmax=vmin_vmax[1])
    extent = [times[0], times[-1], 0, 1]
    csd_imshow = axis.imshow(
        csd, norm=divnorm, origin="lower",
        aspect="auto", extent=extent,
        cmap=cmap, interpolation="none"
    )
    if layer_boundaries is not None:
        for boundary in layer_boundaries:
            axis.axhline(y=boundary, color='k', linestyle='--')

    axis.set_ylim(1, 0)
    axis.set_yticks(np.linspace(0, 1, n_layers))
    axis.set_yticklabels(np.arange(1, n_layers+1))
    if colorbar:
        clbr=plt.colorbar(csd_imshow, ax=axis)
        clbr.set_label('CSD')
    plt.tight_layout()
    return csd_imshow
