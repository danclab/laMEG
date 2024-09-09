"""
Surface processing
==================
"""

# %%
# Create layer surfaces
# ---------------------
# Uses FreeSurfer mris-inflate. Based on the number of surfaces you specify. Each will be approximately equidistant. Vertex correspondence is maintained, i.e each layer will have vertices along the vector connecting a white matter surface vertex with the corresponding pial surface vertex. Each surface will then be converted to gifti format and the vertex coordinates will adjusted by the RAS at the center of the volume to put them in native space.

import os
import numpy as np
import nibabel as nib
import k3d
import matplotlib.pyplot as plt
from lameg.viz import show_surface, rgbtoint

# Get name of each mesh that makes up the layers of the multilayer mesh
n_layers = 11
layers = np.linspace(1, 0, n_layers)

surf_path = '../test_data/sub-104/surf'
layer_fnames = []
for l, layer in enumerate(layers):
    if layer == 1:
        layer_fnames.append(os.path.join(surf_path, f'lh.pial.gii'))
    elif layer > 0 and layer < 1:
        layer_name = '{:.3f}'.format(layer)
        layer_fnames.append(os.path.join(surf_path, f'lh.{layer_name}.gii'))
    elif layer == 0:
        layer_fnames.append(os.path.join(surf_path, f'lh.white.gii'))

cam_view=[-143, 14, 31.5,
          -32, 22.5, 38.5,
          0, 0, 1]
col_r = plt.cm.cool(np.linspace(0,1, num=n_layers))

for l_idx,layer_fname in enumerate(layer_fnames):
    mesh = nib.load(layer_fname)
    plot = show_surface(mesh, camera_view=cam_view, color=col_r[l_idx,:3]*255, height=256)

# %%
#.. image:: ../_static/tutorial_01_pial.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.9.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.8.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.7.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.6.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.5.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.4.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.3.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.2.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_0.1.png
#   :width: 800
#   :alt:
#.. image:: ../_static/tutorial_01_white.png
#   :width: 800
#   :alt:

# %%
# Remove deep vertices
# --------------------
# Freesurfer operates on hemispheres independently, resulting in deep vertices and mesh faces cutting through subcortical structures. These are removed because they are not part of the cortical source space. Any vertex not labelled or labelled as unknown in the Desikan-Killiany atlas are removed. This is applied to each hemisphere of each layer mesh.

# %%
# **Before**
surf_fname = os.path.join(surf_path, f'lh.pial.gii')

cam_view=[110, 18, 36,
          -32, 22, 42,
          0, 0, 1]
col_r = plt.cm.cool(np.linspace(0,1, num=n_layers))

mesh = nib.load(surf_fname)
plot = show_surface(mesh, camera_view=cam_view, color=col_r[0,:3]*255)

# %%
#.. image:: ../_static/tutorial_01_rm_deep_before.png
#   :width: 800
#   :alt:

# %%
# **After**
surf_fname = os.path.join(surf_path, f'lh.pial.nodeep.gii')

mesh = nib.load(surf_fname)
plot = show_surface(mesh, camera_view=cam_view, color=col_r[0,:3]*255)

# %%
#.. image:: ../_static/tutorial_01_rm_deep_after.png
#   :width: 800
#   :alt:

# %%
# Combine hemispheres
# -------------------
# The left and right hemisphere meshes are combined by concatenation of their vertices and faces (left then right). No new faces are created. This is done for each layer.
surf_fname = os.path.join(surf_path, f'pial.gii')

cam_view=[13, 64, 217,
          21, 21, 33,
          0, 1, 0]
col_r = plt.cm.cool(np.linspace(0,1, num=n_layers))

mesh = nib.load(surf_fname)
plot = show_surface(mesh, camera_view=cam_view, color=col_r[0,:3]*255)

# %%
#.. image:: ../_static/tutorial_01_combine_hemi.png
#   :width: 800
#   :alt:

# %%
# Downsample
# ----------
# The surfaces are much too dense to use in source reconstruction. They must be downsampled, but we must maintain vertex correspondence between them (see [Bonaiuto et al. 2020, Estimates of cortical column orientation improve MEG source inversion](https://doi.org/10.1016/j.neuroimage.2020.116862)). The pial surface is therefore downsampled by a factor of 10 using the vtkDecimatePro algorithm, which only removes vertices rather than creating new ones. The removed vertices are also removed from each other layer mesh, and the face structure from the pial mesh is copied to them (though the faces are not used in the source reconstruction if link vector orientations are used).
surf_fname = os.path.join(surf_path, f'pial.ds.link_vector.fixed.gii')

cam_view=[13, 64, 217,
          21, 21, 33,
          0, 1, 0]
col_r = plt.cm.cool(np.linspace(0,1, num=n_layers))

mesh = nib.load(surf_fname)
plot = show_surface(mesh, camera_view=cam_view, color=col_r[0,:3]*255)

# %%
#.. image:: ../_static/tutorial_01_downsample.png
#   :width: 800
#   :alt:

# %%
# Compute link vectors
# --------------------
# The dipole orientation at each source location is computed. Downsampled surface normals, original surface normals, cortical patch statistics, or link vectors can be specified (see [Bonaiuto et al. 2020, Estimates of cortical column orientation improve MEG source inversion](https://doi.org/10.1016/j.neuroimage.2020.116862)). The first three compute vertex normals based on the orientation of the surrounding faces. They can be either computed separately for each layer, or the pial surface orientations can be used for all layers. The link vectors method computes vectors connecting corresponding pial and white matter vertices (pointing toward the white matter), which are therefore the same across layer. The orientation vectors are stored in the normals attribute of the gifti object.
plot = k3d.plot(
    grid_visible=False, menu_visibility=False, camera_auto_fit=False
)
pial_fname = os.path.join(surf_path, f'pial.ds.link_vector.fixed.gii')
pial_surf = nib.load(pial_fname)
pial_vertices, pial_faces, _ = pial_surf.agg_data()

white_fname = os.path.join(surf_path, f'white.ds.link_vector.fixed.gii')
white_surf = nib.load(white_fname)
white_vertices, white_faces, _ = white_surf.agg_data()

cam_view=[70, 40.5, 76.5,
          22.5, 26, 73.5,
          0, 0, 1]
col_r = plt.cm.cool(np.linspace(0,1, num=n_layers))

pial_mesh = k3d.mesh(pial_vertices, pial_faces, side="double", color=rgbtoint(col_r[0,:3]*255), opacity=0.5)
plot += pial_mesh

white_mesh = k3d.mesh(white_vertices, white_faces, side="double", color=rgbtoint(col_r[-1,:3]*255), opacity=1)
plot += white_mesh

dipole_vectors = k3d.vectors(
    white_vertices,
    vectors=pial_vertices - white_vertices,
    head_size=5,
    line_width=0.1
)
plot += dipole_vectors

plot.camera=cam_view

plot.display()

# %%
#.. image:: ../_static/tutorial_01_link_vectors.png
#   :width: 800
#   :alt:

# %%
# Combine layers
# --------------
# The layer meshes are then combined into a single mesh by concatenating their vertices and faces (from pial to white matter). No new faces are created (i.e. there are no edges connecting vertices across layers)
layer_fnames = []
for l, layer in enumerate(layers):
    if layer == 1:
        layer_fnames.append(os.path.join(surf_path, f'pial.ds.link_vector.fixed.gii'))
    elif layer > 0 and layer < 1:
        layer_name = '{:.3f}'.format(layer)
        layer_fnames.append(os.path.join(surf_path, f'{layer_name}.ds.link_vector.fixed.gii'))
    elif layer == 0:
        layer_fnames.append(os.path.join(surf_path, f'white.ds.link_vector.fixed.gii'))

col_r = plt.cm.cool(np.linspace(0,1, num=n_layers+1))

plot = k3d.plot(
    grid_visible=False, menu_visibility=False, camera_auto_fit=False
)

for l_idx,layer_fname in enumerate(layer_fnames):
    surface = nib.load(layer_fname)
    vertices, faces, _ = surface.agg_data()
    
    mesh = k3d.mesh(vertices, faces, side="double", color=rgbtoint(col_r[l_idx,:3]*255), opacity=l_idx/(n_layers-1))
    plot += mesh
    
plot.camera=cam_view

plot.display()

# %%
#.. image:: ../_static/tutorial_01_combine_layers.png
#   :width: 800
#   :alt:

# %%
# Putting it all together
# -----------------------
# All of these steps can be run using the function:
#
# > postprocess_freesurfer_surfaces

from lameg.surf import postprocess_freesurfer_surfaces

# Create a 2-layer surface (only pial and white)
postprocess_freesurfer_surfaces(
    'sub-104',                                
    '../test_data/sub-104/surf', 
    'multilayer.2.ds.link_vector.fixed.gii',                                
    n_surfaces=2,                                 
    ds_factor=0.1,                                 
    orientation='link_vector',                                 
    remove_deep=True
)

# Create an 11-layer surface
postprocess_freesurfer_surfaces(
    'sub-104',                                
    '../test_data/sub-104/surf', 
    'multilayer.11.ds.link_vector.fixed.gii',
    n_surfaces=11, 
    ds_factor=0.1, 
    orientation='link_vector', 
    remove_deep=True
)

# Create a 15-layer surface
postprocess_freesurfer_surfaces(
    'sub-104',                                
    '../test_data/sub-104/surf', 
    'multilayer.15.ds.link_vector.fixed.gii',
    n_surfaces=15, 
    ds_factor=0.1, 
    orientation='link_vector', 
    remove_deep=True
)

# %%



