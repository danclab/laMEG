"""
Laminar CSD analysis
====================
This tutorial demonstrates how to perform laminar inference using a CSD analysis of event-related source signals. A temporal Gaussian function is simulated at a particular cortical location in various layers. Source reconstruction is performed on the whole time window using the Empirical Bayesian Beamformer on the simulated sensor data using a forward model based on the multilayer mesh, thus providing an estimate of source activity on each layer. A laminar CSD is run on the laminar signals at the location with the peak variance.
"""
# %%
# Setting up the simulations
# --------------------------
# 
# Simulations are based on an existing dataset, which is used to define the sampling rate, number of trials, duration of each trial, and the channel layout.

import os
import shutil
import numpy as np
import k3d
import matplotlib.pyplot as plt
import tempfile

from lameg.invert import invert_ebb, coregister, load_source_time_series
from lameg.laminar import compute_csd
from lameg.simulate import run_dipole_simulation
from lameg.surf import LayerSurfaceSet
from lameg.util import get_fiducial_coords
from lameg.viz import show_surface, color_map, plot_csd, rgbtoint
import spm_standalone

# Subject information for data to base the simulations on
subj_id = 'sub-104'
ses_id = 'ses-01'

# Fiducial coil coordinates
fid_coords = get_fiducial_coords(subj_id, '../test_data/participants.tsv')

# Data file to base simulations on
data_file = os.path.join(
    '../test_data',
    subj_id,
    'meg',
    ses_id,
    f'spm/pspm-converted_autoreject-{subj_id}-{ses_id}-001-btn_trial-epo.mat'
)

spm = spm_standalone.initialize()

# %% [markdown]
# The simulations and source reconstructions will be based on a forward model using the multilayer mesh

surf_set = LayerSurfaceSet(subj_id, 15)

verts_per_surf = surf_set.get_vertices_per_layer()

# %% [markdown]
# We're going to copy the data file to a temporary directory and direct all output there.

# Extract base name and path of data file
data_path, data_file_name = os.path.split(data_file)
data_base = os.path.splitext(data_file_name)[0]

# Where to put simulated data
tmp_dir = tempfile.mkdtemp()

# Copy data files to tmp directory
shutil.copy(
    os.path.join(data_path, f'{data_base}.mat'), 
    os.path.join(tmp_dir, f'{data_base}.mat')
)
shutil.copy(
    os.path.join(data_path, f'{data_base}.dat'), 
    os.path.join(tmp_dir, f'{data_base}.dat')
)

# Construct base file name for simulations
base_fname = os.path.join(tmp_dir, f'{data_base}.mat')

# %% [markdown]
# Invert the subject's data using the multilayer mesh. This step only has to be done once - this is just to compute the forward model that will be used in the simulations

# Patch size to use for inversion (in this case it matches the simulated patch size)
patch_size = 5
# Number of temporal modes to use for EBB inversion
n_temp_modes = 4

# Coregister data to multilayer mesh
coregister(
    fid_coords,
    base_fname,
    surf_set,
    spm_instance=spm
)

# Run inversion
[_,_] = invert_ebb(
    base_fname,
    surf_set,
    patch_size=patch_size, 
    n_temp_modes=n_temp_modes,
    spm_instance=spm
)

# %% [markdown]
# Simulating a signal on a superficial surface
# --------------------------------------------
# We're going to simulate 200ms of a Gaussian with a dipole moment of 5nAm and a width of 25ms

# Strength of simulated activity (nAm)
dipole_moment = 5
# Temporal width of the simulated Gaussian
signal_width=.025 # 25ms
# Sampling rate (must match the data file)
s_rate = 600

# Generate 200ms of a Gaussian at a sampling rate of 600Hz (to match the data file)
time=np.linspace(0,.2,121)
zero_time=time[int((len(time)-1)/2+1)]
sim_signal=np.exp(-((time-zero_time)**2)/(2*signal_width**2)).reshape(1,-1)
plt.plot(time,dipole_moment*sim_signal[0,:])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (nAm)')

# %%
#.. image:: ../_static/tutorial_06_sim_signal.png
#   :width: 800
#   :alt:

# %% [markdown]
# We need to pick a location (mesh vertex) to simulate at

# Vertex to simulate activity at
sim_vertex=10561

cam_view = [40, -240, 25,
            60, 37, 17,
            0, 0, 1]
plot = show_surface(
    surf_set,
    marker_vertices=sim_vertex,
    marker_size=5,
    camera_view=cam_view
)

# %%
#.. image:: ../_static/tutorial_06_sim_location.png
#   :width: 800
#   :alt:

# %% [markdown]
# We'll simulate a 5mm patch of activity with -5 dB SNR at the sensor level. The desired level of SNR is achieved by adding white noise to the projected sensor signals

# Simulate at a vertex on the pial surface
pial_vertex = surf_set.get_multilayer_vertex('pial', sim_vertex)
multilayer_mesh = surf_set.load(stage='ds', orientation='link_vector', fixed=True)
sim_unit_norm = multilayer_mesh.darrays[2].data[pial_vertex,:]
prefix = f'sim_{sim_vertex}_pial_'

# Size of simulated patch of activity (mm)
sim_patch_size = 5
# SNR of simulated data (dB)
SNR = -5

# Generate simulated data
pial_sim_fname = run_dipole_simulation(
    base_fname, 
    prefix, 
    pial_vertex, 
    sim_signal, 
    sim_unit_norm, 
    dipole_moment, 
    sim_patch_size, 
    SNR,
    spm_instance=spm
) 

# %% [markdown]
# Inversion
# ---------
# Now we'll run a source reconstruction using the multilayer mesh, select the vertex to examine, extract the source signals at each layer in that location, and compute a laminar CSD

[_,_,MU] = invert_ebb(
    pial_sim_fname,
    surf_set,
    patch_size=patch_size,
    n_temp_modes=n_temp_modes,
    return_mu_matrix=True,
    spm_instance=spm
)

pial_layer_vertices = np.arange(verts_per_surf)
pial_layer_ts, time, _ = load_source_time_series(
    pial_sim_fname,
    mu_matrix=MU,
    vertices=pial_layer_vertices
)

# Layer peak
m_layer_max = np.max(np.mean(pial_layer_ts,axis=-1),-1)
peak = np.argmax(m_layer_max)

print(f'Simulated vertex={sim_vertex}, Prior vertex={peak}')

# %% [markdown]
# We can see that the peak is very close to the location we simulated at

# Plot colors and camera view
max_abs = np.max(np.abs(m_layer_max))
c_range = [-max_abs, max_abs]

# Plot peak
colors,_ = color_map(
    m_layer_max,
    "RdYlBu_r",
    c_range[0],
    c_range[1]
)
thresh_colors=np.ones((colors.shape[0],4))*255
thresh_colors[:,:3]=colors
thresh_colors[m_layer_max<np.percentile(m_layer_max,99.9),3]=0

plot = show_surface(
    surf_set,
    vertex_colors=thresh_colors,
    info=True,
    camera_view=cam_view,
    marker_vertices=peak,
    marker_size=5,
    marker_color=[0,0,255]
)

# %%
#.. image:: ../_static/tutorial_06_localizer.png
#   :width: 800
#   :alt:


# %% [markdown]
# We need the indices of the vertex at each layer for this location, and the distances between them

layer_verts = surf_set.get_layer_vertices(peak)
layer_dists = surf_set.get_interlayer_distance(peak)
print(layer_dists)

# %% [markdown]
# Now we can compute and plot the laminar CSD

# Get source time series for each layer
layer_ts, time, _ = load_source_time_series(pial_sim_fname, vertices=layer_verts)

# Average over trials and compute CSD and smoothed CSD
mean_layer_ts = np.mean(layer_ts, axis=-1)
[csd, smooth_csd] = compute_csd(mean_layer_ts, np.sum(layer_dists), s_rate, smoothing='cubic')

col_r = plt.cm.cool(np.linspace(0, 1, num=surf_set.n_layers))
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
for l in range(surf_set.n_layers):
    plt.plot(time, mean_layer_ts[l, :], label=f'{l}', color=col_r[l, :])
plt.legend(loc='upper left')
plt.xlabel('Time (ms)')
plt.ylabel('Source (nAm)')

plt.subplot(1, 3, 2)
for l in range(surf_set.n_layers):
    plt.plot(time, csd[l, :], label=f'{l}', color=col_r[l, :])
plt.legend(loc='upper left')
plt.xlabel('Time (ms)')
plt.ylabel('CSD')

ax = plt.subplot(1, 3, 3)
plot_csd(smooth_csd, time, ax, n_layers=surf_set.n_layers)
plt.xlabel('Time (ms)')
plt.ylabel('Layer')
plt.tight_layout()

# %%
#.. image:: ../_static/tutorial_06_pial_sim_results.png
#   :width: 800
#   :alt:


# %% [markdown]
# White matter surface simulation with laminar CSD
# ------------------------------------------------
# Let's simulate the same pattern of activity, in the same location, but on the white matter surface.

# Simulate at the corresponding vertex on the white matter surface
white_vertex = surf_set.get_multilayer_vertex('white', sim_vertex)
prefix = f'sim_{sim_vertex}_white_'

# Generate simulated data
white_sim_fname = run_dipole_simulation(
    base_fname,
    prefix,
    white_vertex,
    sim_signal,
    sim_unit_norm,
    dipole_moment,
    sim_patch_size,
    SNR,
    spm_instance=spm
)

[_, _, MU] = invert_ebb(
    white_sim_fname,
    surf_set,
    patch_size=patch_size,
    n_temp_modes=n_temp_modes,
    return_mu_matrix=True,
    spm_instance=spm
)

layer_verts = surf_set.get_layer_vertices(peak)
layer_dists = surf_set.get_interlayer_distance(peak)
print(layer_dists)

# Get source time series for each layer
layer_ts, time, _ = load_source_time_series(white_sim_fname, mu_matrix=MU, vertices=layer_verts)

# Average over trials and compute CSD and smoothed CSD
mean_layer_ts = np.mean(layer_ts, axis=-1)
[csd, smooth_csd] = compute_csd(mean_layer_ts, np.sum(layer_dists), s_rate, smoothing='cubic')

col_r = plt.cm.cool(np.linspace(0, 1, num=surf_set.n_layers))
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
for l in range(surf_set.n_layers):
    plt.plot(time, mean_layer_ts[l, :], label=f'{l}', color=col_r[l, :])
plt.legend(loc='upper left')
plt.xlabel('Time (ms)')
plt.ylabel('Source (nAm)')

plt.subplot(1, 3, 2)
for l in range(surf_set.n_layers):
    plt.plot(time, csd[l, :], label=f'{l}', color=col_r[l, :])
plt.legend(loc='upper left')
plt.xlabel('Time (ms)')
plt.ylabel('CSD')

ax = plt.subplot(1, 3, 3)
plot_csd(smooth_csd, time, ax, n_layers=surf_set.n_layers)
plt.xlabel('Time (ms)')
plt.ylabel('Layer')
plt.tight_layout()

# %%
#.. image:: ../_static/tutorial_06_white_sim_results.png
#   :width: 800
#   :alt:


# %% [markdown]
# Simulation in each layer
# ------------------------
# Let's now simulate on each layer, and for each simulation, run the laminar CSD. We'll turn off SPM visualization here.

# Now simulate at the corresponding vertex on each layer, and for each simulation compute CSD
layer_csds = []
for l in range(surf_set.n_layers):
    print(f'Simulating in layer {l}')
    prefix = f'sim_{sim_vertex}_{l}_'
    l_vertex = surf_set.get_multilayer_vertex(l, sim_vertex)

    l_sim_fname = run_dipole_simulation(
        base_fname,
        prefix,
        l_vertex,
        sim_signal,
        sim_unit_norm,
        dipole_moment,
        sim_patch_size,
        SNR,
        spm_instance=spm
    )

    [_, _, MU] = invert_ebb(
        l_sim_fname,
        surf_set,
        patch_size=patch_size,
        n_temp_modes=n_temp_modes,
        viz=False,
        return_mu_matrix=True,
        spm_instance=spm
    )

    layer_verts = surf_set.get_layer_vertices(peak)
    layer_dists = surf_set.get_interlayer_distance(peak)
    print(layer_dists)

    # Get source time series for each layer
    layer_ts, time, _ = load_source_time_series(l_sim_fname, mu_matrix=MU, vertices=layer_verts)

    mean_layer_ts = np.mean(layer_ts, axis=-1)
    [csd, smooth_csd] = compute_csd(mean_layer_ts, np.sum(layer_dists), 600, smoothing='cubic')

    col_r = plt.cm.cool(np.linspace(0, 1, num=surf_set.n_layers))
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    for l_idx in range(surf_set.n_layers):
        plt.plot(time, mean_layer_ts[l_idx, :], label=f'{l_idx}', color=col_r[l_idx, :])
    plt.legend(loc='upper left')
    plt.xlabel('Time (ms)')
    plt.ylabel('Source (nAm)')

    plt.subplot(1, 3, 2)
    for l_idx in range(surf_set.n_layers):
        plt.plot(time, csd[l_idx, :], label=f'{l_idx}', color=col_r[l_idx, :])
    plt.legend(loc='upper left')
    plt.xlabel('Time (ms)')
    plt.ylabel('CSD')

    ax = plt.subplot(1, 3, 3)
    plot_csd(smooth_csd, time, ax, n_layers=surf_set.n_layers)
    plt.xlabel('Time (ms)')
    plt.ylabel('Layer')
    plt.tight_layout()

    layer_csds.append(smooth_csd)

# %%
#.. image:: ../_static/tutorial_06_pial_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_1_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_2_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_3_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_4_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_5_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_6_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_7_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_8_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_9_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_10_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_11_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_12_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_13_sim_results.png
#   :width: 800
#   :alt:

# %%
#.. image:: ../_static/tutorial_06_white_sim_results.png
#   :width: 800
#   :alt:

# %% [markdown]
# For each simulation, we can plot a slice of the CSD through layers around a central time window. The layer model where the with the maximal CSD signal magnitude should correspond to the layer that the activity was simulated in.

scale_factor=500/surf_set.n_layers
csd_patterns = []
peaks = []
for layer_csd in layer_csds:
    t_idx = np.where((time>=-0.05) & (time<=0.05))[0]
    csd_pattern = np.mean(layer_csd[:,t_idx],axis=1)
    peak = np.argmax(np.abs(csd_pattern))
    peaks.append(np.argmax(np.abs(csd_pattern))/scale_factor)
    csd_patterns.append(csd_pattern)

col_r = plt.cm.cool(np.linspace(0,1, num=surf_set.n_layers))
plt.figure(figsize=(10,4))

# For each simulation, plot the CSD slice
plt.subplot(1,2,1)
for l in range(surf_set.n_layers):
    plt.plot(np.arange(len(csd_patterns[l]))/scale_factor,csd_patterns[l], label=f'{l}', color=col_r[l,:])
plt.legend()
plt.xlabel('Eval layer')
plt.ylabel('CSD')

# For each simulation, find which layer model had the peak CSD
plt.subplot(1,2,2)
plt.plot(np.arange(surf_set.n_layers),peaks)
plt.xlim([-0.5,surf_set.n_layers-.5])
plt.ylim([-0.5,surf_set.n_layers-.5])
plt.plot([0,surf_set.n_layers-1],[0,surf_set.n_layers-1],'k--')
plt.xlabel('Sim layer')
plt.ylabel('Peak CSD')
plt.tight_layout()

# %%
#.. image:: ../_static/tutorial_06_results.png
#   :width: 800
#   :alt:


# %%
csd_patterns=np.array(csd_patterns)
# Transpose for visualization
im=plt.imshow(csd_patterns.T,aspect='auto', cmap='Spectral_r',extent=[0, surf_set.n_layers, surf_set.n_layers, 0])

# Find the indices of the max value in each column
max_indices = np.argmax(csd_patterns, axis=1)

# Plot an 'X' at the center of the square for each column's maximum
for idx, max_idx in enumerate(max_indices):
    plt.text(idx, max_idx/scale_factor, 'X', fontsize=12, ha='center', va='center', color='black', weight='bold')

plt.xlabel('Simulated layer', fontsize=14)
plt.ylabel('Evaluated layer', fontsize=14)
cb=plt.colorbar(im)
cb.set_label('CSD', fontsize=14)

# %%
#.. image:: ../_static/tutorial_06_results_matrix.png
#   :width: 800
#   :alt:

# %% [markdown]
# Beta burst CSD
# --------------
# That was a simulation of a source in a single layer. Let's try a beta burst simulation, with simultaneous sources in deep and superficial layers (see [Bonaiuto et al., 2021, Laminar dynamics of high amplitude beta bursts in human motor cortex](https://doi.org/10.1016/j.neuroimage.2021.118479))

# Strength of each simulated source (nAm)
dipole_moment = [8, 6]
# Temporal width of the simulated superficial signal
superficial_width = .01 # 10ms
# Temporal width of the simulated deep signal
deep_width = .025 # 25ms

# Sampling rate (must match the data file)
s_rate = 600

# Generate 200ms of a Gaussian at a sampling rate of 600Hz (to match the data file)
time = np.linspace(0,.2,121)
zero_time = time[int((len(time)-1)/2+1)]
superficial_signal = np.exp(-((time-zero_time)**2)/(2*superficial_width**2))
deep_signal = np.exp(-((time-zero_time)**2)/(2*deep_width**2))

plt.plot(time,superficial_signal*dipole_moment[0], 'b', label='superficial')
plt.plot(time,deep_signal*dipole_moment[1], 'r', label='deep')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (nAm)')

# %%
#.. image:: ../_static/tutorial_06_burst_sim_signal.png
#   :width: 800
#   :alt:

# %% [markdown]
# We need to pick a location (mesh vertex) to simulate at. The superficial signal will be simulated as a dipole at the corresponding vertex on the pial surface, and the deep signal on the white matter surface. The dipole orientations will be in opposite directions (with the superficial one pointing toward the deep one, and vice versa). This will yield a cumulative dipole moment with a beta burst-like shape

# Location to simulate activity at
sim_vertex=24581
# Corresponding pial and white matter vertices
pial_vertex = sim_vertex
white_vertex = (surf_set.n_layers-1)*int(verts_per_surf)+sim_vertex
pial_coord = multilayer_mesh.darrays[0].data[pial_vertex,:]
white_coord = multilayer_mesh.darrays[0].data[white_vertex,:]

# Orientation of the simulated superficial dipole
pial_ori=multilayer_mesh.darrays[2].data[pial_vertex,:]
# Orientation of the simulated deep dipole
white_ori=-1*multilayer_mesh.darrays[2].data[white_vertex,:]

col_r = plt.cm.cool(np.linspace(0,1, num=surf_set.n_layers))

pial_mesh = surf_set.load(layer_name='pial', stage='combined')
white_mesh = surf_set.load(layer_name='white', stage='combined')

cam_view = [85.5, -10.5, 32,
            0.5, 17, 43,
            0, 0, 1]

plot = k3d.plot(
    grid_visible=False, menu_visibility=False, camera_auto_fit=False
)

pial_vertices, pial_faces, _ = pial_mesh.agg_data()
pial_k3d_mesh = k3d.mesh(pial_vertices, pial_faces, side="double", color=rgbtoint(col_r[0,:3]*255), opacity=0.5)
plot += pial_k3d_mesh

white_vertices, white_faces, _ = white_mesh.agg_data()
white_k3d_mesh = k3d.mesh(white_vertices, white_faces, side="double", color=rgbtoint(col_r[-1,:3]*255), opacity=1)
plot += white_k3d_mesh

pts = k3d.points(
    np.vstack([pial_coord, white_coord]),
    point_size=1,
    color=rgbtoint([0,0,0])
)
plot += pts

dipole_vectors = k3d.vectors(
    np.vstack([pial_coord, white_coord]),
    vectors=np.vstack([pial_ori, white_ori])*2.3,
    head_size=5,
    line_width=0.1,
    colors=[rgbtoint([0,0,255]), rgbtoint([0,0,255]),
            rgbtoint([255,0,0]), rgbtoint([255,0,0])]
)
plot += dipole_vectors

plot.camera=cam_view

plot.display()

# %%
#.. image:: ../_static/tutorial_06_burst_orientation.png
#   :width: 800
#   :alt:

# %% [markdown]
# We'll simulate a 5mm patch of activity with -10 dB SNR at the sensor level. The desired level of SNR is achieved by adding white noise to the projected sensor signals

# Simulate a beta burst as two sources: one deep and one superficial
prefix=f'sim_{sim_vertex}_burst_'

# Size of simulated sources (mm)
sim_dipfwhm=[5, 5] # mm
# SNR of simulated data (dB)
SNR=-10

# Generate simulated data
burst_sim_fname=run_dipole_simulation(
    base_fname, 
    prefix, 
    [pial_vertex, white_vertex],
    np.vstack([superficial_signal, deep_signal]),
    np.vstack([pial_ori, white_ori]),
    dipole_moment, 
    sim_dipfwhm, 
    SNR,
    spm_instance=spm
)

# %% [markdown]
# Now we'll run a source reconstruction using the multilayer mesh, select the vertex to examine, extract the source signals at each layer in that location, and compute a laminar CSD

[_, _, MU] = invert_ebb(
    burst_sim_fname,
    surf_set,
    patch_size=patch_size,
    n_temp_modes=n_temp_modes,
    return_mu_matrix=True,
    spm_instance=spm
)

pial_layer_vertices = np.arange(verts_per_surf)
pial_layer_ts, time, _ = load_source_time_series(
    burst_sim_fname,
    mu_matrix=MU,
    vertices=pial_layer_vertices
)

# Layer peak
m_layer_max = np.max(np.mean(pial_layer_ts, axis=-1), -1)
peak = np.argmax(m_layer_max)

print(f'Simulated vertex={sim_vertex}, Prior vertex={peak}')

layer_verts = surf_set.get_layer_vertices(peak)
layer_dists = surf_set.get_interlayer_distance(peak)
print(layer_dists)

# Get source time series for each layer
layer_ts, time, _ = load_source_time_series(burst_sim_fname, mu_matrix=MU, vertices=layer_verts)

# Average over trials and compute CSD and smoothed CSD
mean_layer_ts = np.mean(layer_ts, axis=-1)
[csd, smooth_csd] = compute_csd(mean_layer_ts, np.sum(layer_dists), s_rate, smoothing='cubic')

col_r = plt.cm.cool(np.linspace(0, 1, num=surf_set.n_layers))
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
for l in range(surf_set.n_layers):
    plt.plot(mean_layer_ts[l, :], label=f'{l}', color=col_r[l, :])
plt.legend(loc='upper left')
plt.xlabel('Time (ms)')
plt.ylabel('Source (nAm)')

plt.subplot(1, 3, 2)
for l in range(surf_set.n_layers):
    plt.plot(csd[l, :], label=f'{l}', color=col_r[l, :])
plt.legend(loc='upper left')
plt.xlabel('Time (ms)')
plt.ylabel('CSD')

ax = plt.subplot(1, 3, 3)
plot_csd(smooth_csd, time, ax)
plt.xlabel('Time (ms)')
plt.ylabel('Layer')
plt.tight_layout()

# %%
#.. image:: ../_static/tutorial_06_burst_results.png
#   :width: 800
#   :alt:

# %%
spm.terminate()

# Delete simulation files
shutil.rmtree(tmp_dir)

# %%



