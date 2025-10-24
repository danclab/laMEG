"""
Sliding time window laminar model comparison using free energy
==============================================================

This tutorial demonstrates how to perform laminar inference of event-related responses in a sliding time window using model comparison based on free energy as a metric of model fit, described in [Bonaiuto et al., 2021, Laminar dynamics of high amplitude beta bursts in human motor cortex](https://doi.org/10.1016/j.neuroimage.2021.118479). A temporal Gaussian function is simulated at a particular cortical location in various layers. Source reconstruction is performed on the whole time window using the Empirical Bayesian Beamformer on the simulated sensor data using a forward model based on the multilayer mesh as a localizer. This is used to select priors on each layer mesh for a sliding time window model comparison using free energy.
"""

# %%
# Setting up the simulations
# --------------------------
#
# Simulations are based on an existing dataset, which is used to define the sampling rate, number of trials, duration of each trial, and the channel layout.


import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tempfile

from lameg.invert import coregister, invert_ebb, load_source_time_series
from lameg.laminar import sliding_window_model_comparison
from lameg.simulate import run_dipole_simulation
from lameg.surf import LayerSurfaceSet
from lameg.util import get_fiducial_coords
from lameg.viz import show_surface, color_map
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
# For source reconstructions, we need an MRI and a surface mesh. The simulations will be based on a forward model using the multilayer mesh, and the model comparison will use each layer mesh

surf_set_bilam = LayerSurfaceSet(subj_id, 2)
surf_set = LayerSurfaceSet(subj_id, 11)

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
# Simulating a signal on the pial surface
# ---------------------------------------
# We're going to simulate 200ms of a Gaussian with a dipole moment of 5nAm and a width of 25ms

# Strength of simulated activity (nAm)
dipole_moment = 10
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
#.. image:: ../_static/tutorial_04_sim_signal.png
#   :width: 800
#   :alt:

# %% [markdown]
# We need to pick a location (mesh vertex) to simulate at

# Vertex to simulate activity at
sim_vertex=24581

inflated_ds_mesh = surf_set.load('inflated', stage='ds')
coord = inflated_ds_mesh.darrays[0].data[sim_vertex,:]
cam_view = [335, 9.5, 51,
            60, 37, 17,
            0, 0, 1]
plot = show_surface(
    surf_set,
    marker_coords=coord,
    marker_size=5,
    camera_view=cam_view
)

# %%
#.. image:: ../_static/tutorial_04_sim_location.png
#   :width: 800
#   :alt:

# %% [markdown]
# We'll simulate a 5mm patch of activity with -5 dB SNR at the sensor level. The desired level of SNR is achieved by adding white noise to the projected sensor signals

# Simulate at a vertex on the pial surface
pial_vertex = sim_vertex
# Orientation of the simulated dipole
multilayer_mesh = surf_set.load(stage='ds', orientation='link_vector', fixed=True)
pial_unit_norm = multilayer_mesh.darrays[2].data[pial_vertex,:]
prefix = f'sim_{sim_vertex}_pial_'

# Size of simulated patch of activity (mm)
sim_patch_size = 5
# SNR of simulated data (dB)
SNR = -10

# Generate simulated data
pial_sim_fname = run_dipole_simulation(
    base_fname, 
    prefix, 
    pial_vertex, 
    sim_signal, 
    pial_unit_norm, 
    dipole_moment, 
    sim_patch_size, 
    SNR,
    spm_instance=spm
) 

# %% [markdown]
# Localizer inversion
# -------------------
# Now we'll run a source reconstruction using the multilayer mesh, extract the signal in the pial layer, and select a prior based on the peak.

[_,_,MU] = invert_ebb(
    pial_sim_fname,
    surf_set,
    patch_size=patch_size,
    n_temp_modes=n_temp_modes,
    return_mu_matrix=True,
    spm_instance=spm
)

layer_vertices = np.arange(verts_per_surf)
layer_ts, time, ch_names = load_source_time_series(
    pial_sim_fname,
    mu_matrix=MU,
    vertices=layer_vertices
)

m_layer_max = np.max(np.mean(layer_ts,axis=-1),-1)
prior = np.argmax(m_layer_max)

# %% [markdown]
# We can see that the prior is the same as the location we simulated at

# Interpolate for display on the original inflated surface
interpolated_data = surf_set.interpolate_layer_data('pial', m_layer_max, from_stage='ds', to_stage='combined')

inflated_ds_mesh = surf_set.load('inflated', stage='ds')
coord = inflated_ds_mesh.darrays[0].data[prior, :]

# Plot colors and camera view
max_abs = np.max(np.abs(interpolated_data))
c_range = [-max_abs, max_abs]
cam_view = [335, 9.5, 51,
            60, 37, 17,
            0, 0, 1]

# Plot peak
colors, _ = color_map(
    interpolated_data,
    "RdYlBu_r",
    c_range[0],
    c_range[1]
)
thresh_colors = np.ones((colors.shape[0], 4)) * 255
thresh_colors[:, :3] = colors
thresh_colors[interpolated_data < np.percentile(interpolated_data, 99.9), 3] = 0

plot = show_surface(
    surf_set,
    vertex_colors=thresh_colors,
    info=True,
    camera_view=cam_view,
    marker_coords=coord,
    marker_size=5,
    marker_color=[0, 0, 255]
)

# %%
#.. image:: ../_static/tutorial_04_localizer.png
#   :width: 800
#   :alt:

# %% [markdown]
# Sliding time window model comparison (pial - white matter)
# ----------------------------------------------------------
# Now we can run sliding time window model comparison between source models based on the pial and white matter surfaces using free energy. Specifically, we'll look at the difference in free energy between the two models (pial - white matter), in sliding and overlapping windows of 16ms. The free energy difference (pial - white matter) should be positive (more model evidence for the pial surface model) because we simulated activity on the pial surface.

# Number of temporal models for sliding time window inversion
sliding_n_temp_modes = 4
# Size of sliding window (in ms)
win_size = 50
# Whether or not windows overlap
win_overlap = True

# Run sliding time window model comparison between the first layer (pial) and the last layer (white matter)
[Fs,wois] = sliding_window_model_comparison(
    prior, 
    fid_coords,
    pial_sim_fname,
    surf_set_bilam,
    spm_instance=spm,
    invert_kwargs={
        'patch_size': patch_size, 
        'n_temp_modes': sliding_n_temp_modes,
        'win_size': win_size, 
        'win_overlap': win_overlap,
    }
)

# %% [markdown]
# Plot difference in free energy over time (pial minus white) - this should be positive
plt.plot(np.mean(wois,axis=-1), Fs[0,:]-Fs[1,:])
plt.xlabel('Time (ms)')
plt.ylabel(r'$\Delta$F')

# %%
#.. image:: ../_static/tutorial_04_pial_sim_results.png
#   :width: 800
#   :alt:

# %% [markdown]
# White matter surface simulation with pial - white matter sliding time window model comparison
# ---------------------------------------------------------------------------------------------
# Let's simulate the same pattern of activity, in the same location, but on the white matter surface. This time, sliding time window model comparison should yield greater model evidence for the white matter surface, and therefore the difference in free energy (pial - white matter) should be negative.

# Simulate at the corresponding vertex on the white matter surface
white_vertex = (surf_set.n_layers-1)*verts_per_surf+sim_vertex
prefix = f'sim_{sim_vertex}_white_'

# Generate simulated data
white_sim_fname = run_dipole_simulation(
    base_fname,
    prefix,
    white_vertex,
    sim_signal,
    pial_unit_norm,
    dipole_moment,
    sim_patch_size,
    SNR,
    spm_instance=spm
)

# Localizer
[_,_,MU] = invert_ebb(
    white_sim_fname,
    surf_set,
    patch_size=patch_size,
    n_temp_modes=n_temp_modes,
    return_mu_matrix=True,
    spm_instance=spm
)

layer_vertices = np.arange(verts_per_surf)
layer_ts, time, _ = load_source_time_series(
    white_sim_fname,
    mu_matrix=MU,
    vertices=layer_vertices
)

# Layer peak
m_layer_max = np.max(np.mean(layer_ts,axis=-1),-1)
prior = np.argmax(m_layer_max)

# Run sliding time window model comparison between the first layer (pial) and the last layer (white matter)
[Fs,wois] = sliding_window_model_comparison(
    prior,
    fid_coords,
    white_sim_fname,
    surf_set_bilam,
    spm_instance=spm,
    invert_kwargs={
        'patch_size': patch_size, 
        'n_temp_modes': sliding_n_temp_modes,
        'win_size': win_size, 
        'win_overlap': win_overlap,
    }
)

# %% [markdown]
# Plot difference in free energy over time (pial minus white) - this should be negative
plt.plot(np.mean(wois,axis=-1), Fs[0,:]-Fs[1,:])
plt.xlabel('Time (ms)')
plt.ylabel(r'$\Delta$F')

# %%
#.. image:: ../_static/tutorial_04_white_sim_results.png
#   :width: 800
#   :alt:


# %% [markdown]
# Simulation in each layer with sliding time window model comparison across layers
# --------------------------------------------------------------------------------
# That was sliding time window model comparison with two candidate models: one based on the white matter surface, and one on the pial. Let's now simulate on each layer, and for each simulation, run sliding time window model comparison across all layers. We'll turn off SPM visualization here.

# Now simulate at the corresponding vertex on each layer, and for each simulation, run sliding window model
# comparison across all layers
all_layerF = []
for l in range(surf_set.n_layers):
    print(f'Simulating in layer {l}')
    l_vertex = l * verts_per_surf + sim_vertex
    prefix = f'sim_{sim_vertex}{l}_'

    l_sim_fname = run_dipole_simulation(
        base_fname,
        prefix,
        l_vertex,
        sim_signal,
        pial_unit_norm,
        dipole_moment,
        sim_patch_size,
        SNR,
        spm_instance=spm
    )

    # Localizer
    [_, _, MU] = invert_ebb(
        l_sim_fname,
        surf_set,
        patch_size=patch_size,
        n_temp_modes=n_temp_modes,
        return_mu_matrix=True,
        viz=False,
        spm_instance=spm
    )

    layer_vertices = np.arange(verts_per_surf)
    layer_ts, time, _ = load_source_time_series(
        l_sim_fname,
        mu_matrix=MU,
        vertices=layer_vertices
    )

    # Layer peak
    m_layer_max = np.max(np.mean(layer_ts, axis=-1), -1)
    prior = np.argmax(m_layer_max)

    [Fs, wois] = sliding_window_model_comparison(
        prior,
        fid_coords,
        l_sim_fname,
        surf_set,
        viz=False,
        spm_instance=spm,
        invert_kwargs={
            'patch_size': patch_size,
            'n_temp_modes': sliding_n_temp_modes,
            'win_size': win_size,
            'win_overlap': win_overlap,
        }
    )

    all_layerF.append(Fs)
all_layerF = np.squeeze(np.array(all_layerF))

# %% [markdown]
# For each simulation, we can plot the free energy for all models relative to the worst model within a central time window. The layer model with the highest free energy should correspond to the layer that the activity was simulated in.

# Average free energy within small time window in center of the epoch
woi_t = np.mean(wois,axis=-1)
woi_idx = np.where((woi_t>=-20) & (woi_t<=20))[0]
m_all_layerF = np.mean(all_layerF[:,:,woi_idx],axis=2)

col_r = plt.cm.cool(np.linspace(0,1, num=surf_set.n_layers))
plt.figure(figsize=(10,4))

# For each simulation, plot the mean free energy of each layer model relative to that of the worst
# model for that simulation
plt.subplot(1,2,1)
for l in range(surf_set.n_layers):
    layerF = m_all_layerF[l,:]
    plt.plot(layerF-np.min(layerF), label=f'{l}', color=col_r[l,:])
plt.legend()
plt.xlabel('Eval layer')
plt.ylabel(r'$\Delta$F')

# For each simulation, find which layer model had the greatest free energy
plt.subplot(1,2,2)
peaks=[]
for l in range(surf_set.n_layers):
    layerF = m_all_layerF[l,:]
    layerF = layerF-np.min(layerF)
    pk = np.argmax(layerF)
    peaks.append(pk)
plt.plot(peaks)
plt.xlim([-0.5,10.5])
plt.ylim([-0.5,10.5])
plt.plot([0,10],[0,10],'k--')
plt.xlabel('Sim layer')
plt.ylabel(r'Peak $\Delta$F')
plt.tight_layout()

# %%
#.. image:: ../_static/tutorial_04_results.png
#   :width: 800
#   :alt:

# Normalization step
norm_layerF = np.zeros(m_all_layerF.shape)
for l in range(surf_set.n_layers):
    norm_layerF[l,:] = m_all_layerF[l,:] - np.min(m_all_layerF[l,:])

# Transpose for visualization
im=plt.imshow(norm_layerF.T, cmap='Spectral_r')

# Find the indices of the max value in each column
max_indices = np.argmax(norm_layerF, axis=1)

# Plot an 'X' at the center of the square for each column's maximum
for idx, max_idx in enumerate(max_indices):
    plt.text(idx, max_idx, 'X', fontsize=12, ha='center', va='center', color='black', weight='bold')

plt.xlabel('Simulated layer', fontsize=14)
plt.ylabel('Evaluated layer', fontsize=14)
cb=plt.colorbar(im)
cb.set_label(r'$\Delta F$', fontsize=14)

# %%
#.. image:: ../_static/tutorial_04_results_matrix.png
#   :width: 800
#   :alt:

# %%
spm.terminate()

# Delete simulation files
shutil.rmtree(tmp_dir)

# %%



