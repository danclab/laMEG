"""
Laminar model comparison with single-shell and BEM forward models
=================================================================

We'll simulate a 20 Hz burst on the white matter surface of sub-104. Can we tell which surface generated the signal? We'll compare layer models twice on the same simulated data, first with an SPM single-shell forward model and then with an MNE boundary element model (BEM). We'll use the free-energy comparison from `Tutorial 02 <https://danclab.github.io/laMEG/auto_tutorials/tutorial_02_model_comparison_free_energy.html>`_ and `Bonaiuto et al. (2018) <https://doi.org/10.1016/j.neuroimage.2017.11.068>`_.
"""

# %%
# Setting up the simulation
# -------------------------
#
# The recording gives us the sensor positions and epoch timing. Run the notebook from the laMEG ``tutorials`` directory with the `test data <https://osf.io/x4zd3/download>`_ in ``../test_data``.

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from IPython.display import Image
import base64

from lameg.forward import ForwardOperator, compute_mne_forward
from lameg.invert import attach_forward_operator, coregister#, invert_ebb
from lameg.simulate import run_current_density_simulation
from lameg.laminar import model_comparison
from lameg.surf import LayerSurfaceSet
from lameg.util import load_meg_sensor_data, get_fiducial_coords
from lameg.viz import show_surface
import spm_standalone
import matlab

import mne
from mne.transforms import Transform, apply_trans

import nibabel as nib

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
    f'spm/spm-converted_autoreject-{subj_id}-{ses_id}-001-btn_trial-epo.mat'
)
fif_file=os.path.join(
    '../test_data',
    subj_id,
    'meg',
    ses_id,
    'mne',
    f'autoreject-{subj_id}-{ses_id}-001-btn_trial-epo.fif')
info = mne.io.read_info(fif_file)

spm = spm_standalone.initialize()

# %%
# The simulations will be based on a forward model using the multilayer mesh, and the model comparison will use each layer mesh

surf_set_bilam = LayerSurfaceSet(subj_id, 2)
surf_set = LayerSurfaceSet(subj_id, 11)

# %%
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

# %%
# Building the BEM forward model
# ------------------------------
#
# Obviously you will want to take more care when constructing a BEM model, but that is outside the scope of this tutorial. SPM and MNE use different MRI coordinate systems. We'll put the inner skull, cortex and MRI fiducials in FreeSurfer surface RAS, then match the MRI fiducials to their measured positions in the MEG head frame.

# Both MRI transforms start at the same voxel coordinates.
voxel_to_ras = nib.load(os.path.join(surf_set.subj_dir, 'mri', 'orig.mgz')).header.get_vox2ras_tkr()
voxel_to_native = nib.load(os.path.join(surf_set.subj_dir, 'mri', 'orig.nii')).affine
native_to_ras = voxel_to_ras @ np.linalg.inv(voxel_to_native)

mri_fids = [fid_coords[key] for key in ('nas', 'lpa', 'rpa')]
mri_fids = apply_trans(native_to_ras, mri_fids) / 1000  # MNE uses metres.
dig = info.get_montage().get_positions()
head_fids = [dig[key] for key in ('nasion', 'lpa', 'rpa')]
trans = Transform('mri', 'head', mne.coreg.fit_matched_points(mri_fids, head_fids))

# %%
# For the MEG forward model, we'll use a three-compartment BEM representing the brain, skull and scalp. The contours show the inner skull (red), outer skull (yellow) and scalp (orange) on the subject's MRI.

bem_dir = os.path.join(surf_set.subj_dir, 'bem')

# Use the three SPM boundaries at the same resolution.
surfaces = {
    'inner_skull': surf_set.load_head_mesh('iskull'),
    'outer_skull': surf_set.load_head_mesh('oskull'),
    'outer_skin': nib.load(os.path.join(surf_set.subj_dir, 'mri', 'origscalp_2562.surf.gii')),
}

for name, mesh in surfaces.items():
    vertices = apply_trans(native_to_ras, mesh.agg_data('pointset'))
    faces = mesh.agg_data('triangle')
    mne.write_surface(os.path.join(bem_dir, f'{name}.surf'), vertices, faces, overwrite=True)

model = mne.make_bem_model(
    subj_id,
    ico=None,
    conductivity=(0.3, 0.006, 0.3),  # brain, skull, scalp (S/m)
)

bem_solution = mne.make_bem_solution(model)

mne.viz.plot_bem(
    subject=subj_id, mri='orig.mgz',
    slices=[85, 128, 170], show=False,
)
plt.show()

# %%
# .. image:: ../_static/tutorial_10_bem.png
#    :width: 600
#    :alt: Inner-skull BEM boundary on three MRI sections of sub-104

# %%
# Coregister subject's data using the multilayer mesh

# Coregister data to multilayer mesh
skull_fname = os.path.join(surf_set.subj_dir, 'mri', 'origiskull_2562.surf.gii')
coregister(
    fid_coords,
    base_fname,
    surf_set,
    iskull_fname=skull_fname,
    spm_instance=spm
)

# %%
# We'll use the BEM solution to compute the gain on each cortical surface. laMEG projects the lead fields onto the link vectors and keeps the vertices that MNE retained at every depth.

prepared = compute_mne_forward(
    surf_set, bem_solution, info, trans,
    reference=base_fname,
    output_dir=tmp_dir,
    allow_nominal_ctf_coils=True
)
bem = prepared.operator
print(f'{bem.n_columns:,} of {surf_set.get_vertices_per_layer():,} vertices retained per surface')

# Use the cortical surfaces retained by the BEM for simulation and model comparison.
surfaces = prepared.surfaces
mesh_fname = surfaces.get_mesh_path()
layer_names = surfaces.get_layer_names()
n_layers = len(layer_names)

# %%
# Building the single-shell forward model
# ---------------------------------------
#
# We'll give SPM the same inner skull and the same retained pial and white vertices. SPM uses the measured CTF pickup geometry. MNE uses the nominal CTF coils for this acquisition.

coregister(
    fid_coords, base_fname, prepared.surfaces,
    forward_model_type='Single Shell', iskull_fname=skull_fname,
    viz=False, spm_instance=spm
)

# Ask SPM to compute the gain, then read it into laMEG.
modes_fname = os.path.join(tmp_dir, 'single_shell_modes.mat')
spm.spm_eeg_inv_prep_modes_xval(
    base_fname, matlab.double([]), modes_fname, 1.0, 0.0, nargout=0
)

single_shell = ForwardOperator.from_spm(
    base_fname, surfaces.get_mesh_path(), surfaces.get_layer_names(), allow_undefined_gain=True
)
models = [('Single shell', single_shell), ('BEM', bem)]

# %%
# Simulating a signal
# ---------------------------
#
# We're going to simulate 1s of a 20Hz sine wave with a dipole moment of 1nAm

# Frequency of simulated sinusoid (Hz)
freq = 20
# Strength of simulated activity (nAm)
dipole_moment = 1
# Sampling rate (must match the data file)
s_rate = 600

# Generate 1s of a sine wave at a sampling rate of 600Hz (to match the data file)
time = np.linspace(0,1,s_rate+1)
sim_signal = np.sin(time*freq*2*np.pi).reshape(1,-1)

fig=plt.figure()
plt.plot(time,dipole_moment*sim_signal[0,:])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (nAm)')
plt.show()

# %%
# .. image:: ../_static/tutorial_10_sim_signal.png
#    :width: 600
#    :alt:

# Vertex to simulate activity at
sim_vertex=44523

cam_view = [331, 26, -40,
            60, 37, 17,
            0.25, 0, 1]
plot = show_surface(
    surf_set,
    marker_vertices=sim_vertex,
    marker_size=5,
    camera_view=cam_view
)

# %%
#.. image:: ../_static/tutorial_10_sim_location.png
#   :width: 800
#   :alt:

# %%
# Plotting the forward models
# ---------------------------
#
# The first two maps shows the field that a 1 nA m dipole at this vertex produces at the sensors for each forward model. The map on the right shows BEM - single shell

column = prepared.support.map_indices([sim_vertex])[0]

# Each subtraction must pair measurements from the same sensor.
assert tuple(single_shell.channel_names) == tuple(bem.channel_names)
fields = [forward.layer('white').gain[:, column] for _, forward in models]
difference = fields[1] - fields[0]

sensors = mne.pick_channels(info.ch_names, list(bem.channel_names), ordered=True)
sensor_info = mne.pick_info(info, sensors)

fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
topomap = dict(ch_type='mag', show=False, cmap='RdBu_r', contours=0, sensors=False)

# Use the same limits for both forward models so their amplitudes are comparable.
limit = np.max(np.abs(fields))
for ax, (name, _), field in zip(axes[:2], models, fields):
    im, _ = mne.viz.plot_topomap(
        field, sensor_info, axes=ax, vlim=(-limit, limit), **topomap
    )
    ax.set_title(name)
fig.colorbar(im, ax=axes[:2], label='Field for a 1 nA m source (fT)')

# Subtract the signed fields at the sensors before interpolating the difference.
diff_limit = np.max(np.abs(difference)) or limit
im_diff, _ = mne.viz.plot_topomap(
    difference, sensor_info, axes=axes[2], vlim=(-diff_limit, diff_limit), **topomap
)
axes[2].set_title('BEM - single shell')
fig.colorbar(im_diff, ax=axes[2], label='Field difference (fT)')
plt.show()

# %%
# .. image:: ../_static/tutorial_10_lead_fields.png
#    :width: 600
#    :alt: Single-shell and BEM sensor fields for the same white matter dipole

# %%
# BEM simulation in each layer with model comparison across layers using each forward model
# -----------------------------------------------------------------------------------------
# Simulate on each layer using BEM, and for each simulation, run model comparison across all layers. Once using the single shell model and once using the BEM

# Select the BEM gain in the SPM recording used by the simulator.
attach_forward_operator(base_fname, bem, 0, spm, mesh_fname)

# Size of simulated patch of activity (mm)
sim_patch_size = 5
# SNR of simulated data (dB)
SNR = -5

# Rows are simulated layers and columns are evaluated layers.
all_layerF = {
    name: np.full((surf_set.n_layers, surf_set.n_layers), np.nan)
    for name, _ in models
}

for l, layer in enumerate(layer_names):
    print(f'Simulating on layer {l} ({layer})')
    l_vertex = surfaces.get_multilayer_vertex(layer, column)

    l_sim_fname = run_current_density_simulation(
        base_fname, f'sim_{sim_vertex}_{l}_', l_vertex,
        sim_signal, dipole_moment, sim_patch_size, SNR,
        viz=False, spm_instance=spm
    )
    layer_F = {}
    for name, forward in models:
        F, _ = model_comparison(
            fid_coords, l_sim_fname, surfaces,
            forward_operator=forward,
            forward_kwargs={'iskull_fname': skull_fname},
            invert_kwargs={'patch_size': 5, 'n_temp_modes': 4},
            viz=False, spm_instance=spm
        )
        layer_F[name] = F

    for name, F in layer_F.items():
        all_layerF[name][l, :] = F

# %%
# For each simulation, we can plot the free energy for all models relative to the worst model. The layer model with the highest free energy should correspond to the layer that the activity was simulated in.

layers = np.arange(n_layers)
col_r = plt.cm.cool(np.linspace(0, 1, n_layers))

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex='col', sharey='col')

for row, (name, _) in enumerate(models):
    F = all_layerF[name]
    ax_curves, ax_peaks = axes[row]

    # Each colored line is one simulated depth.
    for l in layers:
        ax_curves.plot(layers, F[l] - F[l].min(), color=col_r[l], label=str(l))
    ax_curves.set_title(name)
    ax_curves.set_ylabel(r'$\Delta F$')

    # For each simulation, find the evaluated layer with the highest free energy.
    ax_peaks.plot(layers, np.argmax(F, axis=1), 'o-')
    ax_peaks.plot([0, n_layers - 1], [0, n_layers - 1], 'k--')
    ax_peaks.set_title(name)
    ax_peaks.set_ylabel(r'Layer with peak $\Delta F$')
    ax_peaks.set_yticks(layers)

axes[0, 0].legend(title='Sim layer', ncol=2, fontsize=8)
axes[1, 0].set_xlabel('Evaluated layer')
axes[1, 1].set_xlabel('Simulated layer')
for ax in axes[1]:
    ax.set_xticks(layers)

plt.tight_layout()
plt.show()

# %%
# .. image:: ../_static/tutorial_10_results.png
#    :width: 600
#    :alt:

# Subtract the worst evaluated layer separately for each simulated recording.
relative_F = {
    name: F - F.min(axis=1, keepdims=True)
    for name, F in all_layerF.items()
}
vmax = max(F.max() for F in relative_F.values())

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)

for ax, (name, _) in zip(axes, models):
    F = relative_F[name]
    im = ax.imshow(F.T, cmap='Spectral_r', vmin=0, vmax=vmax, interpolation='nearest')

    for l, peak in enumerate(np.argmax(F, axis=1)):
        ax.text(l, peak, 'X', ha='center', va='center', weight='bold')

    ax.set_title(name)
    ax.set_xlabel('Simulated layer')
    ax.set_xticks(layers)
    ax.set_yticks(layers)

axes[0].set_ylabel('Evaluated layer')
fig.colorbar(im, ax=axes, label=r'$\Delta F$')
plt.show()

# %%
# .. image:: ../_static/tutorial_10_results_matrix.png
#    :width: 600
#    :alt:

# %%
# At this vertex, BEM reconstructs laminar activity perfectly, but single shell does not. Note that we chose the vertex to illustrate a disagreement, and the BEM also generated the data.
#
# We'll close SPM and remove the temporary files when we're finished.

spm.terminate()

# Delete simulation files
shutil.rmtree(tmp_dir)
