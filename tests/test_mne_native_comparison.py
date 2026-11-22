"""Exercise MNE BEM import with the matched CTF FIF and SPM recording from sub-104."""
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pytest
from mne.transforms import Transform, apply_trans
from scipy.io import loadmat

from lameg.forward import ForwardOperator, prepare_mne_forward
from lameg.laminar import model_comparison
from tests.forward_test_utils import separate_dataset


def _bem_forwards(surfaces, info, registration, work):
    """Compute a one-compartment BEM on the participant's registered inner skull."""
    mri = Path(surfaces.mri_file).parent
    # The GIfTIs use native MRI coordinates. Go through the MRI voxel indices to
    # express those same points in FreeSurfer's surface RAS frame, in millimetres.
    tkr = nib.load(str(mri / 'orig.mgz')).header.get_vox2ras_tkr()
    native_to_surface_ras = tkr @ np.linalg.inv(nib.load(surfaces.mri_file).affine)
    # SPM's registration maps native MRI to CTF head. MNE needs surface RAS to
    # MNE head. Compose these affine transforms in millimetres, reading right to
    # left in the matrix product, then convert the resulting translation to metres.
    mne_to_ctf = np.linalg.inv(info['ctf_head_t']['trans'])
    mne_to_ctf[:3, 3] *= 1000
    mri_to_mne = np.linalg.inv(mne_to_ctf) @ registration @ np.linalg.inv(native_to_surface_ras)
    mri_to_mne[:3, 3] /= 1000

    # Use the subject's MRI-derived inner-skull boundary for this single-compartment
    # BEM. Transform it with the same affine as the dipoles to keep them registered.
    skull = nib.load(str(mri / 'origiskull_2562.surf.gii'))
    points = skull.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = skull.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    bem_path = work / surfaces.subj_id / 'bem'
    bem_path.mkdir(parents=True)
    mne.write_surface(str(bem_path / 'inner_skull.surf'),
                      apply_trans(native_to_surface_ras, points), faces)
    model = mne.make_bem_model(surfaces.subj_id, ico=None, conductivity=(0.3,), subjects_dir=work)
    bem = mne.make_bem_solution(model)
    forwards = {}
    layers = surfaces.get_layer_names()
    meshes = [nib.load(surfaces.get_mesh_path(layer_name=name)) for name in layers]
    for name, mesh in zip(layers, meshes):
        positions = apply_trans(native_to_surface_ras, mesh.darrays[0].data) / 1000
        # The discrete source space retains the GIfTI's vertex numbering. MNE computes
        # all three Cartesian gain components per active source. The z unit vector
        # satisfies the source-space API. laMEG projects onto the signed GIfTI direction.
        source_space = mne.setup_volume_source_space(
            pos={'rr': positions, 'nn': np.tile([0., 0., 1.], (len(positions), 1))})
        # Zero minimum distance permits sources right up to the inner skull. MNE still
        # excludes exterior points. Include the acquisition's CTF reference sensors.
        forwards[name] = mne.make_forward_solution(
            info, Transform('mri', 'head', mri_to_mne), source_space, bem,
            meg=True, eeg=False, mindist=0., ignore_ref=False)

    return forwards, mne_to_ctf


def test_ctf_bem_layer_comparison(tmp_path, spm, forward_registration):
    """Signed MNE predictions survive unit conversion and installation for seeded MSP."""
    dataset, fids, original_surfaces, donor, native = forward_registration
    fif = (dataset.parent.parent / 'mne' /
           'autoreject-sub-104-ses-01-001-btn_trial-epo.fif')
    if not fif.is_file():
        pytest.skip('requires the matched sub-104 FIF and SPM dataset')
    info = mne.io.read_info(str(fif))
    forwards, head = _bem_forwards(original_surfaces, info, native.mri_to_head_mm, tmp_path)
    # The donor supplies the recording units and registration. The matching epoch Info
    # supplies the CTF head transform and compensation weights. Declare the solve's
    # reference setting and permit comparison of nominal MNE against measured CTF coils.
    with pytest.warns(UserWarning, match='cortical vertices.*pial'):
        preparation = prepare_mne_forward(
            forwards, original_surfaces, tmp_path, reference=donor, info=info,
            ignore_ref=False, allow_nominal_ctf_coils=True)
    surfaces, operator = preparation.surfaces, preparation.operator
    # The forward's vertno lists the original IDs that survived MNE's boundary check.
    # This test used one discrete source space per layer. Intersect their active IDs
    # to check the helper independently and require a genuine exclusion in this case.
    common = np.intersect1d(*(forward['src'][0]['vertno'] for forward in forwards.values()))
    assert 0 < len(common) < native.n_columns
    np.testing.assert_array_equal(preparation.support.original_vertices, common)

    layers = [operator.layer(name) for name in surfaces.get_layer_names()]
    defined = np.flatnonzero(np.logical_and.reduce([layer.orientation_defined for layer in layers]))
    # Drive three known dipoles with unequal, mixed-sign currents. A swapped source
    # column or lost direction sign changes the resulting channel-by-channel prediction.
    patch = defined[[0, len(defined) // 2, -1]]
    amplitudes = np.array([0.73, -1., 1.1])  # signed nA m
    for layer, forward in zip(layers, forwards.values()):
        # Invert the rotation for row vectors to express currents in MNE head axes.
        # MNE uses A m and T here. The imported gain uses recorded fT per nA m.
        # patch addresses compact columns, while common[patch] indexes the MNE sources.
        vectors = (layer.orientations_head[patch] @ head[:3, :3]) * amplitudes[:, None] * 1e-9
        stc = mne.VolVectorSourceEstimate(vectors[:, :, None], [common[patch]], tmin=0., tstep=.01)
        predicted = mne.apply_forward(forward, stc, info).pick_channels(list(layer.channel_names),
                                                                       ordered=True)
        np.testing.assert_allclose(layer.gain[:, patch] @ amplitudes, predicted.data[:, 0] * 1e15,
                                   atol=1e-7, rtol=1e-7)

    working = separate_dataset(dataset, tmp_path / 'supplied')
    # A seed names an original anatomical site. If one layer lacks that site, model
    # comparison must stop before installing any candidate layer's gain on the dataset.
    excluded = np.setdiff1d(np.arange(native.n_columns), common)
    with pytest.raises(ValueError, match=f'vertex {excluded[0]}.*excluded'):
        model_comparison(
            fids, working, surfaces, method='MSP', viz=False, spm_instance=spm,
            forward_operator=operator, invert_kwargs={'priors': [int(excluded[0])]})
    assert not list(Path(working).parent.glob('lameg_gain_*.mat'))

    original_seed = int(common[patch[1]])
    assert original_seed != patch[1]  # Removing pial vertices changed the mesh index.
    evidence, _ = model_comparison(
        fids, working, surfaces, method='MSP', viz=False, spm_instance=spm,
        forward_operator=operator, invert_kwargs={'priors': [original_seed],
                                                  'woi': [100., 200.], 'n_temp_modes': 4})
    assert evidence.size == 2 and np.isfinite(evidence).all()
    # Read SPM's patch selector to check the seed actually fitted. A finite evidence
    # could also come from fitting the wrong cortical location after compaction.
    compact_prior = loadmat(str(Path(working).with_name('patch_0.mat')))['Ip'].item()
    assert compact_prior == patch[1] + 1  # MATLAB uses one-based indexing.
    # The comparison reuses one SPM inversion slot. After the loop the final layer's
    # signed gain and solver provenance should be recoverable from that slot.
    last = surfaces.get_layer_names()[-1]
    saved = ForwardOperator.from_spm(working, surfaces.get_mesh_path(layer_name=last), [last])
    np.testing.assert_array_equal(saved.gain, operator.layer(last).gain)
    assert saved.provenance == operator.provenance
