"""Exercise MNE BEM import with the matched CTF FIF and SPM recording from sub-104."""
from copy import copy
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pytest
from mne.transforms import Transform, apply_trans

from lameg.forward import ForwardOperator
from lameg.laminar import model_comparison
from lameg.surf import _concatenate_surfaces, _remove_vertices
from tests.forward_test_utils import separate_dataset


def _bem_forwards(surfaces, info, registration, work):
    """Compute a one-compartment BEM on the participant's registered inner skull."""
    mri = Path(surfaces.mri_file).parent
    tkr = nib.load(str(mri / 'orig.mgz')).header.get_vox2ras_tkr()
    native_to_surface_ras = tkr @ np.linalg.inv(nib.load(surfaces.mri_file).affine)
    mne_to_ctf = np.linalg.inv(info['ctf_head_t']['trans'])
    mne_to_ctf[:3, 3] *= 1000
    mri_to_mne = np.linalg.inv(mne_to_ctf) @ registration @ np.linalg.inv(native_to_surface_ras)
    mri_to_mne[:3, 3] /= 1000

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
        # MNE computes all three Cartesian components for each discrete source.
        source_space = mne.setup_volume_source_space(
            pos={'rr': positions, 'nn': np.tile([0., 0., 1.], (len(positions), 1))})
        forwards[name] = mne.make_forward_solution(
            info, Transform('mri', 'head', mri_to_mne), source_space, bem,
            meg=True, eeg=False, mindist=0., ignore_ref=False)

    # The same anatomical columns must exist in both models. MNE omits pial vertices
    # outside the skull; explicitly construct both test meshes on their intersection.
    common = np.intersect1d(*(forward['src'][0]['vertno'] for forward in forwards.values()))
    drop = np.setdiff1d(np.arange(len(meshes[0].darrays[0].data)), common)
    meshes = [_remove_vertices(mesh, drop) for mesh in meshes]
    (work / 'surfaces').mkdir()
    surfaces.laminar_surf_dir = str(work / 'surfaces')
    for name, mesh in zip(layers, meshes):
        nib.save(mesh, surfaces.get_mesh_path(layer_name=name))
    nib.save(_concatenate_surfaces(meshes), surfaces.get_mesh_path())
    return forwards, common, mne_to_ctf


def test_ctf_bem_layer_comparison(tmp_path, spm, forward_registration):
    """Signed MNE predictions survive unit conversion and installation for seeded MSP."""
    dataset, fids, original_surfaces, _, native = forward_registration
    surfaces = copy(original_surfaces)
    fif = (dataset.parent.parent / 'mne' /
           'autoreject-sub-104-ses-01-001-btn_trial-epo.fif')
    if not fif.is_file():
        pytest.skip('requires the matched sub-104 FIF and SPM dataset')
    info = mne.io.read_info(str(fif))
    forwards, common, head = _bem_forwards(surfaces, info, native.mri_to_head_mm, tmp_path)
    operator = ForwardOperator.from_mne(
        forwards, surfaces.get_mesh_path(),
        vertices=np.column_stack([np.zeros_like(common), common]),
        channel_units=dict(zip(native.channel_names, native.channel_units)),
        mri_to_head_mm=native.mri_to_head_mm, mne_head_to_head_mm=head, info=info,
        ignore_ref=False, allow_nominal_ctf_coils=True)

    layers = [operator.layer(name) for name in surfaces.get_layer_names()]
    defined = np.flatnonzero(np.logical_and.reduce([layer.orientation_defined for layer in layers]))
    patch = defined[[0, len(defined) // 2, -1]]
    amplitudes = np.array([0.73, -1., 1.1])  # signed nA m
    for layer, forward in zip(layers, forwards.values()):
        vectors = (layer.orientations_head[patch] @ head[:3, :3]) * amplitudes[:, None] * 1e-9
        stc = mne.VolVectorSourceEstimate(vectors[:, :, None], [common[patch]], tmin=0., tstep=.01)
        predicted = mne.apply_forward(forward, stc, info).pick_channels(list(layer.channel_names),
                                                                       ordered=True)
        np.testing.assert_allclose(layer.gain[:, patch] @ amplitudes, predicted.data[:, 0] * 1e15,
                                   atol=1e-7, rtol=1e-7)

    working = separate_dataset(dataset, tmp_path / 'supplied')
    evidence, _ = model_comparison(
        fids, working, surfaces, method='MSP', viz=False, spm_instance=spm,
        forward_operator=operator, invert_kwargs={'priors': [int(patch[1])],
                                                  'woi': [100., 200.], 'n_temp_modes': 4})
    assert evidence.size == 2 and np.isfinite(evidence).all()
    last = surfaces.get_layer_names()[-1]
    saved = ForwardOperator.from_spm(working, surfaces.get_mesh_path(layer_name=last), [last])
    np.testing.assert_array_equal(saved.gain, operator.layer(last).gain)
    assert saved.provenance == operator.provenance
