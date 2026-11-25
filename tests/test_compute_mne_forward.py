"""Compare laMEG's MNE builder with explicitly computed per-layer forwards."""
from pathlib import Path

import mne
from mne.transforms import Transform, apply_trans
import nibabel as nib
import numpy as np
import pytest
from scipy.io import savemat

from lameg.forward import compute_mne_forward, prepare_mne_forward
from lameg.surf import LayerSurfaceSet
from tests.test_forward import _mesh
from tests.test_mne_forward import _bem_and_recording


def _write_layer_surfaces(subjects_dir):
    """Build two layers with known positions, orientations, and MRI coordinate systems."""
    surfaces = LayerSurfaceSet('example', 2, subjects_dir=str(subjects_dir))
    Path(surfaces.laminar_surf_dir).mkdir(parents=True)

    # Encode a known scanner-RAS to surface-RAS rotation and translation in
    # orig.mgz. LayerSurfaceSet.create saves a NIfTI copy of this same image.
    native_to_ras = np.array([[0., -1., 0., 5.], [1., 0., 0., -9.],
                              [0., 0., 1., 3.], [0., 0., 0., 1.]])
    ras_to_native = np.linalg.inv(native_to_ras)
    voxels = np.zeros((12, 10, 8), np.float32)
    voxel_to_ras = nib.MGHImage(voxels, np.eye(4)).header.get_vox2ras_tkr()
    orig = nib.MGHImage(voxels, ras_to_native @ voxel_to_ras)
    mri_dir = Path(surfaces.subj_dir) / 'mri'
    mri_dir.mkdir()
    nib.save(orig, str(mri_dir / 'orig.mgz'))
    nib.save(orig, surfaces.mri_file)

    # The pial vertex at x=92 mm lies outside the fixture's 90 mm conductor.
    # Its white counterpart lies inside, so common support excludes it at both depths.
    points = {
        'pial': np.array([[18., 12., 4.], [-26., 9., 19.], [13., -21., 8.], [92., 4., 1.]]),
        'white': np.array([[15., 12., 9.], [-29., 7., 20.], [14., -17., 4.], [46., 8., -2.]]),
    }
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    # Store signed link vectors in native MRI axes; these are oblique to the
    # X/Y/Z basis that MNE uses for its free-orientation gains.
    links = (points['white'] - points['pial']) @ ras_to_native[:3, :3].T
    links /= np.linalg.norm(links, axis=1, keepdims=True)
    all_points = []
    for name, ras_points in points.items():
        native = apply_trans(ras_to_native, ras_points)
        _mesh(surfaces.get_mesh_path(name), native, faces, links)
        all_points.append(native)
    _mesh(surfaces.get_mesh_path(), np.vstack(all_points),
          np.vstack((faces, faces + len(links))), np.tile(links, (2, 1)))
    return surfaces, points, native_to_ras


@pytest.mark.parametrize('use_bem_file', [False, True], ids=['in-memory-bem', 'saved-bem'])
def test_builder_matches_per_layer_bem(tmp_path, use_bem_file):
    """Recover the same signed gains and common anatomical IDs as the tutorial loop."""
    bem, info = _bem_and_recording(tmp_path)
    surfaces, points, native_to_ras = _write_layer_surfaces(tmp_path)

    # Register a translated MRI to head coordinates. Request the opposite of
    # MNE's channel order, with different units for the gradiometer and magnetometer.
    mri_head_mm = np.eye(4)
    mri_head_mm[:3, 3] = [1., -2., 3.]
    mri_head_m = mri_head_mm.copy()
    mri_head_m[:3, 3] /= 1000.
    trans = Transform('mri', 'head', mri_head_m)
    with info._unlock():  # pylint: disable=protected-access
        info['ctf_head_t'] = Transform('ctf_head', 'head', np.eye(4))
    slot = {'mesh': {'Affine': mri_head_mm @ native_to_ras},
            'datareg': {'modality': 'MEG', 'fromMNI': np.eye(4)},
            'forward': {'modality': 'MEG', 'sensors': {'coordsys': 'ctf'}}}
    reference = tmp_path / 'registered.mat'
    channels = [{'label': 'MEG0112', 'type': 'MEGPLANAR', 'units': 'fT/cm', 'bad': 0},
                {'label': 'MEG0111', 'type': 'MEGMAG', 'units': 'fT', 'bad': 0}]
    savemat(str(reference), {'D': {'channels': channels, 'other': {'inv': [slot]}}})

    # Reproduce the tutorial's two-header transform and check it against the
    # known RAS positions before solving and importing each layer explicitly.
    orig = nib.load(str(Path(surfaces.mri_file).with_name('orig.mgz')))
    tutorial_transform = (orig.header.get_vox2ras_tkr()
                          @ np.linalg.inv(nib.load(surfaces.mri_file).affine))
    forwards = {}
    for name, ras_points in points.items():
        native = surfaces.load(name, stage='ds', orientation='link_vector', fixed=True)
        positions = apply_trans(tutorial_transform, native.agg_data('pointset'))
        np.testing.assert_allclose(positions, ras_points)
        src = mne.setup_volume_source_space(
            pos={'rr': positions / 1000., 'nn': np.tile([0., 0., 1.], (len(positions), 1))},
            verbose=False)
        fwd = mne.make_forward_solution(info, trans, src, bem, meg=True, eeg=False,
                                        mindist=0., ignore_ref=False, verbose=False)
        forwards[name] = str(tmp_path / f'{name}-fwd.fif')
        mne.write_forward_solution(forwards[name], fwd, verbose=False)
    with pytest.warns(UserWarning, match='Using 3 of 4'):
        expected = prepare_mne_forward(forwards, surfaces, tmp_path, reference=reference,
                                       info=info, ignore_ref=False)
    solution = bem
    if use_bem_file:
        solution = tmp_path / 'example-bem-sol.fif'
        mne.write_bem_solution(str(solution), bem, verbose=False)
    with pytest.warns(UserWarning, match='Using 3 of 4'):
        actual = compute_mne_forward(surfaces, solution, info, trans, reference=reference,
                                     output_dir=tmp_path)
    assert 'make_forward_solution' in actual.operator.provenance
    assert 'mne-solve-' not in actual.operator.provenance

    assert actual.manifest.is_file()
    assert not list(tmp_path.glob('mne-solve-*'))
    assert actual.operator.channel_names == expected.operator.channel_names
    assert actual.operator.channel_names == ('MEG0112', 'MEG0111')
    assert actual.operator.channel_units == expected.operator.channel_units == ('fT/cm', 'fT')
    assert actual.operator.layer_names == expected.operator.layer_names == ('pial', 'white')
    assert actual.support.excluded_by_layer == expected.support.excluded_by_layer == {
        'pial': [3], 'white': []}
    np.testing.assert_array_equal(actual.support.original_vertices, [0, 1, 2])
    np.testing.assert_array_equal(actual.support.original_vertices,
                                  expected.support.original_vertices)
    # FIF rounds gains to float32. Compare all signed coefficients in physical units.
    np.testing.assert_allclose(actual.operator.gain, expected.operator.gain, rtol=2e-6, atol=1e-8)
