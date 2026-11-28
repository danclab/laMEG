"""Test DUNEuro source identity, frame changes and physical channel gains."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat

from lameg import _duneuro
from lameg._forward_support import ForwardSupport
from lameg.forward import ForwardOperator, compute_duneuro_forward
from lameg.surf import LayerSurfaceSet
from tests.test_forward import _mesh


@pytest.fixture(name='registration')
def make_registration(tmp_path):
    """Two signed laminar meshes after an independent BEM support reduction."""
    # The first triangle survives each FEM mask. The last column fails one
    # check at one depth. Vertex 2's direction is undefined in both GIfTIs.
    (tmp_path / 'example').mkdir()
    surfaces = LayerSurfaceSet('example', 2, subjects_dir=str(tmp_path))
    Path(surfaces.laminar_surf_dir).mkdir(parents=True)
    pial = np.array([[0., 0., 12.], [8., 0., 12.], [0., 8., 12.], [8., 8., 12.]])
    white = pial - [0., 0., 3.]
    directions = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    native = np.vstack((pial, white))
    for layer, points in zip(surfaces.get_layer_names(), (pial, white)):
        _mesh(surfaces.get_mesh_path(layer), points, faces, directions)
    _mesh(surfaces.get_mesh_path(), native, np.vstack((faces, faces + len(pial))),
          np.tile(directions, (2, 1)))
    surfaces.forward_support = ForwardSupport(
        7, np.array([1, 2, 4, 5]), {'pial': [0, 6], 'white': [3]})

    # Rotation and translation are intentional. A copied single-shell effective
    # location would shift every FEM source, while transforming a direction as
    # a point would give it part of the translation.
    mri_to_head = np.array([[1., 0., 0., 11.], [0., 0., -1., -4.],
                            [0., 1., 0., 7.], [0., 0., 0., 1.]])
    head = native @ mri_to_head[:3, :3].T + mri_to_head[:3, 3]
    moments = np.tile(directions, (2, 1)) @ mri_to_head[:3, :3].T
    geometry = {
        'coilpos': np.array([[0.12, 0.04, 0.08], [0.14, 0.07, 0.11], [0.16, 0.06, 0.13]]),
        'coilori': np.eye(3),
        # Third coil is inactive; the other two contribute signed weights to
        # both channels, including a gradiometer in reciprocal metres.
        'tra': np.array([[1., -0.25, 0.], [-50., 50., 0.]])}
    gain = np.arange(16.).reshape(2, 8)
    gain[:, [2, 6]] = 0.
    reference = ForwardOperator(
        gain, ('B', 'A'), ('fT', 'fT/cm'), ('pial', 'white'), native, head + [0., 0., 0.5],
        moments, mri_to_head, faces, geometry, 'single shell')
    return surfaces, reference


def test_mixed_layer_sizes_before_loading_charm(tmp_path, registration):
    """Report an already cropped single-layer file before any FEM work is needed."""
    surfaces, reference = registration
    # The combined mesh has four columns. Simulate a white surface left over
    # from a preparation that retained only its first triangle.
    points = reference.layer('white').requested_mri_mm[:3]
    _mesh(surfaces.get_mesh_path('white'), points, [[0, 1, 2]], np.ones((3, 3)))
    with pytest.raises(ValueError, match='white mesh .* has 3 vertices.*4 vertices per layer'):
        compute_duneuro_forward(surfaces, tmp_path / 'unread.msh', reference=reference,
                                output_dir=tmp_path)


def _recording_header(tmp_path, sensor_geometry, mri_to_head):
    """Write just the coregistration, good channels and SI pickup geometry to disk."""
    # The recording rows are B, A. SPM's physical sensor table uses the
    # opposite order, so the reader must match the signed weights by label.
    channels = np.array([('B', 'MEGMAG', 'fT', 0), ('A', 'MEGGRAD', 'fT/cm', 0)],
                        dtype=[('label', 'O'), ('type', 'O'), ('units', 'O'), ('bad', 'O')])
    geometry = dict(sensor_geometry)
    geometry.update(label=np.array(['A', 'B'], dtype=object),
                    chanunit=np.array(['T/m', 'T'], dtype=object), unit='m', coordsys='ctf',
                    tra=geometry['tra'][::-1])
    registered = {'datareg': {'modality': 'MEG', 'fromMNI': mri_to_head},
                  'mesh': {'Affine': np.eye(4)},
                  'forward': {'modality': 'MEG', 'siunits': 1, 'sensors': geometry}}
    # Slot 0 is empty. Slot 1 has no gainmat, source tessellation or forward
    # channel list (SPM creates that list when it calculates the gain).
    path = tmp_path / 'recording.mat'
    savemat(path, {'D': {'channels': channels, 'other': {'inv': [{}, registered]}}})
    return path


def _mock_source_locations(monkeypatch, mesh, fem_m, native_tag):
    """Exercise both independent source checks before the pickup solve."""
    def compartments(model, positions):
        assert model is mesh
        np.testing.assert_allclose(positions, fem_m, atol=1e-14)
        tags = np.full(len(positions), 2)
        if native_tag is None:
            tags[3] = 3  # Last pial vertex lies in CSF.
        return tags

    def native_tags(model, positions):
        assert model is mesh
        # A pial tissue exclusion makes its white counterpart ineligible too.
        # Neither search nor transfer should see an undefined dipole moment.
        indices = [0, 1, 4, 5] if native_tag is None else [0, 1, 3, 4, 5, 7]
        np.testing.assert_allclose(positions, fem_m[indices], atol=1e-14)
        tags = np.full(len(indices), 2)
        if native_tag is not None:
            tags[-1] = native_tag  # DUNEuro failed to confirm GM at white depth.
        return tags

    monkeypatch.setattr(_duneuro, 'source_compartments', compartments)
    monkeypatch.setattr(_duneuro, 'native_source_tags', native_tags)


def _assert_exclusions(report, native_tag):
    # The last input-mesh vertex has original anatomical ID 5.
    # Keep the source of the exclusion separate from inherited BEM exclusions.
    if native_tag is None:
        assert report['excluded_by_layer'] == {'pial': [0, 5, 6], 'white': [3]}
        assert report['excluded_by_tissue'] == {'pial': [5], 'white': []}
        assert report['excluded_by_native_search'] == {'pial': [], 'white': []}
    else:
        assert report['excluded_by_layer'] == {'pial': [0, 6], 'white': [3, 5]}
        assert report['excluded_by_tissue'] == {'pial': [], 'white': []}
        assert report['excluded_by_native_search'] == {'pial': [], 'white': [5]}
    assert report['vertices_without_faces'] == []


@pytest.mark.parametrize('native_tag', [None, 0, 3], ids=['pial_csf', 'native_miss', 'native_csf'])
@pytest.mark.parametrize('from_recording', [True, False], ids=['spm_header', 'operator'])
def test_fem_frames_channels_and_common_support(tmp_path, registration, monkeypatch,
                                                from_recording, native_tag):
    """A gain-free SPM header and a comparison operator produce the same physical gains."""
    surfaces, reference = registration
    mri_to_fem = np.array([[0., -1., 0., 20.], [1., 0., 0., 10.],
                           [0., 0., 1., -14.], [0., 0., 0., 1.]])
    native = reference.requested_mri_mm
    fem_m = (native @ mri_to_fem[:3, :3].T + mri_to_fem[:3, 3]) / 1000.
    head_to_fem = mri_to_fem @ np.linalg.inv(reference.mri_to_head_mm)
    geom = reference.sensor_geometry
    mesh = SimpleNamespace(conductivities={2: 0.33}, tetrahedra=np.array([[0, 1, 2, 3]]))
    monkeypatch.setattr(_duneuro, 'read_charm_mesh', lambda *args, **kwargs: mesh)

    def physical_gain(model, positions, moments, coils, normals, weights, *,
                      n_jobs, output_dir, verbose):
        assert (n_jobs, output_dir, verbose) == (2, tmp_path, True)
        assert model is mesh
        # Two defined dipoles per layer reach the solver. The zero-link source
        # keeps its anatomical index and receives a zero gain in each model.
        defined = [0, 1, 4, 5]
        np.testing.assert_allclose(positions, fem_m[defined])
        np.testing.assert_allclose(moments,
                                   reference.orientations_head[defined] @ head_to_fem[:3, :3].T)
        np.testing.assert_allclose(coils, geom['coilpos'][:2] @ head_to_fem[:3, :3].T
                                   + head_to_fem[:3, 3] / 1000.)
        np.testing.assert_allclose(normals, geom['coilori'][:2] @ head_to_fem[:3, :3].T)
        np.testing.assert_array_equal(weights, geom['tra'][:, :2])
        return np.array([[2., -3., 4., -5.], [6., -7., 8., -9.]])

    _mock_source_locations(monkeypatch, mesh, fem_m, native_tag)
    monkeypatch.setattr(_duneuro, 'solve_forward', physical_gain)
    options = ({'reference': _recording_header(tmp_path, geom, reference.mri_to_head_mm),
                'inversion_idx': 1}
               if from_recording else {'reference': reference})
    with pytest.warns(UserWarning, match='Using 3 of 7'):
        fem = compute_duneuro_forward(
            surfaces, tmp_path / 'example.msh', output_dir=tmp_path,
            conductivities={2: 0.33}, mri_to_fem_mm=mri_to_fem, n_jobs=2, verbose=True, **options)
    # The SI gain was per A m. Conversion to fT per nA m gives 10^6, and
    # fT/cm per nA m gives 10^4 for the second row (T/m from the solver).
    expected = [[2e6, -3e6, 0., 4e6, -5e6, 0.], [6e4, -7e4, 0., 8e4, -9e4, 0.]]
    np.testing.assert_allclose(fem.operator.gain, expected)
    assert fem.operator.channel_names == reference.channel_names
    assert fem.operator.channel_units == reference.channel_units
    kept = [0, 1, 2, 4, 5, 6]
    np.testing.assert_allclose(fem.operator.effective_head_mm,
                               reference.effective_head_mm[kept] - [0., 0., 0.5])
    np.testing.assert_array_equal(fem.support.original_vertices, [1, 2, 4])
    np.testing.assert_array_equal(fem.support.map_indices([4, 8], multilayer=True), [2, 3])
    with pytest.raises(ValueError, match='excluded'):
        fem.support.map_indices([5])
    # Restrict two independent comparison models, keeping their own gains and
    # physical positions. All three must match the same combined and layer meshes.
    bem = replace(reference, gain=2 * reference.gain, provenance='BEM')
    for source in (reference, bem):
        fitted = fem.restrict_operator(source)
        np.testing.assert_array_equal(fitted.gain, source.gain[:, kept])
        np.testing.assert_array_equal(fitted.effective_head_mm, source.effective_head_mm[kept])
        fitted.validate_mesh(None, fem.surfaces.get_mesh_path())
    fem.operator.validate_mesh('white', fem.surfaces.get_mesh_path('white'))
    report = json.loads(fem.manifest.read_text(encoding='utf-8'))
    _assert_exclusions(report, native_tag)
    assert report['original_vertices'] == [1, 2, 4]
    # A matrix whose channel/column counts happen to match is insufficient.
    # A different source order is rejected even on an equal-sized input mesh.
    reordered = replace(reference, requested_mri_mm=reference.requested_mri_mm[::-1])
    with pytest.raises(ValueError, match='positions'):
        fem.restrict_operator(reordered)
