"""Numerical and mesh conventions at the DUNEuro Python binding boundary."""
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import sys

import nibabel as nib
import numpy as np
import pytest

from lameg import _duneuro
from lameg.forward import compute_duneuro_forward
from lameg.surf import LayerSurfaceSet, _concatenate_surfaces, _create_surf_gifti
from tests.test_duneuro_forward import _recording_header


def _write_volume(path, points_mm, tetrahedra, tags):
    """Write a CHARM-like Gmsh file with labeled tetrahedra and a boundary triangle."""
    meshio = pytest.importorskip('meshio')
    cells = [('triangle', tetrahedra[:1, :3]), ('tetra', tetrahedra)]
    # CHARM stores meshes in mm and tags boundary elements separately from
    # tetrahedra. A surface tag such as 1002 must never index a conductivity.
    tag_blocks = [np.array([1002]), np.asarray(tags)]
    meshio.write_points_cells(str(path), points_mm, cells, file_format='gmsh22',
                              cell_data={'gmsh:physical': tag_blocks,
                                         'gmsh:geometrical': tag_blocks})


@pytest.fixture(name='charm_mesh')
def make_charm_mesh(tmp_path):
    """Adjacent tetrahedra with nonconsecutive physical tissue IDs."""
    mesh_path = tmp_path / 'head.msh'
    points = np.array([[0., 0., 0.], [40., 0., 0.], [0., 40., 0.],
                       [0., 0., 40.], [0., 0., -40.]])
    tets = np.array([[1, 0, 2, 4], [0, 1, 2, 3]])
    _write_volume(mesh_path, points, tets, [8, 2])
    return mesh_path


def test_volume_tags_and_exact_containment(charm_mesh):
    """Keep native tetrahedral labels after converting the subject mesh to metres."""
    mesh = _duneuro.read_charm_mesh(charm_mesh, {2: 0.33, 8: 0.02})
    # The third point lies outside both tets even though it is inside their
    # bounding box. The last point lies on the mixed GM/skull face and has
    # no unique physical compartment. Nearest-vertex lookup would force a tag.
    positions = np.array([[0.01, 0.01, 0.01], [0.01, 0.01, -0.01], [0.035, 0.035, 0.015],
                          [0.01, 0.01, 0.]])
    np.testing.assert_array_equal(_duneuro.source_compartments(mesh, positions), [2, 8, 0, 0])
    # Known CHARM tags resolve against the standard S/m conductivity table.
    standard = _duneuro.read_charm_mesh(charm_mesh)
    assert standard.conductivities == {2: 0.275, 8: 0.025}
    with pytest.raises(ValueError, match='8'):
        _duneuro.read_charm_mesh(charm_mesh, {2: 0.33})


def test_unrecognized_tissue_requires_explicit_conductivity(tmp_path):
    """A custom compartment must not inherit a conductivity from its array index."""
    path = tmp_path / 'unknown.msh'
    _write_volume(path, np.eye(4, 3) * 20., np.array([[0, 1, 2, 3]]), [42])
    with pytest.raises(ValueError, match='42'):
        _duneuro.read_charm_mesh(path)
    mesh = _duneuro.read_charm_mesh(path, {42: 0.19})
    assert mesh.conductivities == {42: 0.19}


def test_native_search_on_nonconvex_mesh(tmp_path):
    """Match native source projection on the inside edge of a U-shaped conductor."""
    pytest.importorskip('duneuropy', reason='requires the compiled DUNEuro Python binding')
    # Seven 10 mm cubes form a connected U. Split each cube into six tetrahedra.
    # A walk between the two arms can hit the external boundary across the gap.
    cubes = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0),
                      (2, 0, 0), (2, 1, 0), (2, 2, 0)])
    offsets = [np.vstack(([0, 0, 0], np.cumsum(order, axis=0)))
               for order in permutations(np.eye(3, dtype=int))]
    corners = cubes[None, :, None, :] + np.array(offsets)[:, None, :, :]
    points_mm, ids = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    points_mm = points_mm.astype(float) * 10.
    tets = ids.reshape(-1, 4)
    inverted = np.linalg.det(points_mm[tets[:, 1:]] - points_mm[tets[:, :1]]) < 0
    tets[inverted, :2] = tets[inverted, 1::-1]
    # Distinct tissue labels remain distinguishable with equal conductivities.
    tags = np.full(len(tets), 2)
    tags[1] = 8
    path = tmp_path / 'u-shaped.msh'
    _write_volume(path, points_mm, tets, tags)
    mesh = _duneuro.read_charm_mesh(path, {2: 0.33, 8: 0.33})
    sources = mesh.points_m[mesh.tetrahedra[[0, 1, 9]]].mean(axis=1)
    np.testing.assert_array_equal(_duneuro.source_compartments(mesh, sources), [2, 8, 2])

    # The third point is a true tetrahedron centroid. The legacy native solver
    # raises the same edge-hopping error as on a CHARM head. Take the outcome
    # from the installed solver so an improved locator can retain this dipole.
    expected = [2, 8, 2]
    try:
        _duneuro.solve_forward(mesh, sources[2:], np.array([[1., 0., 0.]]),
                               np.array([[0.05, 0.03, 0.04]]), np.array([[0., 0., 1.]]), np.eye(1))
    except RuntimeError as exc:
        if 'coordinate is outside of the grid, or grid is not convex' not in str(exc):
            raise
        expected[2] = 0
    np.testing.assert_array_equal(_duneuro.native_source_tags(mesh, sources), expected)


def test_native_search_propagates_other_errors(monkeypatch):
    """A failed native allocation must abort rather than exclude a cortical column."""
    mesh = SimpleNamespace(points_m=np.eye(4, 3), tetrahedra=np.array([[0, 1, 2, 3]]),
                           tissue_tags=np.array([2]), conductivities={2: 0.33})

    def fail_lookup(_dipole):
        raise RuntimeError('native allocation failed')

    stats = SimpleNamespace(conductivity=fail_lookup)
    dune = SimpleNamespace(DipoleStatistics3D=lambda _config: stats,
                           Dipole3d=lambda position, moment: (position, moment))
    monkeypatch.setattr(_duneuro, '_bindings', lambda: dune)
    with pytest.raises(RuntimeError, match='native allocation failed'):
        _duneuro.native_source_tags(mesh, np.array([[0.25, 0.25, 0.25]]))


def test_missing_python_binding(monkeypatch):
    """Give a Python-specific setup error when the optional extension is absent."""
    monkeypatch.setitem(sys.modules, 'duneuropy', None)
    with pytest.raises(ImportError, match='duneuro-py'):
        _duneuro._bindings()  # pylint: disable=protected-access


def test_primary_secondary_and_signed_coil_response(monkeypatch):
    """Exercise the unit, sign, and coil conventions without a compiled binding."""
    # A minimal in-memory conductor keeps this numerical regression available
    # in normal CI, independently of the mesh reader and the compiled solver.
    mesh = SimpleNamespace(points_m=np.eye(4, 3), tetrahedra=np.array([[0, 1, 2, 3]]),
                           tissue_tags=np.array([2]), conductivities={2: 0.33})
    # Use enough dipoles to cross the primary-field memory limit. The native
    # solver can still share one mesh-wide source model across all of them.
    source = np.tile([0., 0., 0.02], (514, 1))
    directions = np.tile([1., 0., 0.], (len(source), 1))
    # This sign pattern also crosses each 256-dipole boundary at a different phase.
    directions[1::3, 0] = -1.
    coils = np.array([[0., 0., 0.12], [0., 0., 0.10]])
    coil_directions = np.array([[0., 1., 0.], [0., -1., 0.]])
    weights = np.array([[1., -0.5], [0., 1.]])
    transfers = []
    applications = []
    primary_sizes = []

    class Driver:
        """Return a known projected volume-current kernel for each signed dipole."""

        def __init__(self, config):
            assert config['element_type'] == 'tetrahedron'
            # Older bindings otherwise allocate one mesh-sized flux integral per pickup.
            assert config['meg']['cache']['enable'] is False

        def setCoilsAndProjections(self, points, projections):  # pylint: disable=invalid-name
            """Check the coil positions and the direction of each projection."""
            np.testing.assert_allclose(points, coils)
            np.testing.assert_allclose(np.asarray(projections).reshape(-1, 3), coil_directions)

        def computeMEGTransferMatrix(self, config):  # pylint: disable=invalid-name
            """Record each expensive transfer solve independently of the dipoles."""
            assert config['numberOfThreads'] == config['solver']['numberOfThreads'] == 1
            transfers.append(config)
            return np.ones((2, 5)), {}

        def applyMEGTransfer(self, transfer, dipoles, config):  # pylint: disable=invalid-name
            """Use signed, nonzero volume-current responses to test the total field."""
            assert np.asarray(transfer).size == 10
            assert 'source_model' in config
            assert config['numberOfThreads'] == 1
            applications.append(len(dipoles))
            return [moment[0] * np.array([20., -35.]) for _, moment in dipoles], {}

    def dipole(position, moment=None):
        if moment is None:  # The binding also accepts one concatenated six-vector.
            return np.asarray(position[:3]), np.asarray(position[3:])
        return position, moment

    binding = SimpleNamespace(MEEGDriver3d=Driver, FieldVector3D=np.asarray, Dipole3d=dipole)
    monkeypatch.setattr(_duneuro, '_bindings', lambda: binding)
    primary_field = _duneuro._primary_field  # pylint: disable=protected-access

    def primary_in_blocks(positions, moments, pickups, projections):
        primary_sizes.append(len(positions))
        return primary_field(positions, moments, pickups, projections)

    monkeypatch.setattr(_duneuro, '_primary_field', primary_in_blocks)
    actual = _duneuro.solve_forward(mesh, source, directions, coils, coil_directions, weights)
    # Direct Biot–Savart at 0.10 m and 0.08 m gives projected kernels -100
    # and +156.25; subtract volume-current kernels +20 and -35 before applying
    # mu0/(4*pi). Reversing the signed dipole reverses both contributions.
    expected_coils = 1e-7 * np.array([[-120.], [191.25]]) * directions[:, 0]
    expected = weights @ expected_coils
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)
    assert len(transfers) == 1
    assert applications == [len(source)]
    assert sum(primary_sizes) == len(source) and max(primary_sizes) < len(source)
    # Requiring an explicit positive worker count prevents an accidental request
    # for one full FEM driver per visible core (joblib's meaning of -1).
    for invalid_count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='n_jobs'):
            _duneuro.solve_forward(mesh, source, directions, coils, coil_directions, weights,
                                   n_jobs=invalid_count)


def _write_sphere(path):
    """Generate a small, homogeneous tetrahedral approximation to a 90 mm sphere."""
    # Split each grid cube along its body diagonal. Each axis permutation
    # traces a different one of the six tetrahedra in the cube. The radial map
    # puts all cube boundary vertices on the sphere.
    count = 13
    axis = np.linspace(-1., 1., count)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    radii = np.linalg.norm(points, axis=1)
    shell = np.max(abs(points), axis=1)
    points *= (90. * shell / np.where(radii == 0, 1., radii))[:, None]
    indices = np.arange(len(points)).reshape((count, count, count))
    starts = np.indices((count - 1,) * 3).reshape(3, -1).T
    tetrahedra = []
    for order in permutations(np.eye(3, dtype=int)):
        offsets = np.vstack(([0, 0, 0], np.cumsum(order, axis=0)))
        corners = starts[:, None, :] + offsets[None, :, :]
        tetrahedra.append(indices[tuple(corners.transpose(2, 0, 1))])
    tets = np.vstack(tetrahedra)
    # Give all elements positive orientation in the physical coordinate frame.
    inverted = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) < 0.
    tets[inverted, :2] = tets[inverted, 1::-1]
    _write_volume(path, points, tets, np.full(len(tets), 2))


@pytest.fixture(name='sphere_path', scope='module')
def make_real_sphere(tmp_path_factory):
    """Keep one sphere mesh on disk for both compiled-binding tests."""
    pytest.importorskip('duneuropy', reason='requires the compiled DUNEuro Python binding')
    path = tmp_path_factory.mktemp('sarvas') / 'sphere.msh'
    _write_sphere(path)
    return path


def test_real_duneuro_spherical_meg(sphere_path):
    """Compare total FEM gain against Sarvas (1987), doi:10.1088/0031-9155/32/1/004."""
    mesh = _duneuro.read_charm_mesh(sphere_path, {2: 0.33})
    # Two readouts of the Sarvas solution give independent checks. With the
    # radial pickup, volume current has no radial component, so the absolute
    # gain for a tangential dipole equals the projected Biot–Savart field.
    # With the tangential pickup, a radial dipole is silent outside a sphere:
    # the volume-current field must cancel its large primary contribution.
    depth, radius = 0.034, 0.12
    positions = np.tile([0., 0., depth], (2, 1))
    moments = np.array([[0., 1., 0.], [0., 0., 1.]])
    coils = np.tile([radius, 0., 0.], (2, 1))
    normals = np.array([[1., 0., 0.], [0., 1., 0.]])
    gain = _duneuro.solve_forward(mesh, positions, moments, coils, normals, np.eye(2))
    distance = np.hypot(radius, depth)
    expected_radial = -1e-7 * depth / distance**3
    np.testing.assert_allclose(gain[0, 0], expected_radial, rtol=0.05, atol=0)
    # A coarse tetrahedral approximation leaves a small field. Normalize by
    # the known free-space amplitude so a reversed or missing volume term fails.
    primary_tangential = 1e-7 * radius / distance**3
    assert abs(gain[1, 1]) < 0.12 * primary_tangential


def test_real_layer_operator_matches_spherical_gain(tmp_path, sphere_path, monkeypatch):
    """Compute the analytic two-layer case directly from the recording, surfaces and FEM."""
    (tmp_path / 'example').mkdir()
    surfaces = LayerSurfaceSet('example', 2, subjects_dir=str(tmp_path))
    Path(surfaces.laminar_surf_dir).mkdir(parents=True)
    pial = np.array([[0., 0., 34.], [6., 0., 32.], [0., 0., 30.]])
    links = np.array([[0., 1., 0.], [0., -1., 0.], [0., 1., 0.]])
    white = pial + 2 * links
    face = np.array([[0, 1, 2]])
    layers = [_create_surf_gifti(xyz, face, links) for xyz in (pial, white)]
    for name, gifti in zip(surfaces.get_layer_names(), layers):
        nib.save(gifti, surfaces.get_mesh_path(name))
    nib.save(_concatenate_surfaces(layers), surfaces.get_mesh_path())

    # Both coils point radially out from the conductor. Their responses equal
    # the radial primary field of each signed dipole in this spherical geometry.
    radii = np.array([0.12, 0.14])
    coils = np.column_stack((radii, np.zeros((2, 2))))
    geometry = {'coilpos': coils, 'coilori': np.tile([1., 0., 0.], (2, 1)),
                'tra': np.array([[1., -0.25], [50., -50.]])}
    points = np.vstack((pial, white))
    moments = np.tile(links, (2, 1))
    # The sphere and sensors share one coordinate frame, so the MRI-to-head
    # transform is the identity. The header has no gain or other forward operator.
    recording = _recording_header(tmp_path, geometry, np.eye(4))
    prepared = compute_duneuro_forward(surfaces, sphere_path, reference=recording,
                                       inversion_idx=1, output_dir=tmp_path)
    distance = np.linalg.norm(coils[:, None] - points[None] / 1000., axis=2)
    signed_depth = moments[:, 1] * points[:, 2] / 1000.
    ideal_coils = -1e-7 * signed_depth[None, :] / distance**3
    expected = (geometry['tra'] @ ideal_coils) * np.array([[1e6], [1e4]])
    np.testing.assert_allclose(prepared.operator.gain, expected, rtol=0.05, atol=0)
    np.testing.assert_array_equal(prepared.support.original_vertices, [0, 1, 2])
    prepared.operator.validate_mesh(None, prepared.surfaces.get_mesh_path())

    # Each channel uses both pickups, with positive and negative weights.
    # Splitting the coils across two independent drivers must preserve every
    # dipole's gain, including the compensation applied across the two groups.
    def unexpected_local_driver(*_args):
        pytest.fail('The parallel solve loaded the native binding in the parent process.')

    # Screening can query native dipole statistics in the parent. The expensive
    # transfer solves and their native MEEG drivers still belong to the workers.
    dune = _duneuro._bindings()  # pylint: disable=protected-access
    monkeypatch.setattr(dune, 'MEEGDriver3d', unexpected_local_driver)
    parallel_dir = tmp_path / 'parallel'
    parallel = compute_duneuro_forward(surfaces, sphere_path, reference=recording,
                                       inversion_idx=1, output_dir=parallel_dir, n_jobs=2)
    np.testing.assert_allclose(parallel.operator.gain, prepared.operator.gain,
                               rtol=1e-6, atol=1e-10)
    # Temporary pickup fields and shared arrays are only needed during the solve.
    assert list(parallel_dir.iterdir()) == [Path(parallel.surfaces.laminar_surf_dir)]
