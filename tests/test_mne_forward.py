"""Check imports against MNE's prediction for a signed patch on a two-layer BEM."""
import mne
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np
import pytest

from lameg.forward import ForwardOperator
from tests.test_forward import _mesh


def _bem_and_recording(subjects_dir):
    """Build a closed triangular conductor and an uncompensated MEG recording."""
    bem_dir = subjects_dir / 'example' / 'bem'
    bem_dir.mkdir(parents=True)
    poles = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    triangles = [[j, (j + 1) % 4, 4] for j in range(4)]
    triangles += [[(j + 1) % 4, j, 5] for j in range(4)]
    mne.write_surface(str(bem_dir / 'inner_skull.surf'), 90. * poles, np.array(triangles))
    model = mne.make_bem_model('example', ico=None, conductivity=(0.3,),
                               subjects_dir=str(subjects_dir), verbose=False)
    bem = mne.make_bem_solution(model, verbose=False)
    info = mne.create_info(['MEG0111', 'MEG0112'], 1000., ['mag', 'grad'])
    for channel, position in zip(info['chs'], ([0.025, 0.012, 0.11], [-0.02, 0.01, 0.11])):
        channel['loc'] = np.r_[position, np.eye(3).ravel()]
    return bem, info


@pytest.fixture(name='mne_example', scope='module')
def make_mne_example(tmp_path_factory):
    """Compute two independent BEM gains on corresponding left/right tetrahedra."""
    root = tmp_path_factory.mktemp('mne-bem')
    bem, info = _bem_and_recording(root)
    surf_dir = root / 'example' / 'surf'
    surf_dir.mkdir()
    corners = np.array([[1., 1., 1.], [-1., -1., 1.], [-1., 1., -1.], [1., -1., -1.]])
    corners *= [1., 1.4, 0.75]
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    # Native MRI -> MNE head (metres), then MNE head -> recipient head (mm).
    mri_head = np.array([[0., -1., 0., 0.003], [1., 0., 0., -0.001],
                         [0., 0., 1., 0.002], [0., 0., 0., 1.]])
    head_head = np.array([[1., 0., 0., 5.], [0., 0., -1., 7.],
                          [0., 1., 0., -2.], [0., 0., 0., 1.]])
    mri_head_mm = mri_head.copy()
    mri_head_mm[:3, 3] *= 1000.

    model_points, forwards = [], {}
    for layer, radius in [('pial', 5.), ('white', 3.)]:
        hemispheres = []
        for side, centre in [('lh', [-24., 5., 3.]), ('rh', [24., 5., 3.])]:
            points = centre + radius * corners
            hemispheres.append(points)
            mne.write_surface(str(surf_dir / f'{side}.{layer}'), points, faces)
        model_points.append(np.concatenate(hemispheres))
        src = mne.setup_source_space('example', spacing='all', surface=layer, add_dist=False,
                                     subjects_dir=str(root), verbose=False)
        forwards[layer] = mne.make_forward_solution(
            info, mne.transforms.Transform('mri', 'head', mri_head), src, bem,
            eeg=False, mindist=0., verbose=False)

    # Reorder both hemispheres and within-hemisphere vertices in the laMEG mesh.
    order = np.array([5, 4, 7, 6, 2, 3, 0, 1])
    directions = np.tile(corners / np.linalg.norm(corners, axis=1, keepdims=True), (2, 1))
    local_faces = np.argsort(order)[np.r_[faces, faces + 4]]
    mesh = root / 'multilayer.gii'
    _mesh(mesh, np.concatenate([points[order] for points in model_points]),
          np.r_[local_faces, local_faces + 8], np.tile(directions[order], (2, 1)))
    options = {'vertices': np.column_stack((order // 4, order % 4)),
               'channel_units': {'MEG0112': 'fT/cm', 'MEG0111': 'fT'},
               'mri_to_head_mm': head_head @ mri_head_mm, 'mne_head_to_head_mm': head_head}
    return forwards, mesh, options, order, directions @ mri_head[:3, :3].T, info


def test_mne_signed_bem_prediction(mne_example, tmp_path):
    """An independent vector-source projection reproduces each signed sensor in physical units."""
    forwards, mesh, options, order, mne_directions, info = mne_example
    # Exercise FIF and the two MNE free-orientation bases in the same import.
    pial_fif = tmp_path / 'pial-fwd.fif'
    mne.write_forward_solution(str(pial_fif), forwards['pial'], verbose=False)
    white = mne.convert_forward_solution(forwards['white'], surf_ori=True, use_cps=False,
                                         verbose=False)
    operator = ForwardOperator.from_mne({'pial': pial_fif, 'white': white}, mesh, **options)
    assert operator.channel_names == ('MEG0112', 'MEG0111')
    assert operator.channel_units == ('fT/cm', 'fT')
    assert operator.layer_names == ('pial', 'white')

    # The separate pulses identify each source. The final sample drives a bipolar patch.
    weights = np.array([1., -2., 3., -4., -5., 6., -7., 8.])
    moments = np.column_stack((np.diag(weights), weights))  # nA m, in MNE vertex order
    vector_am = mne_directions[:, :, None] * moments[:, None, :] * 1e-9
    currents = mne.VectorSourceEstimate(vector_am, [np.arange(4), np.arange(4)], 0., 0.001)
    for name, forward in forwards.items():
        expected = mne.apply_forward(forward, currents, info, verbose=False)
        _assert_sensor_prediction(operator.layer(name), moments[order], expected)


def _assert_sensor_prediction(operator, moments, expected):
    # MNE outputs T/m for the planar gradiometer and T for the magnetometer.
    converted = expected.data[[1, 0]] * np.array([[1e13], [1e15]])
    assert np.all(np.linalg.norm(converted, axis=1) > 0.)
    np.testing.assert_allclose(operator.gain @ moments, converted, rtol=2e-6, atol=1e-8)


def test_fixed_bem_opposite_directions(mne_example, tmp_path):
    """Flipping the sign of a fixed dipole flips the corresponding gain column."""
    forwards, mesh, options, order, _, info = mne_example
    fixed = mne.convert_forward_solution(forwards['pial'], force_fixed=True, use_cps=False,
                                         verbose=False)
    points, faces = (part.data for part in nib.load(str(mesh)).darrays[:2])
    signs = np.array([1., -1., 1., -1., -1., 1., -1., 1.])
    signed_head = fixed['source_nn'] * signs[:, None]
    native = signed_head @ fixed['mri_head_t']['trans'][:3, :3]
    pial_mesh = tmp_path / 'signed-pial.gii'
    _mesh(pial_mesh, points[:8], faces[np.all(faces < 8, axis=1)], native[order])
    operator = ForwardOperator.from_mne({'pial': fixed}, pial_mesh, **options)
    moments = np.arange(1., 9.)[:, None] * np.array([[-1., 2.]])
    currents = mne.SourceEstimate(moments * signs[:, None] * 1e-9,
                                  [np.arange(4), np.arange(4)], 0., 0.001)
    expected = mne.apply_forward(fixed, currents, info, verbose=False)
    _assert_sensor_prediction(operator, moments[order], expected)


def test_mri_coordinate_forward(mne_example):
    """The same Cartesian gain expressed in MRI or head axes gives the same imported operator."""
    forwards, mesh, options = mne_example[:3]
    head = forwards['pial']
    native = head.copy()
    native['coord_frame'] = FIFF.FIFFV_COORD_MRI
    for space in native['src']:
        mne.transforms.transform_surface_to(space, 'mri', native['mri_head_t'])
    native['source_rr'] = np.vstack([space['rr'][space['vertno']] for space in native['src']])
    gain = head['sol']['data']
    rotation = head['mri_head_t']['trans'][:3, :3]
    native['sol']['data'] = (gain.reshape(len(gain), -1, 3) @ rotation).reshape(gain.shape)
    reference = ForwardOperator.from_mne(forwards, mesh, **options)
    converted = ForwardOperator.from_mne(dict(forwards, pial=native), mesh, **options)
    np.testing.assert_allclose(converted.effective_head_mm, reference.effective_head_mm)
    np.testing.assert_allclose(converted.gain, reference.gain, atol=1e-8)


@pytest.mark.parametrize(('problem', 'message'), [
    ('duplicate_vertex', 'Duplicate'), ('missing_vertex', 'missing from MNE'),
    ('registration', 'source positions'), ('gradient_unit', 'dimension'),
    ('fixed_orientation', 'Fixed MNE orientation')])
def test_mne_rejects_ambiguous_inputs(mne_example, problem, message):
    """Incompatible units, geometry, vertex identities or fixed directions fail before inversion."""
    forwards, mesh, options = mne_example[:3]
    options = options.copy()
    if problem in ('duplicate_vertex', 'missing_vertex'):
        vertices = options['vertices'].copy()
        vertices[0] = vertices[1] if problem == 'duplicate_vertex' else [0, 100]
        options['vertices'] = vertices
    elif problem == 'registration':
        transform = options['mri_to_head_mm'].copy()
        transform[0, 3] += 1.
        options['mri_to_head_mm'] = transform
    elif problem == 'gradient_unit':
        options['channel_units'] = dict(options['channel_units'], MEG0112='fT')
    else:
        fixed = mne.convert_forward_solution(forwards['pial'], force_fixed=True,
                                             use_cps=False, verbose=False)
        forwards = dict(forwards, pial=fixed)
    with pytest.raises(ValueError, match=message):
        ForwardOperator.from_mne(forwards, mesh, **options)
