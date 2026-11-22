"""Fixed-orientation source identity, physical units and SPM cache import."""
from dataclasses import replace
import hashlib

import h5py
import nibabel as nib
import numpy as np
import pytest
from scipy.io import savemat

from lameg.forward import ForwardOperator


def _mesh(path, points, faces, normals):
    arrays = [nib.gifti.GiftiDataArray(data, intent=intent) for data, intent in (
        (np.asarray(points, np.float32), 'NIFTI_INTENT_POINTSET'),
        (np.asarray(faces, np.int32), 'NIFTI_INTENT_TRIANGLE'),
        (np.asarray(normals, np.float32), 'NIFTI_INTENT_VECTOR'))]
    nib.save(nib.gifti.GiftiImage(darrays=arrays), str(path))


@pytest.fixture(name="operator")
def example_operator():
    """A two-layer example with matched vertices and known registration."""
    xyz = np.array([[0, 0, 2], [0, 1, 2], [1, 0, 2],
                    [0, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    affine = np.array([[0, -1, 0, 10], [1, 0, 0, 2], [0, 0, 1, 1], [0, 0, 0, 1]], float)
    used = xyz @ affine[:3, :3].T + affine[:3, 3]
    # Move only the solver positions for the three pial dipoles. Their anatomical vertex
    # IDs still refer to the points in xyz throughout import and layer selection.
    used[:3, 2] -= 0.6
    return ForwardOperator(np.arange(12.0).reshape(2, 6), ('L', 'R'), ('fT', 'fT'),
                           ('pial', 'white'), xyz, used, np.tile([0, 1, 0], (6, 1)), affine,
                           np.array([[0, 1, 2]]), {
                               'coilpos': np.eye(3), 'coilori': np.eye(3),
                               'tra': np.array([[1, 0, -1], [0, 1, -1]])}, 'test fixture')


def test_source_identity(tmp_path, operator):
    """Matched layers require the original topology, geometry and signed directions."""
    white = operator.layer('white')
    np.testing.assert_array_equal(white.gain, operator.gain[:, 3:])
    mesh_file = tmp_path / 'white.gii'
    # The fixture rotates native +x to head +y. Reversing the MRI vector changes
    # the sign of the predicted signal for the same positive current.
    directions = np.tile([1, 0, 0], (3, 1))
    _mesh(mesh_file, white.requested_mri_mm, white.faces, directions)
    operator.validate_mesh('white', mesh_file)
    with pytest.raises(ValueError, match='positions'):
        operator.validate_mesh('pial', mesh_file)
    _mesh(mesh_file, white.requested_mri_mm, white.faces, -directions)
    with pytest.raises(ValueError, match='orientations'):
        white.validate_mesh(None, mesh_file)
    _mesh(mesh_file, white.requested_mri_mm, white.faces[:, ::-1], directions)
    with pytest.raises(ValueError, match='topology'):
        white.validate_mesh(None, mesh_file)


def test_invalid_operator(operator):
    """Reject ambiguous sensor metadata, source layout and unsigned gains."""
    with pytest.raises(ValueError, match='duplicate'):
        replace(operator, channel_names=('L', 'L'))
    with pytest.raises(ValueError, match='physical'):
        replace(operator, channel_units=('fT', 'unknown'))
    with pytest.raises(ValueError, match='tra'):
        replace(operator, sensor_geometry=dict(operator.sensor_geometry, tra=np.ones((2, 4))))
    with pytest.raises(ValueError, match='equal-sized'):
        replace(operator, layer_names=('pial', '0.66', '0.33', 'white'))
    normals = operator.orientations_head.copy()
    normals[0] = 0
    with pytest.raises(ValueError, match='undefined'):
        replace(operator, orientations_head=normals)
    native = replace(operator, orientations_head=normals, allow_undefined_gain=True)
    assert not native.layer('pial').orientation_defined[0]
    np.testing.assert_array_equal(native.gain, operator.gain)


def _char(group, name, text):
    return group.create_dataset(name, data=np.array(list(map(ord, text)), 'uint16')[:, None])

def _strings(group, name, vals):
    dataset = group.create_dataset(name, (len(vals), 1), dtype=h5py.ref_dtype)
    for i, text in enumerate(vals):
        dataset[i, 0] = _char(group.file, f'{name}_{len(group.file)}_{i}', text).ref


@pytest.fixture(name="native_files")
def make_native_files(tmp_path, operator):
    """Represent MATLAB strings, cells and transposed v7.3 datasets explicitly."""
    cache = tmp_path / 'gain.mat'
    data = tmp_path / 'data.mat'
    mesh = tmp_path / 'multilayer.2.ds.gii'
    faces = np.r_[operator.faces, operator.faces + 3]
    _mesh(mesh, operator.requested_mri_mm, faces, np.tile([1, 0, 0], (6, 1)))

    # Reverse the gain rows, store the matrix in MATLAB axis order and multiply by
    # SPM's scale. A correct import must undo all three conventions independently.
    with h5py.File(cache, 'w') as header:
        _strings(header, 'label', ['R', 'L'])
        header['G'] = 10 * operator.gain[::-1].T
    with h5py.File(data, 'w') as header:
        dataset = header.create_group('D')
        channels = dataset.create_group('channels')
        _strings(channels, 'label', ['R', 'REF', 'L'])
        _strings(channels, 'units', ['fT', 'T', 'fT'])
        inv = header.create_group('entry')
        slots = dataset.create_group('other').create_dataset('inv', (1, 1), dtype=h5py.ref_dtype)
        slots[0, 0] = inv.ref
        _char(inv, 'gainmat', cache.name)
        reg = inv.create_group('datareg')
        _char(reg, 'modality', 'MEG')
        reg['fromMNI'] = operator.mri_to_head_mm.T
        mesh_group = inv.create_group('mesh')
        mesh_group['Affine'] = np.eye(4)
        _char(mesh_group, 'tess_ctx', mesh.name)
        fwd = inv.create_group('forward')
        _char(fwd, 'voltype', 'MEG OpenMEEG BEM')
        _char(fwd, 'modality', 'MEG')
        _strings(fwd, 'channels', ['L', 'R'])
        fwd['siunits'], fwd['scale'] = 1, 10.
        forward_mesh = fwd.create_group('mesh')
        # SPM saves one-based triangle indices and source positions in head metres.
        # The GIfTI uses zero-based connectivity and MRI millimetres.
        forward_mesh['face'] = (faces + 1).T
        forward_mesh['vert'] = operator.effective_head_mm.T / 1000.
        sens = fwd.create_group('sensors')
        _char(sens, 'unit', 'm')
        _strings(sens, 'label', ['R', 'L'])
        _strings(sens, 'chanunit', ['T', 'T'])
        for k in ('coilpos', 'coilori'):
            sens[k] = operator.sensor_geometry[k].T
        sens['tra'] = operator.sensor_geometry['tra'][::-1].T
    return data, mesh, cache


def test_spm_labels_units_scale(native_files, operator):
    """SPM can have different data, coil and gain label order."""
    data, mesh, _ = native_files
    imported = ForwardOperator.from_spm(data, mesh, operator.layer_names)
    np.testing.assert_array_equal(imported.gain, operator.gain)
    np.testing.assert_array_equal(imported.sensor_geometry['tra'], operator.sensor_geometry['tra'])
    np.testing.assert_array_equal(imported.orientations_head, operator.orientations_head)
    np.testing.assert_allclose(imported.effective_head_mm, operator.effective_head_mm)
    assert imported.channel_names == operator.channel_names
    assert imported.channel_units == operator.channel_units
    imported.validate_mesh(None, mesh)
    with pytest.raises(ValueError, match='inversion_idx'):
        ForwardOperator.from_spm(data, mesh, operator.layer_names, inversion_idx=1)

    with h5py.File(data, 'a') as header:
        del header['entry/forward/voltype']
        _char(header['entry/forward'], 'voltype', 'Single Shell')
    with pytest.raises(ValueError, match='uncorrected'):
        ForwardOperator.from_spm(data, mesh, operator.layer_names)



@pytest.mark.parametrize('nominal_ctf', [False, True])
def test_external_cache_reexport(native_files, operator, nominal_ctf):
    """Preserve a relocated external gain when its native template is single shell."""
    data, mesh, cache = native_files
    faces = np.r_[operator.faces, operator.faces + 3]
    with h5py.File(data, 'a') as header:
        del header['entry/forward/voltype']
        _char(header['entry/forward'], 'voltype', 'Single Shell')
        marker = header['entry'].create_group('lameg_external_gain')
        marker['format_version'] = 1
        _char(marker, 'file', cache.name)
        header['entry/forward/scale'][()] = 1.
    affine = operator.mri_to_head_mm
    registered = operator.requested_mri_mm @ affine[:3, :3].T + affine[:3, 3]
    fields = {
        'G': operator.gain[::-1], 'label': np.array([['R'], ['L']], dtype=object),
        'channel_units': np.array([['fT'], ['fT']], dtype=object),
        'format_version': 1, 'provenance': '"custom solver"',
        'requested_m': registered / 1000., 'positions_m': operator.effective_head_mm / 1000.,
        'normals': operator.orientations_head, 'coilpos': operator.sensor_geometry['coilpos'],
        'coilori': operator.sensor_geometry['coilori'],
        'tra': operator.sensor_geometry['tra'][::-1]}
    if nominal_ctf:
        fields['allow_nominal_ctf_coils'] = True
    savemat(str(cache), fields)
    with h5py.File(data, 'a') as header:
        _char(header['entry/lameg_external_gain'], 'sha256',
              hashlib.sha256(cache.read_bytes()).hexdigest())
    reexported = ForwardOperator.from_spm(data, mesh, operator.layer_names)
    np.testing.assert_array_equal(reexported.gain, operator.gain)
    np.testing.assert_array_equal(reexported.orientations_head, operator.orientations_head)
    np.testing.assert_allclose(reexported.effective_head_mm, operator.effective_head_mm)
    np.testing.assert_array_equal(reexported.sensor_geometry['tra'],
                                  operator.sensor_geometry['tra'])
    assert reexported.provenance == 'custom solver'
    assert reexported.allow_nominal_ctf_coils is nominal_ctf
    assert reexported.layer('white').allow_nominal_ctf_coils is nominal_ctf
    alternate = cache.parent / 'moved' / cache.name
    alternate.parent.mkdir()
    alternate.write_bytes(cache.read_bytes())
    moved = ForwardOperator.from_spm(data, mesh, operator.layer_names, gain_fname=alternate)
    np.testing.assert_array_equal(moved.gain, operator.gain)
    # Negating G preserves every dimension and RMS amplitude but changes the forward
    # prediction. The stored checksum detects that the moved cache was replaced.
    fields['G'] = -fields['G']
    savemat(str(alternate), fields)
    with pytest.raises(ValueError, match='checksum'):
        ForwardOperator.from_spm(data, mesh, operator.layer_names, gain_fname=alternate)
    _mesh(mesh, operator.requested_mri_mm + 1., faces, np.tile([1, 0, 0], (6, 1)))
    with pytest.raises(ValueError, match="requested_m"):
        ForwardOperator.from_spm(data, mesh, operator.layer_names)
