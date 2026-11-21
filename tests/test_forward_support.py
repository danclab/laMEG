"""Common anatomical support for precomputed MNE forward operators."""
import json
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pytest

from lameg.forward import ForwardOperator, prepare_mne_forward
from tests.test_forward import _mesh
from tests.test_mne_forward import make_mne_example  # pylint: disable=unused-import


class Surfaces:
    """A minimal LayerSurfaceSet with the same naming behavior as the public workflow."""
    def __init__(self, path):
        self.laminar_surf_dir = str(path)
        self.subj_id = "test"

    @staticmethod
    def get_layer_names():
        """Layer order of this two-depth fixture."""
        return ['pial', 'white']

    def get_mesh_path(self, layer_name=None, **_mesh_options):
        """Resolve a requested mesh within the current surface directory."""
        return str(Path(self.laminar_surf_dir) / f"{layer_name or 'multilayer'}.gii")


@pytest.fixture(name='support_data')
def _support_data(mne_example, tmp_path):
    forwards, full_path, options = mne_example[:3]
    surfaces = Surfaces(tmp_path)
    combined = nib.load(str(full_path))
    nib.save(combined, surfaces.get_mesh_path())
    xyz, faces, normals = (a.data for a in combined.darrays)
    # This fixture has eight vertices per layer in a deliberately shuffled GIfTI
    # order. Save both per-layer surfaces for the same registration checks SPM uses.
    for index, key in enumerate(surfaces.get_layer_names()):
        selection = slice(8 * index, 8 * (index + 1))
        layer_faces = faces[faces[:, 0] // 8 == index] - 8 * index
        _mesh(surfaces.get_mesh_path(key), xyz[selection], layer_faces, normals[selection])
    # Keep vertex 0 in different hemispheres at the two depths. Both gains have
    # seven sources, but only six anatomical locations are available at both depths.
    restricted = {}
    for key, spaces in [('white', [np.arange(4), np.arange(1, 4)]),
                        ('pial', [np.arange(1, 4), np.arange(4)])]:
        stc = mne.SourceEstimate(np.ones((7, 1)), spaces, 0., 0.01)
        restricted[key] = mne.forward.restrict_forward_to_stc(forwards[key], stc)
    return surfaces, restricted, options


def test_common_support_vertices(support_data, tmp_path):
    """Map reordered hemisphere/vertex IDs and independent omissions across both layers."""
    surfaces, restricted, options = support_data
    original = Path(surfaces.get_mesh_path()).read_bytes()
    # These known (hemisphere, vertex) pairs give an independent expected answer.
    # The first call receives only the forwards and registration to recover that map.
    keys = options['vertices']
    keep = np.flatnonzero(keys[:, 1] != 0)
    inputs = {k: v for k, v in options.items() if k != 'vertices'}
    with pytest.warns(UserWarning, match='Using 6 of 8'):
        prepared = prepare_mne_forward(restricted, surfaces, tmp_path, **inputs)
    assert prepared.operator.layer_names == ('pial', 'white')
    assert prepared.surfaces.forward_support is prepared.support
    assert Path(surfaces.get_mesh_path()).read_bytes() == original
    report = json.loads(prepared.manifest.read_text(encoding='utf-8'))
    assert report['original_vertices'] == keep.tolist()
    # The omissions are reported as original GIfTI IDs despite MNE's use of
    # local vertex 0 in each hemisphere.
    assert report['excluded_by_layer'] == {'pial': [6], 'white': [1]}
    np.testing.assert_array_equal(prepared.support.map_indices(keep), np.arange(6))
    # The second layer starts at index 8 on the original mesh and 6 on the
    # prepared mesh. Original index 14 therefore refers to excluded local vertex 6.
    np.testing.assert_array_equal(prepared.support.map_indices(8 + keep, multilayer=True),
                                  6 + np.arange(6))
    with pytest.raises(ValueError, match='vertex 6.*pial'):
        prepared.support.map_indices([14], multilayer=True)
    # Compare every gain column with a direct import using the known correspondences.
    # Agreement also checks that filtering kept the original signed source directions.
    native_order = ForwardOperator.from_mne(
        {k: restricted[k] for k in ('pial', 'white')}, prepared.surfaces.get_mesh_path(),
        vertices=keys[keep], **inputs)
    np.testing.assert_array_equal(prepared.operator.gain, native_order.gain)
    for layer in prepared.operator.layer_names:
        prepared.operator.validate_mesh(layer, prepared.surfaces.get_mesh_path(layer))
    with pytest.warns(UserWarning, match='Using 6 of 8'):
        repeated = prepare_mne_forward(restricted, surfaces, tmp_path,
                                       **dict(options, vertices=keys.astype(np.uint64)))
    assert repeated.manifest != prepared.manifest
    np.testing.assert_array_equal(repeated.operator.gain, prepared.operator.gain)
    with pytest.raises(ValueError, match='original surfaces'):
        prepare_mne_forward(restricted, prepared.surfaces, tmp_path, **inputs)



def test_unknown_and_ambiguous_grid(support_data, tmp_path):
    """Registration errors and ambiguous coordinates must not appear as solver omissions."""
    surfaces, restricted, options = support_data
    options = {k: v for k, v in options.items() if k != 'vertices'}
    fwd = restricted['pial'].copy()
    # Coincident points in the full source grid, even when only one of them is active.
    space = fwd['src'][0]
    space['rr'] = np.vstack((space['rr'], space['rr'][0]))
    restricted = dict(restricted, pial=fwd)
    destination = tmp_path / 'output'
    with pytest.raises(ValueError, match='Ambiguous'):
        prepare_mne_forward(restricted, surfaces, destination, **options)
    # A 1 mm translation moves these mesh points away from the full MNE source
    # grid. Treating them as BEM exclusions would hide a registration error.
    transform = options['mri_to_head_mm'].copy()
    transform[0, 3] += 1.
    with pytest.raises(ValueError, match='absent from the full MNE source'):
        prepare_mne_forward(restricted, surfaces, destination,
                            **dict(options, mri_to_head_mm=transform))
    assert not destination.exists()
