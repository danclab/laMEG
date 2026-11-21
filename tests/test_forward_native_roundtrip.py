"""End-to-end round trip using the same sub-104 data as the laMEG integration suite."""
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pytest

from lameg.forward import ForwardOperator
from lameg.invert import coregister
from lameg.laminar import model_comparison
from lameg.surf import LayerSurfaceSet, _concatenate_surfaces, _remove_vertices
from lameg.util import get_fiducial_coords


def separate_dataset(source, directory):
    """Preserve paired SPM filenames while giving each run independent gain caches."""
    directory.mkdir()
    for path in (source, source.with_suffix('.dat')):
        shutil.copy2(str(path), str(directory / path.name))
    return str(directory / source.name)


def compact_surfaces(data_root, destination):
    """Keep one connected patch of corresponding vertices from the two actual cortical layers."""
    surfaces = LayerSurfaceSet('sub-104', 2, subjects_dir=str(data_root / 'fs'))
    names = surfaces.get_layer_names()
    meshes = [nib.load(surfaces.get_mesh_path(layer_name=name)) for name in names]
    points, faces = (part.data for part in meshes[0].darrays[:2])
    keep = np.zeros(len(points), dtype=bool)
    keep[faces[0]] = True
    while np.count_nonzero(keep) < 350:
        previous_size = np.count_nonzero(keep)
        keep[faces[keep[faces].any(axis=1)]] = True
        assert np.count_nonzero(keep) > previous_size
    patches = [_remove_vertices(mesh, np.flatnonzero(~keep)) for mesh in meshes]
    destination.mkdir()
    surfaces.laminar_surf_dir = str(destination)
    for layer, mesh in zip(names, patches):
        nib.save(mesh, surfaces.get_mesh_path(layer_name=layer))
    nib.save(_concatenate_surfaces(patches), surfaces.get_mesh_path())
    return surfaces


def test_native_and_supplied_forward_give_same_layer_evidence(tmp_path, spm):
    """Native and reimported gains yield the same single-patch evidence per layer."""
    data_root = Path(__file__).resolve().parents[1] / 'test_data'
    source = (data_root / 'sub-104' / 'meg' / 'ses-01' / 'spm' /
              'spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat')
    if not source.is_file() or not source.with_suffix('.dat').is_file():
        pytest.skip('requires the laMEG sub-104 integration dataset')

    surfaces = compact_surfaces(data_root, tmp_path / 'surfaces')
    fids = get_fiducial_coords('sub-104', str(data_root / 'participants.tsv'))
    donor = separate_dataset(source, tmp_path / 'multilayer')
    native = separate_dataset(source, tmp_path / 'native')
    supplied = separate_dataset(source, tmp_path / 'supplied')
    coregister(fids, donor, surfaces, viz=False, spm_instance=spm)

    # Build the native full gain once. SPM stores the computed matrix beside the donor.
    donor_literal = donor.replace("'", "''")
    spm.spm_standalone('eval', f"D=spm_eeg_load('{donor_literal}'); spm_eeg_lgainmat(D);",
                       nargout=0)
    operator = ForwardOperator.from_spm(
        donor, surfaces.get_mesh_path(), surfaces.get_layer_names(), allow_undefined_gain=True
    )
    layers = [operator.layer(name) for name in surfaces.get_layer_names()]
    valid = np.logical_and.reduce([layer.orientation_defined for layer in layers])
    # Use one defined, nonzero patch in both layers.
    gain_norm = np.minimum.reduce([np.linalg.norm(layer.gain, axis=0) for layer in layers])
    candidate = np.where(valid, gain_norm, 0.)
    assert candidate.max() > 0
    prior = int(np.argmax(candidate))
    kwargs = {'method': 'MSP', 'viz': False, 'spm_instance': spm,
              'invert_kwargs': {'woi': [100., 200.], 'priors': [prior], 'n_temp_modes': 4}}
    native_f, native_cv = model_comparison(fids, native, surfaces, **kwargs)
    supplied_f, supplied_cv = model_comparison(fids, supplied, surfaces,
                                               forward_operator=operator, **kwargs)
    assert native_f.size == supplied_f.size == 2
    assert np.all(np.isfinite(native_f))
    assert abs(native_f.ravel()[1] - native_f.ravel()[0]) > 1e-4
    # Compare absolute per-layer log evidences and layer contrast in log-evidence units.
    np.testing.assert_allclose(supplied_f, native_f, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.diff(supplied_f, axis=0), np.diff(native_f, axis=0),
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(supplied_cv, native_cv, rtol=1e-7, atol=1e-9)
