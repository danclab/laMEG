"""Shared compact surfaces and registration for native and MNE forward comparisons."""
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pytest

from lameg.forward import ForwardOperator
from lameg.invert import coregister
from lameg.surf import LayerSurfaceSet, _concatenate_surfaces, _remove_vertices
from lameg.util import get_fiducial_coords


def separate_dataset(source, directory):
    """Preserve paired SPM filenames while giving each run independent gain caches."""
    # The .mat header locates the .dat samples by filename. Copy the pair together so
    # SPM can update a run's inversion slot and adjacent gain files in its own directory.
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
    # Seed a triangle, then grow one ring of adjoining triangles at a time. This gives
    # patch smoothing a connected neighborhood while keeping the compiled tests small.
    keep[faces[0]] = True
    while np.count_nonzero(keep) < 350:
        previous_size = np.count_nonzero(keep)
        keep[faces[keep[faces].any(axis=1)]] = True
        assert np.count_nonzero(keep) > previous_size
    # The two surfaces have matching vertex IDs. Filtering both with the same mask
    # preserves the pial/white column correspondence when the remaining IDs are renumbered.
    patches = [_remove_vertices(mesh, np.flatnonzero(~keep)) for mesh in meshes]
    destination.mkdir()
    surfaces.laminar_surf_dir = str(destination)
    for layer, mesh in zip(names, patches):
        nib.save(mesh, surfaces.get_mesh_path(layer_name=layer))
    # Layer comparisons read individual meshes. Multilayer inversion reads these same
    # vertices concatenated in layer order, so both routes exercise the same anatomy.
    nib.save(_concatenate_surfaces(patches), surfaces.get_mesh_path())
    return surfaces


def prepare_registration(tmp_path, spm):
    """Compute the native gain and registration once for the compiled attachment tests."""
    data_root = Path(__file__).resolve().parents[1] / 'test_data'
    source = (data_root / 'sub-104' / 'meg' / 'ses-01' / 'spm' /
              'spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat')
    if not source.is_file() or not source.with_suffix('.dat').is_file():
        pytest.skip('requires the laMEG sub-104 integration dataset')

    surfaces = compact_surfaces(data_root, tmp_path / 'surfaces')
    fids = get_fiducial_coords('sub-104', str(data_root / 'participants.tsv'))
    donor = separate_dataset(source, tmp_path / 'multilayer')
    coregister(fids, donor, surfaces, viz=False, spm_instance=spm)

    # Coregistration establishes the source and sensor frames. Calling SPM's gain reader
    # triggers the forward solve and creates the native gain cache alongside the donor.
    # allow_undefined_gain retains archived columns whose mesh directions are missing.
    # Tests that need a signed seed select it with operator.orientation_defined.
    donor_literal = donor.replace("'", "''")
    spm.spm_standalone('eval', f"D=spm_eeg_load('{donor_literal}'); spm_eeg_lgainmat(D);",
                       nargout=0)
    operator = ForwardOperator.from_spm(
        donor, surfaces.get_mesh_path(), surfaces.get_layer_names(), allow_undefined_gain=True
    )
    return source, fids, surfaces, donor, operator
