"""Prepare the sub-104 forwards and SPM recordings used in Tutorial 10."""
from copy import copy
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import mne
import nibabel as nib
import numpy as np
from mne.transforms import Transform, apply_trans
from scipy.io import savemat

from lameg.forward import ForwardOperator, prepare_mne_forward
from lameg.invert import coregister
from lameg.surf import LayerSurfaceSet, _concatenate_surfaces, _remove_vertices
from lameg.util import get_fiducial_coords, load_meg_sensor_data


def _occipital_patch(surfaces, directory):
    """Keep corresponding pial and white neighbourhoods around the first pial triangle."""
    layers = surfaces.get_layer_names()
    meshes = [nib.load(surfaces.get_mesh_path(layer_name=layer)) for layer in layers]
    faces = meshes[0].get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    np.testing.assert_array_equal(faces, meshes[1].get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data)
    # Grow complete rings on the pial mesh. Vertex IDs correspond across depths,
    # so the same mask selects the matching patch on the white surface.
    keep = np.zeros(len(meshes[0].darrays[0].data), dtype=bool)
    keep[faces[0]] = True
    while keep.sum() < 350:
        previous_count = keep.sum()
        keep[faces[keep[faces].any(axis=1)]] = True
        if keep.sum() == previous_count:
            raise ValueError("The first triangle's connected component is too small.")
    directory.mkdir()
    patch = copy(surfaces)
    patch.laminar_surf_dir = str(directory)
    # Cropping also renumbers triangles and preserves each vertex's signed dipole direction.
    cropped = [_remove_vertices(mesh, np.flatnonzero(~keep)) for mesh in meshes]
    for layer, mesh in zip(layers, cropped):
        nib.save(mesh, patch.get_mesh_path(layer_name=layer))
    nib.save(_concatenate_surfaces(cropped), patch.get_mesh_path())
    return patch, np.flatnonzero(keep)


def _single_shell(source, directory, surfaces, fids, skull, spm):
    """Compute SPM's single-shell gain in a private copy of the recording."""
    directory.mkdir()
    # The SPM header refers to its .dat file by name. Keep them together.
    for path in (source, source.with_suffix('.dat')):
        shutil.copy2(str(path), str(directory / path.name))
    recording = directory / source.name
    coregister(fids, str(recording), surfaces, forward_model_type='Single Shell',
               iskull_fname=str(skull), viz=False, spm_instance=spm)
    # Coregistration supplies the geometry. Request the lead field explicitly, then
    # read its signed gain rows together with their sensor units and source directions.
    filename = str(recording).replace("'", "''")
    spm.spm_standalone('eval', f"D=spm_eeg_load('{filename}'); spm_eeg_lgainmat(D);", nargout=0)
    operator = ForwardOperator.from_spm(
        str(recording), surfaces.get_mesh_path(), surfaces.get_layer_names(), allow_undefined_gain=True)
    return recording, operator


def _subject_bem(surfaces, skull, info, mri_to_ctf_mm, directory):
    """Build a one-compartment MNE BEM with the same inner skull and MRI registration."""
    # GIfTI points are in native MRI mm. MNE's anatomical frame is FreeSurfer
    # surface RAS. Use their common voxel coordinates to obtain the transform.
    mri = Path(surfaces.mri_file)
    tkr = nib.load(str(mri.with_name('orig.mgz'))).header.get_vox2ras_tkr()
    native_to_ras = tkr @ np.linalg.inv(nib.load(str(mri)).affine)
    # Compose surface RAS → native MRI → CTF head → MNE head in mm. MNE
    # stores transforms in metres, so only the resulting translation changes scale.
    mne_to_ctf = np.linalg.inv(info['ctf_head_t']['trans'])
    mne_to_ctf[:3, 3] *= 1000
    ras_to_head = np.linalg.inv(mne_to_ctf) @ mri_to_ctf_mm @ np.linalg.inv(native_to_ras)
    ras_to_head[:3, 3] /= 1000
    trans = Transform('mri', 'head', ras_to_head)

    inner = nib.load(str(skull))
    points = inner.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = inner.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    bem_dir = directory / surfaces.subj_id / 'bem'
    bem_dir.mkdir(parents=True)
    mne.write_surface(str(bem_dir / 'inner_skull.surf'), apply_trans(native_to_ras, points), faces)
    model = mne.make_bem_model(surfaces.subj_id, ico=None, conductivity=(0.3,), subjects_dir=directory)
    bem = mne.make_bem_solution(model)
    forwards = {}
    for layer in surfaces.get_layer_names():
        mesh = nib.load(surfaces.get_mesh_path(layer_name=layer))
        positions = apply_trans(native_to_ras, mesh.darrays[0].data) / 1000
        # The z-vector satisfies the discrete-source API. MNE computes three gain
        # components per retained site. laMEG projects them onto the signed GIfTI direction.
        src = mne.setup_volume_source_space(
            pos={'rr': positions, 'nn': np.tile([0., 0., 1.], (len(positions), 1))})
        forwards[layer] = mne.make_forward_solution(
            info, trans, src, bem, meg=True, eeg=False, mindist=0., ignore_ref=False)
    return forwards


def prepare_forwards(data_root, work_dir, spm):
    """Return the original recording, fiducials, prepared BEM, shell gain, patch IDs and skull.

    Use a fresh working directory for each run. full_vertex_ids[j] maps original
    patch index j to the full downsampled cortex. prepared.support.original_vertices
    gives the patch indices retained in the gain at both depths.
    """
    data_root, work = Path(data_root).resolve(), Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    session = data_root / 'sub-104' / 'meg' / 'ses-01'
    source = session / 'spm' / 'spm-converted_autoreject-sub-104-ses-01-001-btn_trial-epo.mat'
    info = mne.io.read_info(str(session / 'mne' / 'autoreject-sub-104-ses-01-001-btn_trial-epo.fif'))
    fids = get_fiducial_coords('sub-104', str(data_root / 'participants.tsv'))
    original = LayerSurfaceSet('sub-104', 2, subjects_dir=str(data_root / 'fs'))
    patch, full_vertex_ids = _occipital_patch(original, work / 'patch')

    # SPM writes meshes beside its input MRI. Copy that MRI and its normalization
    # into the work directory. Both solvers read the same distributed inner skull.
    (work / 'mri').mkdir()
    mri = Path(original.mri_file)
    for name in ('orig.nii', 'orig.mgz', 'y_orig.nii', 'orig_seg8.mat'):
        shutil.copy2(str(mri.with_name(name)), str(work / 'mri' / name))
    patch.mri_file = str(work / 'mri' / 'orig.nii')
    skull = mri.with_name('origiskull_2562.surf.gii')
    reference, native = _single_shell(source, work / 'registration', patch, fids, skull, spm)
    forwards = _subject_bem(patch, skull, info, native.mri_to_head_mm, work / 'bem')
    prepared = prepare_mne_forward(forwards, patch, work, reference=reference, info=info,
                                   ignore_ref=False, allow_nominal_ctf_coils=True)
    # MNE excludes some pial vertices. Recompute the single-shell gain on the mesh
    # shared by both depths so anatomical locations and gain columns line up.
    _, shell = _single_shell(source, work / 'single-shell', prepared.surfaces, fids, skull, spm)
    bem = prepared.operator
    if (shell.layer_names != bem.layer_names or shell.channel_names != bem.channel_names
            or shell.channel_units != bem.channel_units):
        raise ValueError("Both forwards must use identical layers, channels and physical units.")
    np.testing.assert_array_equal(shell.requested_mri_mm, bem.requested_mri_mm)
    np.testing.assert_allclose(shell.orientations_head, bem.orientations_head, atol=1e-5)
    return source, fids, prepared, shell, full_vertex_ids, skull


def source_indices(full_vertex, full_vertex_ids, prepared):
    """Convert a full-cortex ID to an MSP prior and a within-layer gain column."""
    # The cortical patch uses consecutive IDs. MSP takes the original ID within this
    # patch, including positions that the BEM may later exclude.
    patch = np.flatnonzero(full_vertex_ids == full_vertex)
    if not patch.size:
        raise ValueError(f"Full cortical vertex {full_vertex} is outside the tutorial patch.")
    patch_prior = int(patch[0])
    # Compact gain columns skip excluded sites. Both layers share the same map, so
    # this column identifies the same cortical site in each solver's single-layer gain.
    support = prepared.support
    gain = np.flatnonzero(support.original_vertices == patch_prior)
    if not gain.size:
        layers = [layer for layer, omissions in support.excluded_by_layer.items()
                  if patch_prior in omissions]
        raise ValueError(f"Full cortical vertex {full_vertex} was excluded by MNE "
                         f"on the {', '.join(layers)} surface.")
    return patch_prior, int(gain[0])


def write_evoked(source, destination, channel_names, sensor_data, spm):
    """Write one channels × time simulation into a new SPM recording with sub-104 metadata."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.with_suffix('.dat').exists():
        raise FileExistsError(f"A recording already exists at {destination}")
    # Clone with SPM so the simulated dataset inherits the original sensors, units,
    # fiducials and timing. Look up channels by label before writing the samples.
    with TemporaryDirectory(dir=str(destination.parent)) as temporary:
        inputs = Path(temporary) / 'inputs.mat'
        savemat(str(inputs), {'source': str(Path(source).resolve()),
                              'destination': str(destination.with_suffix('.dat')),
                              'names': np.asarray(channel_names, dtype=object),
                              'y': np.asarray(sensor_data, dtype=np.float32)})
        filename = str(inputs).replace("'", "''")
        command = f"cfg=load('{filename}');" + """
        D = spm_eeg_load(cfg.source);
        assert(montage(D,'getindex') == 0, 'Expected the original sub-104 montage.');
        S = clone(D, cfg.destination, [D.nchannels D.nsamples 1], 2, 1);
        good = S.indchantype({'MEG','MEGPLANAR'},'GOOD');
        [present,rows] = ismember(S.chanlabels(good), cfg.names);
        assert(all(present) && numel(unique(rows)) == numel(cfg.names));
        assert(size(cfg.y,2) == S.nsamples);
        S(good,:,1) = cfg.y(rows,:);
        S = conditions(S,1,'Simulated'); S = type(S,'evoked'); save(S);
        """
        spm.spm_standalone('eval', command, nargout=0)
    if not destination.is_file():
        raise RuntimeError("SPM did not create the simulated evoked dataset.")
    # Read back the float32 samples to verify that both inversions receive the
    # simulated recording in the intended channel order.
    stored, _, labels = load_meg_sensor_data(str(destination))
    rows = [list(channel_names).index(label) for label in labels]
    np.testing.assert_array_equal(stored.reshape(len(labels), -1),
                                  np.asarray(sensor_data, np.float32)[rows])
    return destination
