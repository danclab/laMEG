"""Map MNE source indices and project their gain onto signed laMEG orientations."""

from collections.abc import Mapping
from os import PathLike

import mne
import numpy as np
from mne.io.constants import FIFF

from ._mne_sensors import mne_channel_geometry


def _source_order(forward, vertex_map, n_vertices):
    """Locate each requested (source-space index, vertex ID) in the MNE gain."""
    keys = np.asarray(vertex_map)
    if keys.shape != (n_vertices, 2) or keys.dtype.kind not in 'iu':
        raise ValueError("vertex_map must contain an integer (space, vertex ID) for each vertex.")
    if len(np.unique(keys, axis=0)) != n_vertices:
        raise ValueError("Duplicate (source space, vertex ID) in vertex_map.")

    indices = {}
    offset = 0
    for space_index, space in enumerate(forward['src']):
        vertno = np.asarray(space['vertno'])
        if (vertno.ndim != 1 or vertno.dtype.kind not in 'iu' or np.any(vertno < 0)
                or len(np.unique(vertno)) != len(vertno)):
            raise ValueError("Invalid or duplicate MNE vertex IDs in source space.")
        if space['coord_frame'] != forward['coord_frame']:
            raise ValueError("MNE source-space frame differs from the gain coordinate frame.")
        positions = np.asarray(space['rr'])
        source_slice = slice(offset, offset + len(vertno))
        if (positions.ndim != 2 or positions.shape[1] != 3 or np.any(vertno >= len(positions))
                or not np.allclose(positions[vertno], forward['source_rr'][source_slice],
                                   atol=1e-7, rtol=0)):
            raise ValueError("MNE source_rr differs from source-space positions or vertex order.")
        indices.update(((space_index, vertex), offset + i) for i, vertex in enumerate(vertno))
        offset += len(vertno)
    if offset != forward['nsource']:
        raise ValueError("MNE source-space vertex count differs from gain.")
    try:
        return np.array([indices[tuple(key)] for key in keys])
    except KeyError as exc:
        raise ValueError(f"Requested source space and vertex ID {exc} missing from MNE forward.") \
            from exc


def _source_to_head_mm(forward, mne_head_to_head_mm):
    """Map positions in the forward's coordinate frame (mm) into laMEG head mm."""
    if forward['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
        return mne_head_to_head_mm
    if forward['coord_frame'] != FIFF.FIFFV_COORD_MRI:
        raise ValueError("MNE sources must use the head or MRI coordinate frame.")
    trans = forward['mri_head_t']
    if (trans['from'] != FIFF.FIFFV_COORD_MRI or trans['to'] != FIFF.FIFFV_COORD_HEAD):
        raise ValueError("MNE forward requires an MRI-to-head transform for MRI-frame sources.")
    affine = np.array(trans['trans'], dtype=float, copy=True)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(affine[:3, :3].T @ affine[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or np.linalg.det(affine[:3, :3]) <= 0):
        raise ValueError("MNE MRI-to-head transform must be rigid.")
    affine[:3, 3] *= 1000
    return mne_head_to_head_mm @ affine


def _projection_weights(forward, order, directions):
    """Express each target current direction in the saved gain-column basis."""
    components = {FIFF.FIFFV_MNE_FIXED_ORI: 1, FIFF.FIFFV_MNE_FREE_ORI: 3}.get(
        forward['source_ori'])
    if components is None:
        raise ValueError("Unknown MNE forward source orientation convention.")
    basis = np.asarray(forward['source_nn'])
    if (basis.shape != (forward['nsource'] * components, 3)
            or basis.dtype.kind not in 'iuf'):
        raise ValueError("MNE orientation basis differs from the declared source count.")
    basis = basis.reshape(-1, components, 3)[order]
    if not np.allclose(basis @ basis.transpose(0, 2, 1), np.eye(components), atol=1e-5, rtol=0):
        raise ValueError("MNE source orientation basis must be orthonormal.")
    weights = np.einsum('vij,vj->vi', basis, directions)
    if components == 1:
        # A fixed gain can change polarity, but cannot be rotated onto an oblique link vector.
        weights = np.sign(weights)
        if not np.allclose(weights * basis[:, 0], directions, atol=1e-5, rtol=0):
            raise ValueError("Fixed MNE orientation differs from the signed laMEG direction.")
    return weights


def project_sources(forward, vertex_map, requested_head_mm, orientations_head, mne_head_to_head_mm):
    """Project one MNE layer onto a laMEG mesh, retaining the actual solver locations."""
    positions = np.asarray(forward['source_rr'])
    if positions.shape != (forward['nsource'], 3):
        raise ValueError("MNE source_rr differs from the declared source count.")
    order = _source_order(forward, vertex_map, len(requested_head_mm))
    affine = _source_to_head_mm(forward, mne_head_to_head_mm)
    effective = (1000 * positions[order]) @ affine[:3, :3].T + affine[:3, 3]
    if not np.allclose(effective, requested_head_mm, atol=1e-3, rtol=0):
        raise ValueError("MNE source positions differ from the mapped laMEG vertices.")

    # Directions are row vectors. The inverse of a rigid rotation takes head directions
    # into the frame of source_nn; translations and position units have no effect on them.
    weights = _projection_weights(forward, order, orientations_head @ affine[:3, :3])
    gain = np.asarray(forward['sol']['data'])
    components = weights.shape[1]
    if (gain.ndim != 2 or gain.shape[1] != forward['nsource'] * components
            or gain.dtype.kind not in 'iuf'):
        raise ValueError("MNE gain shape differs from its declared sources and orientations.")
    blocks = gain.reshape(gain.shape[0], forward['nsource'], components)
    projected = np.empty((gain.shape[0], len(order)), dtype=np.result_type(gain, weights))
    # Gather source blocks in batches so reordering a full cortex never copies the full 3-D gain.
    for start in range(0, len(order), 512):
        batch = slice(start, start + 512)
        projected[:, batch] = np.einsum('cvk,vk->cv', blocks[:, order[batch], :], weights[batch])
    return projected, effective


def _same_sensors(first, second):
    # Allow FIFF float32 serialization while keeping the SPM attachment tolerances.
    tolerances = {'coilpos': 1e-7, 'coilori': 1e-7, 'tra': 1e-9}
    return all(first[k].shape == second[k].shape and np.allclose(
        first[k], second[k], atol=atol, rtol=1e-7) for k, atol in tolerances.items())


def load_forward(source, layer):
    """Read a single layer while retaining the gain's origin for provenance."""
    if isinstance(source, (str, PathLike)):
        return mne.read_forward_solution(str(source), verbose=False), str(source)
    if isinstance(source, mne.Forward):
        return source, source['info'].get('command_line') or 'in-memory Forward'
    raise TypeError(f"Expected an MNE Forward or FIF path for layer {layer!r}.")


def read_layers(forwards, vertices, names, units, head_mm, requested, orientations, info,
                ignore_ref, allow_nominal_ctf_coils):
    """Import and validate each layer, reading at most one FIF at a time."""
    head_m = head_mm.astype(float, copy=True)
    head_m[:3, 3] /= 1000
    n_columns = len(requested) // len(forwards)
    gain = np.empty((len(names), len(requested)), dtype=float)
    effective = np.empty(requested.shape, dtype=float)
    geometry = None
    descriptions = []
    for index, (name, source) in enumerate(forwards.items()):
        forward, description = load_forward(source, name)
        rows, scale, sensors = mne_channel_geometry(
            forward, names, units, head_m, info, ignore_ref, allow_nominal_ctf_coils)
        if geometry is None:
            geometry = sensors
        elif not _same_sensors(geometry, sensors):
            raise ValueError(f"MEG sensor geometry or compensation differs for layer {name!r}.")
        block = slice(index * n_columns, (index + 1) * n_columns)
        vertex_map = vertices[name] if isinstance(vertices, Mapping) else vertices
        projected, effective[block] = project_sources(
            forward, vertex_map, requested[block], orientations[block], head_mm)
        np.multiply(projected[rows], np.asarray(scale)[:, None], out=gain[:, block])
        descriptions.append(f"{name}: {description}")
        del forward, projected
    provenance = f"MNE-Python importer {mne.__version__}; " + "; ".join(descriptions)
    return gain, effective, geometry, provenance
