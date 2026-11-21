"""Verified common cortical vertices for comparisons with externally computed gains."""
from collections.abc import Mapping
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import TYPE_CHECKING
import warnings

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

from ._mne_forward import _source_to_head_mm, load_forward

if TYPE_CHECKING:
    from .forward import ForwardOperator
    from .surf import LayerSurfaceSet


@dataclass(frozen=True, eq=False)
class ForwardSupport:
    """Original within-layer vertex IDs and the omissions of each forward."""
    # Every layer uses the same original anatomical numbering. Element j of
    # original_vertices identifies the anatomical location of gain column j within a
    # prepared layer. excluded_by_layer records which forward omitted each location.
    original_n_columns: int
    original_vertices: np.ndarray
    excluded_by_layer: dict

    def map_indices(self, priors, multilayer=False):
        """Translate original zero-based anatomical IDs into prepared mesh indices."""
        original = np.asarray(priors)
        n_layers = len(self.excluded_by_layer) if multilayer else 1
        if original.dtype.kind not in 'iu':
            raise TypeError("Original vertex IDs must be integers.")
        if np.any(original < 0) or np.any(original >= n_layers * self.original_n_columns):
            raise ValueError("Requested vertex lies outside the original source space.")
        # The combined mesh concatenates layers. Divide by the original layer size
        # before filtering so a prior on a deeper layer retains both its depth and vertex.
        layers, local = np.divmod(original.astype(np.intp), self.original_n_columns)
        # Invert the compact-to-original map. The -1 sentinel lets us distinguish an
        # excluded seed from compact vertex 0, which is itself a valid prior.
        lookup = np.full(self.original_n_columns, -1, dtype=np.intp)
        lookup[self.original_vertices] = np.arange(len(self.original_vertices))
        compact = lookup[local]
        if np.any(compact < 0):
            missing = int(local.ravel()[np.flatnonzero(compact < 0)[0]])
            omitted = [k for k, ids in self.excluded_by_layer.items() if missing in ids]
            raise ValueError(f"Original cortical vertex {missing} is excluded from the forward "
                             f"for {', '.join(omitted)} and cannot be used as a prior.")
        # The inferred layer block now starts at a multiple of the retained count.
        return layers * len(self.original_vertices) + compact


@dataclass(frozen=True, eq=False)
class PreparedForward:
    """Common registered surfaces, imported operator and saved source-row identity."""
    surfaces: 'LayerSurfaceSet'
    operator: 'ForwardOperator'
    support: ForwardSupport
    manifest: Path


def map_surface_vertices(surf_set, indices, *, multilayer=False):
    """Map original zero-based indices on prepared surfaces; preserve ordinary mesh indices."""
    support = getattr(surf_set, '__dict__', {}).get('forward_support')
    if support is None:
        return indices
    return support.map_indices(indices, multilayer=multilayer)


def _full_source_ids(forward, points, head, explicit, layer):
    """Identify each GIfTI point in the full MNE source spaces, before vertno selection."""
    transform = _source_to_head_mm(forward, head)
    spaces = forward['src']
    if not spaces or any(s['coord_frame'] != forward['coord_frame'] for s in spaces):
        raise ValueError("Expected all MNE source spaces in the forward coordinate frame.")
    # rr contains the full coordinates in metres, including points MNE omitted from
    # the gain. Bring them into the same head frame and mm units as the GIfTI points.
    coords = np.concatenate([s['rr'] for s in spaces]) * 1000
    coords = coords @ transform[:3, :3].T + transform[:3, 3]
    # Left and right hemispheres can both have a vertex 42. Keep the starting index
    # of each source space to translate between flat search indices and (space, ID).
    offsets = np.r_[0, np.cumsum([len(s['rr']) for s in spaces])]
    if explicit is None:
        # At this tolerance (one micrometre) we expect the same physical point.
        # A second candidate within tolerance makes anatomical identity ambiguous.
        distances, indices = cKDTree(coords).query(points, k=2, distance_upper_bound=1e-3)
        missing = np.flatnonzero(~np.isfinite(distances[:, 0]))
        if missing.size:
            raise ValueError(f"{layer} vertex {missing[0]} is absent from the full MNE source "
                             "space. Check registration and use forwards made on these surfaces.")
        ambiguous = np.flatnonzero(np.isfinite(distances[:, 1]))
        if ambiguous.size:
            raise ValueError(f"Ambiguous MNE source positions for {layer} vertex {ambiguous[0]}; "
                             "supply explicit vertex IDs.")
        flat = indices[:, 0]
        which = np.searchsorted(offsets[1:], flat, side='right')
        keys = np.column_stack((which, flat - offsets[which]))
    else:
        keys = np.asarray(explicit)
        if (keys.shape != (len(points), 2) or keys.dtype.kind not in 'iu' or np.any(keys < 0)
                or np.any(keys[:, 0] >= len(spaces))):
            raise ValueError("Expected one integer (source space, vertex ID) for each GIfTI point.")
        # Even a known hemisphere/vertex ID could refer to a different mesh revision.
        # Check its registered position before accepting the supplied correspondence.
        which, vertex = keys.T
        if np.any(vertex >= np.diff(offsets)[which]):
            raise ValueError("Vertex ID outside the full MNE source space.")
        flat = offsets[which] + vertex.astype(np.intp)
        if np.any(np.linalg.norm(coords[flat] - points, axis=1) > 1e-3):
            raise ValueError("Explicit vertex IDs disagree with registered MNE positions.")
    if len(np.unique(flat)) != len(points):
        raise ValueError("Multiple GIfTI vertices map to the same MNE source.")
    # vertno selects the vertices that actually contributed gain columns. Return this
    # mask in original GIfTI order so masks from different depths line up anatomically.
    active = np.zeros(len(coords), dtype=bool)
    for i, space in enumerate(spaces):
        ids = np.asarray(space['vertno'])
        if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= np.diff(offsets)[i]):
            raise ValueError("Invalid active MNE vertex IDs.")
        active[offsets[i] + ids] = True
    return keys, active[flat]


def _subset_gifti(mesh, retained):
    """Filter oriented GIfTI geometry, retaining data-array and coordinate-system metadata."""
    vertices, = mesh.get_arrays_from_intent('NIFTI_INTENT_POINTSET')
    faces, = mesh.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')
    normals, = mesh.get_arrays_from_intent('NIFTI_INTENT_VECTOR')
    if len(mesh.darrays) != 3:
        raise ValueError("Expected a laMEG surface with position, triangle and direction arrays.")
    if normals.data.shape != vertices.data.shape or len(retained) > len(vertices.data):
        raise ValueError("Invalid laMEG surface geometry or signed directions.")
    triangles = faces.data
    if (triangles.dtype.kind not in 'iu' or triangles.size == 0 or triangles.min() < 0
            or triangles.max() >= len(vertices.data)):
        raise ValueError("Invalid GIfTI triangle indices.")
    # A face survives only if all three original corners survive. Reindex those
    # triangles in compact order, preserving their winding and the holes at exclusions.
    lookup = np.full(len(vertices.data), -1, dtype=np.intp)
    lookup[retained] = np.arange(len(retained))
    mapped = lookup[triangles]
    faces.data = mapped[(mapped >= 0).all(axis=1)].astype(triangles.dtype)
    if faces.data.size == 0:
        raise ValueError("No cortical triangles remain on the common MNE source support.")
    # Preserve the supplied signed dipole direction at each vertex. Recomputing normals
    # on the cropped mesh could rotate a dipole at the new boundary.
    vertices.data = vertices.data[retained]
    normals.data = normals.data[retained]
    # The GIfTI header stores dimensions separately from the array payload. Update
    # both so readers see the reduced shapes and can use the original coordinate metadata.
    for array in mesh.darrays:
        array.dims = list(array.data.shape)
    return mesh


def _match_layers(forwards, surfaces, vertices, mesh_options, options):
    """Match each original mesh to its MNE source space before intersecting the masks."""
    affine = options['mri_to_head_mm']
    meshes, maps, masks = {}, {}, {}
    topology = None
    for name, source in forwards.items():
        mesh = nib.load(surfaces.get_mesh_path(layer_name=name, **mesh_options))
        points, = mesh.get_arrays_from_intent('NIFTI_INTENT_POINTSET')
        points = points.data @ affine[:3, :3].T + affine[:3, 3]
        forward, _ = load_forward(source, name)
        explicit = vertices[name] if isinstance(vertices, Mapping) else vertices
        # Each original layer vertex gets both an MNE identity and an availability flag.
        # Record these separately because a known vertex can be absent from the gain.
        maps[name], masks[name] = _full_source_ids(
            forward, points, options['mne_head_to_head_mm'], explicit, name)
        faces, = mesh.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')
        # Common vertex IDs across depths assume corresponding faces. Equal vertex
        # counts alone would also admit a permuted lamina with different connectivity.
        if topology is None:
            topology = faces.data
        elif not np.array_equal(topology, faces.data):
            raise ValueError(f"Layer {name} has different original cortical vertex topology.")
        meshes[name] = mesh
        del forward
    return meshes, maps, masks


def prepare_support(forwards, surfaces, output_dir, vertices, mesh_options, options, importer):
    """Verify correspondence on original geometry before constructing the analysis meshes."""
    # Preparing an already cropped mesh would silently relabel compact vertex IDs
    # as original ones in the new map.
    if getattr(surfaces, '__dict__', {}).get('forward_support') is not None:
        raise ValueError("Pass the original surfaces, before common source support preparation.")
    layers = surfaces.get_layer_names()
    if not isinstance(forwards, Mapping) or set(forwards) != set(layers):
        raise ValueError("MNE forward layer names must match the surface set.")
    if isinstance(vertices, Mapping) and set(vertices) != set(layers):
        raise ValueError("Per-layer vertex mappings must match the surface set.")
    # Use the surface set's anatomical layer order when assembling the gain blocks.
    forwards = {name: forwards[name] for name in layers}
    meshes, maps, masks = _match_layers(forwards, surfaces, vertices, mesh_options, options)
    sizes = {len(mask) for mask in masks.values()}
    if len(sizes) != 1:
        raise ValueError("The original cortical layers have different numbers of vertices.")
    original_count = sizes.pop()
    # Take the intersection in anatomical vertex order. If the pial forward dropped
    # a location, remove its white and intermediate counterparts from the comparison.
    retained = np.flatnonzero(np.logical_and.reduce(list(masks.values())))
    if retained.size == 0:
        raise ValueError("The MNE forwards have no common cortical vertices.")
    # Record where each omission arose. A white vertex removed solely to match
    # its missing pial counterpart still counts as an omission of the pial forward.
    excluded = {name: np.flatnonzero(~mask).tolist() for name, mask in masks.items()}
    retained.setflags(write=False)
    support = ForwardSupport(original_count, retained, excluded)
    maps = {name: keys[retained] for name, keys in maps.items()}
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as rollback:
        # Keep each preparation and its later SPM smoothing files in their own directory.
        # If validation fails, remove only this attempt's files.
        work = Path(mkdtemp(prefix='mne-forward-', dir=str(directory)))
        rollback.callback(shutil.rmtree, work)
        prepared = copy(surfaces)
        prepared.laminar_surf_dir = str(work)
        prepared.forward_support = support
        for name, mesh in meshes.items():
            nib.save(_subset_gifti(mesh, retained),
                     prepared.get_mesh_path(layer_name=name, **mesh_options))
        # The combined GIfTI has separate vertices for each depth. Offset the same
        # retained IDs into each original block to reproduce the per-layer ordering.
        combined = nib.load(surfaces.get_mesh_path(**mesh_options))
        selected = np.concatenate([i * original_count + retained for i in range(len(layers))])
        target = prepared.get_mesh_path(**mesh_options)
        nib.save(_subset_gifti(combined, selected), target)
        operator = importer(forwards, target, vertices=maps, **options)
        # Validate the combined gain against each saved GIfTI, since layer model
        # comparisons register those individual files with SPM.
        for name in layers:
            operator.validate_mesh(name, prepared.get_mesh_path(layer_name=name, **mesh_options))
        # Save the compact-to-anatomical mapping beside the meshes so a saved source
        # row can still be located on the original cortex in a later analysis.
        manifest = work / 'source_support.json'
        report = {'subject': surfaces.subj_id, 'mesh_options': mesh_options, 'layer_names': layers,
                  'original_n_columns': original_count, 'original_vertices': retained.tolist(),
                  'excluded_by_layer': excluded}
        manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if len(retained) != original_count:
            counts = ', '.join(f"{name}: {len(ids)}" for name, ids in excluded.items() if ids)
            warnings.warn(f"Using {len(retained)} of {original_count} original cortical vertices "
                          f"at every depth. Vertices omitted by MNE ({counts}). See {manifest}.",
                          UserWarning, stacklevel=3)
        rollback.pop_all()
    return PreparedForward(prepared, operator, support, manifest)
