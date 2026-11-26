"""Verified common cortical vertices for comparisons with externally computed gains."""
from collections.abc import Mapping
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass, field, replace
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
            reason = (f"for {', '.join(omitted)}" if omitted
                      else "because no cortical face survives there")
            raise ValueError(f"Original cortical vertex {missing} is excluded from the forward "
                             f"{reason} and cannot be used as a prior.")
        # The inferred layer block now starts at a multiple of the retained count.
        return layers * len(self.original_vertices) + compact


@dataclass(frozen=True, eq=False)
class PreparedForward:
    """Common registered surfaces, imported operator and saved source-row identity."""
    surfaces: 'LayerSurfaceSet'
    operator: 'ForwardOperator'
    support: ForwardSupport
    manifest: Path
    _input_surfaces: 'LayerSurfaceSet' = field(default=None, repr=False)
    _mesh_options: dict = field(default_factory=dict, repr=False)

    def restrict_operator(self, operator):
        """Select these exact cortical columns from another gain on the input surfaces.

        This is useful after FEM tissue checks exclude some of a BEM's vertices.
        A solver keeps its own effective positions, directions, units and coils.
        """
        if self._input_surfaces is None or operator.layer_names != self.operator.layer_names:
            raise ValueError("Expected a forward defined on the preparation's input layers.")
        operator.validate_mesh(None, self._input_surfaces.get_mesh_path(**self._mesh_options))
        # The support IDs always index the original anatomy. The input gain may
        # already have been compacted by an earlier preparation (e.g. MNE).
        keep = map_surface_vertices(self._input_surfaces, self.support.original_vertices)
        if len(keep) == operator.n_columns:
            return operator
        columns = np.concatenate([i * operator.n_columns + keep
                                  for i in range(len(operator.layer_names))])
        return replace(operator, gain=operator.gain[:, columns], faces=self.operator.faces,
                       requested_mri_mm=operator.requested_mri_mm[columns],
                       effective_head_mm=operator.effective_head_mm[columns],
                       orientations_head=operator.orientations_head[columns])


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
        raise ValueError("No cortical triangles remain on the common source support.")
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
    def import_retained(prepared, retained):
        selected = {name: keys[retained] for name, keys in maps.items()}
        return importer(forwards, prepared.get_mesh_path(**mesh_options), vertices=selected,
                        **options)

    return prepare_subset(surfaces, masks, output_dir, mesh_options, import_retained,
                          solver='MNE', meshes=meshes)


def _write_surfaces(surfaces, prepared, meshes, retained, input_count, mesh_options):
    """Apply one anatomical column selection to the individual and multilayer GIfTIs."""
    for name, mesh in meshes.items():
        nib.save(_subset_gifti(mesh, retained),
                 prepared.get_mesh_path(layer_name=name, **mesh_options))
    # The combined GIfTI has separate vertices for each depth. Offset the same
    # retained IDs into each original block to reproduce the per-layer ordering.
    combined = nib.load(surfaces.get_mesh_path(**mesh_options))
    selected = np.concatenate([i * input_count + retained for i in range(len(meshes))])
    target = prepared.get_mesh_path(**mesh_options)
    nib.save(_subset_gifti(combined, selected), target)


def _native_exclusions(native_rejections, masks, input_ids):
    """Translate native search failures into original per-layer cortical vertex IDs."""
    if native_rejections is None:
        return {name: [] for name in masks}
    if set(native_rejections) != set(masks):
        raise ValueError("Native search masks must cover every cortical layer.")
    native = {}
    for name, rejected in native_rejections.items():
        rejected = np.asarray(rejected)
        if (rejected.dtype != bool or rejected.shape != input_ids.shape
                or np.any(rejected & masks[name])):
            raise ValueError("Native search rejections must identify excluded input vertices.")
        native[name] = input_ids[rejected].tolist()
    return native


def _fem_vertices_on_faces(available, faces, input_ids):
    """Keep available columns that can be used for patch simulation and smoothing."""
    # A retained triangle must still have all three original corners. Every other
    # available vertex is isolated after intersecting the masks across depths.
    connected = np.zeros_like(available)
    connected[faces[available[faces].all(axis=1)]] = True
    return connected, input_ids[available & ~connected].tolist()


def prepare_subset(surfaces, masks, output_dir, mesh_options, make_operator, *,
                   solver='DUNEuro', meshes=None, native_rejections=None):
    """Intersect layer masks, write consistent meshes, then build a gain on retained vertices."""
    layers = surfaces.get_layer_names()
    if set(masks) != set(layers):
        raise ValueError("Source masks must cover every cortical layer.")
    if meshes is None:
        meshes = {name: nib.load(surfaces.get_mesh_path(layer_name=name, **mesh_options))
                  for name in layers}
    input_count = len(meshes[layers[0]].agg_data('pointset'))
    for name in layers:
        if (np.asarray(masks[name]).dtype != bool or np.shape(masks[name]) != (input_count,)
                or len(meshes[name].agg_data('pointset')) != input_count):
            raise ValueError("Expected one boolean mask per layer in the input mesh order.")
    available = np.logical_and.reduce([masks[name] for name in layers])
    parent = getattr(surfaces, '__dict__', {}).get('forward_support')
    input_ids = np.arange(input_count) if parent is None else parent.original_vertices
    original_count = input_count if parent is None else parent.original_n_columns
    # Exclusions from successive models refer to the original cortical numbering.
    # Keep the specific depth where a location was lost, even though the vertex
    # is removed at every depth to compare gains on identical supports.
    omitted = {name: input_ids[~masks[name]].tolist() for name in layers}
    excluded = {name: sorted(set(omitted[name]).union(
        parent.excluded_by_layer[name] if parent else [])) for name in layers}
    native = _native_exclusions(native_rejections, masks, input_ids)
    native_count = sum(map(len, native.values()))
    native_message = (f" The native gray-matter check rejected {native_count} of the candidate "
                      "dipoles." if native_count else "")
    isolated = []
    if solver == 'DUNEuro':
        available, isolated = _fem_vertices_on_faces(
            available, meshes[layers[0]].agg_data('triangle'), input_ids)
    retained = np.flatnonzero(available)
    if retained.size == 0:
        raise ValueError(f"The {solver} forwards have no common cortical vertices on a mesh face."
                         f"{native_message}")
    original_ids = input_ids[retained]
    original_ids.setflags(write=False)
    support = ForwardSupport(original_count, original_ids, excluded)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as rollback:
        # Keep each preparation and its later SPM smoothing files in their own directory.
        # If validation fails, remove only this attempt's files.
        work = Path(mkdtemp(prefix=f'{solver.lower()}-forward-', dir=str(directory)))
        rollback.callback(shutil.rmtree, work)
        prepared = copy(surfaces)
        prepared.laminar_surf_dir = str(work)
        prepared.forward_support = support
        _write_surfaces(surfaces, prepared, meshes, retained, input_count, mesh_options)
        operator = make_operator(prepared, retained)
        # Validate the combined gain against each saved GIfTI, since layer model
        # comparisons register those individual files with SPM.
        for name in layers:
            operator.validate_mesh(name, prepared.get_mesh_path(layer_name=name, **mesh_options))
        # Save the compact-to-anatomical mapping beside the meshes so a saved source
        # row can still be located on the original cortex in a later analysis.
        manifest = work / 'source_support.json'
        report = {'subject': surfaces.subj_id, 'mesh_options': mesh_options, 'layer_names': layers,
                  'original_n_columns': original_count, 'original_vertices': original_ids.tolist(),
                  'excluded_by_layer': excluded}
        if solver == 'DUNEuro':
            # Keep the anatomical tissue test separate from the solver's lookup.
            # A native search can fail for a point inside a gray-matter tetrahedron.
            tissue = {name: sorted(set(omitted[name]).difference(native[name])) for name in layers}
            report.update(solver=solver, excluded_by_tissue=tissue,
                          excluded_by_native_search=native, vertices_without_faces=isolated)
            if parent is not None:
                report['input_manifest'] = str(Path(surfaces.laminar_surf_dir) / manifest.name)
        manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if len(retained) != original_count:
            counts = ', '.join(f"{name}: {len(ids)}" for name, ids in excluded.items() if ids)
            warnings.warn(f"Using {len(retained)} of {original_count} original cortical vertices "
                          f"at every depth after {solver} preparation ({counts})."
                          f"{native_message} See {manifest}.",
                          UserWarning, stacklevel=3)
        rollback.pop_all()
    return PreparedForward(prepared, operator, support, manifest, surfaces, dict(mesh_options))
