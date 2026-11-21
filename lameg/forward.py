"""Fixed-orientation MEG forward operators on laMEG surface vertices."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from types import MappingProxyType

import h5py
import nibabel as nib
import numpy as np
from scipy.io import loadmat

from ._gain_cache import (cell_strings, gain_reference, inversion_slot, mat_string,
                          validate_slot_index, verified_gain)
from ._mat_sparse import h5_to_csc


def _labels(values, name, unique=True):
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of names.")
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate entries.")
    return values


def _numeric(value, shape, name):
    array = np.asarray(value)
    if (array.ndim != len(shape)
            or any(want is not None and have != want for have, want in zip(array.shape, shape))
            or 0 in array.shape or array.dtype.kind not in 'iuf'
            or not np.isfinite(array).all()):
        raise ValueError(f"Invalid {name}: expected finite, real data of shape {shape}.")
    # Layer views share potentially large gain matrices. Mark this view read-only so that
    # an in-place edit through one layer cannot change the operator used by another.
    view = array.view()
    view.setflags(write=False)
    return view


def _affine_mm(value, name, rigid=False):
    affine = _numeric(value, (4, 4), name)
    rotation = affine[:3, :3]
    if (not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or np.linalg.matrix_rank(rotation) != 3):
        raise ValueError(f"{name} must be an invertible homogeneous affine.")
    if rigid and (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                  or np.linalg.det(rotation) < 0):
        raise ValueError(f"{name} must be a rigid transform with a proper rotation.")
    return affine


def _directions(native, mri_to_head_mm):
    # SPM transforms GIfTI normals with the inverse transpose of the MRI-to-head linear
    # transform. Here normals are row vectors, so they multiply the inverse on the right.
    vectors = np.asarray(native, dtype=float) @ np.linalg.inv(mri_to_head_mm[:3, :3])
    # Normalization preserves the mesh polarity. An exactly zero input remains undefined.
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


def _mesh_arrays(path):
    mesh = nib.load(str(path))
    def read_intent(intent):
        matches = mesh.get_arrays_from_intent(intent)
        if len(matches) != 1:
            raise ValueError(f"Expected one {intent} array in {path}.")
        return matches[0].data

    return (read_intent('NIFTI_INTENT_POINTSET'), read_intent('NIFTI_INTENT_TRIANGLE'),
            read_intent('NIFTI_INTENT_VECTOR'))


def _local_faces(faces, n_columns, n_layers):
    faces = _numeric(faces, (None, 3), 'mesh faces')
    if (faces.dtype.kind not in 'iu' or faces.min() < 0
            or faces.max() >= n_columns * n_layers):
        raise ValueError("Mesh faces must be zero-based triangle indices in the source space.")
    # Combined meshes concatenate whole layers. Dividing a vertex ID by the layer size
    # gives the layer it belongs to. Each triangle must lie entirely within one layer.
    blocks = faces // n_columns
    if np.any(blocks != blocks[:, :1]):
        raise ValueError("Triangles cross cortical layer blocks.")
    reference = faces[blocks[:, 0] == 0]
    if reference.size == 0:
        raise ValueError("Empty first-layer topology.")
    # Subtract each layer offset to compare its connectivity in the same anatomical index space.
    for index in range(1, n_layers):
        layer = faces[blocks[:, 0] == index] - index * n_columns
        if not np.array_equal(layer, reference):
            raise ValueError("Ordered vertex connectivity differs between layers.")
    return reference


def _layer_mesh(path, layers):
    points, faces, normals = _mesh_arrays(path)
    if len(points) % len(layers) or normals.shape != points.shape:
        raise ValueError("Mesh has unequal layer sizes or unmatched source orientations.")
    local_faces = _local_faces(faces, len(points) // len(layers), len(layers))
    return points, faces, normals, local_faces


@dataclass(frozen=True, eq=False)
class ForwardOperator:  # pylint: disable=too-many-instance-attributes
    """Unprojected channels × sources gain in each channel's physical units per nA m.

    Columns follow consecutive layer blocks in `layer_names`, each in the
    original GIfTI vertex order. `requested_mri_mm` is the native GIfTI
    point set. `mri_to_head_mm` maps its coordinates into the MEG head
    frame, where `effective_head_mm` records the actual solver positions
    and `orientations_head` gives the signed fixed current directions.
    Positions and directions have shape (n_sources, 3). `faces` is a
    single layer's zero-based triangle array.

    `sensor_geometry` contains coil positions `coilpos` (head metres),
    signed unit vectors `coilori`, and the channel-by-coil transfer matrix
    `tra`, including reference compensation. The transfer rows follow
    `channel_names`. `provenance` identifies the solver and input.

    A zero direction means undefined. Set `allow_undefined_gain=True`
    explicitly to preserve legacy nonzero gain columns with missing
    directions. `orientation_defined` identifies sources usable for
    oriented-source comparisons. Input arrays are shared; treat them as
    immutable for the lifetime of this operator and its layer views.
    `allow_nominal_ctf_coils` opts into matching nominal CTF pickup coils
    to measured coils from the same acquisition when attaching to SPM.
    """

    gain: np.ndarray
    channel_names: tuple
    channel_units: tuple
    layer_names: tuple
    requested_mri_mm: np.ndarray
    effective_head_mm: np.ndarray
    orientations_head: np.ndarray
    mri_to_head_mm: np.ndarray
    faces: np.ndarray
    sensor_geometry: dict
    provenance: str
    allow_undefined_gain: bool = False
    allow_nominal_ctf_coils: bool = False

    def __post_init__(self):
        def set_field(key, value):
            object.__setattr__(self, key, value)

        if not isinstance(self.allow_nominal_ctf_coils, (bool, np.bool_)):
            raise ValueError("allow_nominal_ctf_coils must be a boolean.")
        names = _labels(self.channel_names, 'channel_names')
        units = _labels(self.channel_units, 'channel_units', unique=False)
        if len(units) != len(names) or any(
                not re.fullmatch(r'(?:[fpnuµμm]?T)(?:/(?:m|cm|mm))?', u) for u in units):
            raise ValueError("Each channel requires a physical MEG field or gradient unit.")
        layers = _labels(self.layer_names, 'layer_names')
        gain = _numeric(self.gain, (len(names), None), 'gain')
        if gain.shape[1] % len(layers):
            raise ValueError("Gain columns do not form equal-sized cortical layer blocks.")
        set_field('channel_names', names)
        set_field('channel_units', units)
        set_field('layer_names', layers)
        set_field('gain', gain)
        # Row j of each array describes gain column j. If a solver evaluates the dipole at
        # a displaced position, the original MRI vertex still identifies its cortical patch.
        for key in ('requested_mri_mm', 'effective_head_mm', 'orientations_head'):
            set_field(key, _numeric(getattr(self, key), (gain.shape[1], 3), key))
        direction_norm = np.linalg.norm(self.orientations_head, axis=1)
        valid = direction_norm != 0
        if not np.allclose(direction_norm[valid], 1, atol=1e-6, rtol=0):
            raise ValueError("Source orientations must be signed unit vectors or exactly zero.")
        if not self.allow_undefined_gain and np.any(gain[:, ~valid] != 0):
            raise ValueError("Nonzero gain at an undefined orientation; verify source convention.")
        set_field('mri_to_head_mm', _affine_mm(self.mri_to_head_mm, 'mri_to_head_mm'))
        faces = _numeric(self.faces, (None, 3), 'faces')
        if faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= self.n_columns:
            raise ValueError("faces must index a single cortical layer, starting at zero.")
        set_field('faces', faces)
        geo = self.sensor_geometry
        coils = _numeric(geo['coilpos'], (None, 3), 'coilpos')
        coilori = _numeric(geo['coilori'], coils.shape, 'coilori')
        if not np.allclose(np.linalg.norm(coilori, axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("Sensor coil orientations must be unit vectors.")
        # A channel can combine several physical pickups, including reference coils with signed
        # weights. Each row of tra must describe the same channel as its corresponding gain row.
        tra = _numeric(geo['tra'], (len(names), len(coils)), 'tra')
        if not np.all(np.any(tra != 0, axis=1)):
            raise ValueError("Each sensor channel must have a nonzero coil weight.")
        set_field('sensor_geometry', MappingProxyType({
            'coilpos': coils, 'coilori': coilori, 'tra': tra}))
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Give the forward solver and source of the gain as provenance.")

    @property
    def n_columns(self):
        """Number of within-surface vertex indices."""
        return self.gain.shape[1] // len(self.layer_names)

    @property
    def orientation_defined(self):
        """Boolean mask of sources with a defined signed current direction."""
        return np.any(self.orientations_head != 0, axis=1)

    def layer(self, name):
        """Return a single-layer operator retaining its explicit vertex order."""
        if name not in self.layer_names:
            raise ValueError(f"Unknown layer {name!r}; available layers: {self.layer_names}.")
        if len(self.layer_names) == 1:
            return self
        # Select the same layer block from the gain and its source metadata. Sensor rows and
        # the single-layer face indices already apply to every layer.
        start = self.layer_names.index(name) * self.n_columns
        columns = slice(start, start + self.n_columns)
        return replace(self, gain=self.gain[:, columns], layer_names=(name,),
                       requested_mri_mm=self.requested_mri_mm[columns],
                       effective_head_mm=self.effective_head_mm[columns],
                       orientations_head=self.orientations_head[columns])

    def validate_mesh(self, layer_name, path):
        """Require matching vertices, connectivity and signed orientations for a laMEG mesh.

        Pass None for the combined mesh, or a layer name and its individual
        GIfTI. Effective source displacement is allowed and is recorded in
        head coordinates separately from the original MRI points.
        """
        model = self if layer_name is None else self.layer(layer_name)
        points, faces, normals = _mesh_arrays(path)
        checked = _local_faces(faces, model.n_columns, len(model.layer_names))
        if not np.array_equal(checked, model.faces):
            raise ValueError("Cortical topology differs from the operator's vertex ordering.")
        if (points.shape != model.requested_mri_mm.shape
                or not np.allclose(points, model.requested_mri_mm, rtol=0, atol=1e-4)):
            raise ValueError("Requested MRI positions or cortical vertex ordering differ.")
        if (normals.shape != model.orientations_head.shape or not np.allclose(
                _directions(normals, model.mri_to_head_mm), model.orientations_head,
                rtol=0, atol=1e-5)):
            raise ValueError("Signed source orientations differ from the requested mesh.")

    @classmethod
    def from_spm(cls, data_fname, mesh_fname, layer_names, inversion_idx=0, gain_fname=None,
                 allow_undefined_gain=False):
        """Read a saved SPM v7.3 MEG forward and its referenced gain file.

        This importer supports one MEG modality, SPM's SI-aware forward
        path, and an explicitly oriented laMEG GIfTI. It reads only the
        header and gain cache. `inversion_idx` is zero-based. Supply the
        original source mesh; if moved, `gain_fname` can locate the original
        cache by its saved basename. The header's scale is undone once.
        SPM gain files contain channel labels and gain values; source
        correspondence depends on using the dataset's original cache.
        Relocated sources require the actual original GIfTI. For native
        single-shell and sphere models, the saved source positions provide
        an additional independent check of MRI coordinates.
        """
        validate_slot_index(inversion_idx)
        layers = _labels(layer_names, 'layer_names')
        # The surface specifies the source order. The selected SPM slot supplies the channel
        # order, sensor response, registration, and identity of the gain computed on that mesh.
        points, faces, normals, local_faces = _layer_mesh(mesh_fname, layers)
        metadata, cache_name, scale, expected_hash = _spm_registration(
            data_fname, inversion_idx, mesh_fname, points, faces, normals)
        cache = Path(data_fname).parent / cache_name if gain_fname is None else Path(gain_fname)
        if cache.name != cache_name:
            raise ValueError("Gain cache basename differs from the selected inversion's cache.")
        with verified_gain(cache, expected_hash):
            gain, cache_labels, fields = _read_gain_cache(cache, expected_hash)
        # Use labels to recover the physical channel represented by each row, even when the
        # cache and recording contain the same number of channels in different orders.
        rows = _index_labels(cache_labels, metadata['channel_names'])
        if gain.shape != (len(cache_labels), len(points)):
            raise ValueError("SPM gain shape differs from the cortical mesh or channel labels.")
        if cache_labels != metadata['channel_names']:
            gain = gain[rows]
        if expected_hash is not None:
            _restore_external(fields, metadata, rows, scale, points)
        # SPM may condition its native gain before saving it. Undo that factor exactly once
        # to recover gain in the recorded channel units per nA m.
        if scale != 1:
            gain = gain / scale
        return cls(gain=gain, layer_names=layers, requested_mri_mm=points, faces=local_faces,
                   allow_undefined_gain=allow_undefined_gain, **metadata)

    @classmethod
    def from_mne(cls, forwards, mesh_fname, *, vertices, channel_units,
                 mri_to_head_mm, mne_head_to_head_mm, info=None, ignore_ref=None,
                 allow_nominal_ctf_coils=False):
        """Import precomputed MNE-Python MEG forwards on an oriented laMEG GIfTI.

        `forwards` maps layer names in GIfTI block order to MNE Forward objects
        or FIF paths. `vertices` maps each within-layer GIfTI vertex to an
        integer (MNE source-space index, vertex ID) pair. Pass a single
        (n_columns, 2) array or one array per layer keyed by name. Every
        requested position must be present; extra MNE sources may be omitted.

        `channel_units` maps the recipient's good MEG labels to physical units
        in the desired row order. For compensated forwards supply the complete
        acquisition `info` and explicitly confirm the original solve used
        `ignore_ref=False`. The FIF does not record this setting.
        Free-orientation gains are projected onto the signed GIfTI directions.
        Fixed-orientation gains require collinear dipoles at defined vertices.

        Supply the GIfTI MRI-to-recipient-head affine and the rigid MNE-head-
        to-recipient-head transform explicitly. Both map millimetres to
        millimetres. The latter applies to both sources and sensor coils.
        For matching CTF acquisitions, `allow_nominal_ctf_coils=True`
        permits small differences between nominal and measured coil geometry
        during SPM attachment (see the precomputed forwards guide).
        """
        # SPM-only readers do not need to initialize MNE.
        from ._mne_forward import read_layers  # pylint: disable=import-outside-toplevel

        if not isinstance(forwards, Mapping) or not isinstance(channel_units, Mapping):
            raise TypeError("forwards and channel_units must be mappings.")
        layers = _labels(forwards, 'layer_names')
        names = _labels(channel_units, 'channel_names')
        units = _labels(channel_units.values(), 'channel_units', unique=False)
        if isinstance(vertices, Mapping) and set(vertices) != set(layers):
            raise ValueError("Per-layer vertex mappings must match the forward layer names.")
        points, _, normals, faces = _layer_mesh(mesh_fname, layers)
        affine = _affine_mm(mri_to_head_mm, 'mri_to_head_mm')
        head = _affine_mm(mne_head_to_head_mm, 'mne_head_to_head_mm', rigid=True)
        # The registered GIfTI points and signed directions provide the anatomical targets
        # for checking MNE source positions and projecting the dipoles in head coordinates.
        requested = points @ affine[:3, :3].T + affine[:3, 3]
        orientations = _directions(normals, affine)
        gain, effective, sensors, provenance = read_layers(
            forwards, vertices, names, units, head, requested, orientations, info, ignore_ref,
            allow_nominal_ctf_coils)
        return cls(gain=gain, channel_names=names, channel_units=units, layer_names=layers,
                   requested_mri_mm=points, effective_head_mm=effective,
                   orientations_head=orientations, mri_to_head_mm=affine, faces=faces,
                   sensor_geometry=sensors, provenance=provenance,
                   allow_nominal_ctf_coils=allow_nominal_ctf_coils)


def _spm_source_coordinates(inversion, mesh_fname, points, faces, normals):
    forward = inversion['forward']
    registered = inversion['datareg']
    mesh = inversion['mesh']
    if (mat_string(forward['modality']) != 'MEG' or mat_string(registered['modality']) != 'MEG'
            or forward['siunits'][()].item() != 1):
        raise ValueError("A single MEG registration with physical sensor units is required.")
    recorded = Path(mat_string(mesh['tess_ctx']).replace('\\', '/'))
    if recorded.name != Path(mesh_fname).name:
        raise ValueError("GIfTI basename differs from the dataset's recorded source mesh.")
    if recorded.is_file() and not recorded.samefile(mesh_fname):
        original = _mesh_arrays(recorded)
        if any(not np.array_equal(a, b) for a, b in zip(original, (points, faces, normals))):
            raise ValueError("Supplied GIfTI differs from the original source mesh.")
    if not np.array_equal(forward['mesh/face'][()].T, faces.astype(np.int64) + 1):
        raise ValueError("Original and saved forward meshes have different face indices.")
    # Compose the native MRI-to-MNI and MNI-to-head transforms. MATLAB v7.3 datasets
    # have their axes reversed in h5py. SPM stores effective head positions in metres.
    affine = registered['fromMNI'][()].T @ mesh['Affine'][()].T
    effective = forward['mesh/vert'][()].T * 1000
    return affine, effective


def _spm_sensors(header, forward, labels):
    sensors = forward['sensors']
    rows = _index_labels(_mat_strings(header, sensors['label']), labels)
    if mat_string(sensors['unit']) != 'm':
        raise ValueError("Expected SPM's physical sensor coil coordinates in metres.")
    sensor_units = _mat_strings(header, sensors['chanunit'])
    if any(sensor_units[i] not in ('T', 'T/m') for i in rows):
        raise ValueError("Expected SI-calibrated MEG sensor coil weights.")
    geometry = {'coilpos': sensors['coilpos'][()].T, 'coilori': sensors['coilori'][()].T,
                'tra': _mat_matrix(sensors['tra'])[rows, :]}
    # The forward sensor table describes SI pickup geometry. Recorded data can be in fT
    # or other physical units. Retrieve these units independently by channel label.
    channels = header['D/channels']
    data_rows = _index_labels(_mat_strings(header, channels['label']), labels)
    units = _mat_strings(header, channels['units'])
    return geometry, tuple(units[i] for i in data_rows)


def _spm_registration(data_fname, inversion_idx, mesh_fname, points, faces, normals):
    with h5py.File(data_fname, 'r') as header:
        inversion = inversion_slot(header, inversion_idx)
        if inversion is None:
            raise ValueError("Requested inversion_idx is absent from the SPM header.")
        forward = inversion['forward']
        affine, effective = _spm_source_coordinates(inversion, mesh_fname, points, faces, normals)
        labels = _mat_strings(header, forward['channels'])
        geometry, units = _spm_sensors(header, forward, labels)
        scale = float(forward['scale'][()].item())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid SPM forward conditioning scale.")
        recorded, expected_hash = gain_reference(inversion)
        cache_name = Path(recorded.replace('\\', '/')).name
        if not cache_name:
            raise ValueError("No SPM gain cache is recorded for this inversion.")
        solver = mat_string(forward['voltype'])
        if expected_hash is None and solver in (
                'Single Shell', 'Single Sphere', 'MEG Local Spheres'):
            requested = points @ affine[:3, :3].T + affine[:3, 3]
            if (requested.shape != effective.shape
                    or not np.allclose(requested, effective, atol=1e-4, rtol=0)):
                raise ValueError("GIfTI coordinates differ from the native uncorrected forward.")
    metadata = {'channel_names': labels, 'channel_units': units, 'sensor_geometry': geometry,
                'effective_head_mm': effective, 'orientations_head': _directions(normals, affine),
                'mri_to_head_mm': affine,
                'provenance': f"SPM {solver}; {Path(data_fname).name}; slot {inversion_idx + 1}"}
    return metadata, cache_name, scale, expected_hash


def _restore_external(fields, metadata, rows, scale, points):
    # The retained headmodel identifies the registration template; the
    # attached cache carries the solver used for the actual lead field.
    if scale != 1 or int(fields['format_version'].item()) != 1:
        raise ValueError("Unexpected external-gain cache version or conditioning.")
    units = cell_strings(fields['channel_units'])
    if tuple(units[i] for i in rows) != metadata['channel_units']:
        raise ValueError("External gain channel units differ from the SPM dataset.")
    affine = metadata['mri_to_head_mm']
    requested = points @ affine[:3, :3].T + affine[:3, 3]
    # An attached cache carries its own geometry. Check intended and actual head positions
    # independently so the cache can preserve a solver displacement across re-import.
    for key, target, atol in [('requested_m', requested / 1000, 1e-6),
                              ('positions_m', metadata['effective_head_mm'] / 1000, 1e-9),
                              ('normals', metadata['orientations_head'], 1e-5)]:
        if fields[key].shape != target.shape or not np.allclose(
                fields[key], target, rtol=0, atol=atol):
            raise ValueError(f"External gain {key} differs from the supplied GIfTI or header.")
    metadata['orientations_head'] = fields['normals']
    source = json.loads(str(fields['provenance'].item()))
    if not isinstance(source, str):
        source = json.dumps(source, sort_keys=True)
    metadata['provenance'] = source
    if 'allow_nominal_ctf_coils' in fields:
        if fields['allow_nominal_ctf_coils'].item() != 1:
            raise ValueError('Invalid nominal CTF coil marker in gain cache.')
        metadata['allow_nominal_ctf_coils'] = True
    if all(k in fields for k in ('coilpos', 'coilori', 'tra')):
        metadata['sensor_geometry'] = {'coilpos': fields['coilpos'], 'coilori': fields['coilori'],
                                       'tra': fields['tra'][rows]}


def _mat_strings(header, cells):
    return tuple(mat_string(header[ref]) for ref in cells[()].ravel())


def _index_labels(available, required):
    lookup = {name: i for i, name in enumerate(available)}
    if len(lookup) != len(available) or not set(required).issubset(lookup):
        raise ValueError("Channel labels are duplicated or missing in the SPM files.")
    return [lookup[name] for name in required]


def _read_gain_cache(cache, expected_hash):
    if h5py.is_hdf5(str(cache)):
        with h5py.File(str(cache), 'r') as header:
            if expected_hash is not None:
                raise ValueError("Expected a laMEG version-1 MATLAB v5 external gain cache.")
            return np.asarray(header['G']).T, _mat_strings(header, header['label']), {}
    keys = ['G', 'label']
    if expected_hash is not None:
        keys += ['requested_m', 'positions_m', 'normals', 'provenance', 'channel_units',
                 'coilpos', 'coilori', 'tra', 'format_version', 'allow_nominal_ctf_coils']
    fields = loadmat(str(cache), variable_names=keys)
    return fields.pop('G'), cell_strings(fields.pop('label')), fields


def _mat_matrix(node):
    if isinstance(node, h5py.Dataset):
        return node[()].T
    return h5_to_csc(node).toarray()
