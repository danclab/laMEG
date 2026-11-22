"""Attach a validated, fixed-orientation gain to an SPM MEG inversion slot."""
from functools import lru_cache
import hashlib
import json
import operator
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import loadmat, savemat

from ._gain_cache import (active_slot_index, cache_hash, cell_strings, file_identity,
                          gain_reference, good_meg_channels, inversion_slot, open_spm_header,
                          validate_slot_index, verified_gain)
from .forward import ForwardOperator
from .util import spm_context


# Keep this in the package so the same code runs in SPM standalone and MATLAB.
# SPM reads the gain from the dataset directory. Patch smoothing still uses the
# registered GIfTI, including its anatomical connectivity and vertex order.
_ATTACH = """
try
    cfg = load(config_file);
    op = load(cfg.staged_gain);
    D = spm_eeg_load(cfg.dataset);
    slot = cfg.slot;
    assert(slot <= numel(D.inv), 'No coregistration in the selected inversion slot.');
    original = D.inv{slot};
    fw = original.forward;
    assert(isscalar(fw) && strcmp(fw.modality, 'MEG'), 'One MEG forward is required.');
    % Labels and physical units come from the recipient's active montage. Map each
    % good sensor to its row in G so a different channel order preserves the topography.
    channels = D.indchantype({'MEG','MEGPLANAR'},'GOOD');
    labels = D.chanlabels(channels);
    units = D.units(channels);
    [found,rows] = ismember(labels(:), op.label(:));
    assert(all(found) && numel(labels)==numel(op.label), 'Good MEG channels differ.');
    assert(isequal(units(:), op.channel_units(rows)), 'MEG channel units differ.');

    ctx = gifti(original.mesh.tess_ctx);
    if isfield(cfg,'input_mesh')
        input_mesh = gifti(cfg.input_mesh);
        assert(isequal(double(input_mesh.faces),double(ctx.faces)), ...
            'Inversion mesh topology differs from the registered mesh.');
    end
    % Map the registered cortical vertices through MNI into the recording's head
    % coordinates. Compare the requested sites here even if the solver displaced dipoles.
    M = fw.fromMNI * original.mesh.Affine;
    requested = spm_eeg_inv_transform_points(M, double(ctx.vertices));
    assert(isequal(size(requested),size(op.requested_m)), 'Different source count.');
    assert(all(sqrt(sum((requested-op.requested_m).^2,2)) <= 1e-6), ...
        'Registered source positions or vertex order differ (metres).');
    assert(isequal(double(ctx.faces), double(fw.mesh.face)), 'Surface topology differs.');
    if isfield(ctx, 'normals')
        % For row vectors the inverse-transpose normal transform is right division
        % by the linear part of M. Keep undefined (zero) directions zero.
        normals = double(ctx.normals) / M(1:3,1:3);
        lengths = sqrt(sum(normals.^2,2));
        defined = lengths > 0;
        normals(defined,:) = bsxfun(@rdivide,normals(defined,:),lengths(defined));
    else
        normals = spm_mesh_normals(struct('faces',fw.mesh.face,'vertices',fw.mesh.vert),true);
    end
    assert(all(sqrt(sum((normals-op.normals).^2,2)) <= 1e-5), ...
        'Signed dipole orientations differ.');

    if isfield(op, 'coilpos')
        sens = ft_convert_units(fw.sensors,'m');
        [found,srows] = ismember(labels(:), sens.label(:));
        assert(all(found), 'MEG forward lacks a selected channel.');
        a = sens.tra(srows,:); b = op.tra(rows,:);
        % The physical channel response for colocated coils is the weighted
        % sum of their orientation vectors. This handles compensation, coil
        % permutations, triaxial sensors and equivalent sign conventions.
        % Keep reference pickups whenever they contribute to a good sensor. Their
        % signed coefficients are part of that sensor's response after compensation.
        ca = any(a~=0,1); cb = any(b~=0,1);
        pa = sens.coilpos(ca,:); pb = op.coilpos(cb,:);
        oa = sens.coilori(ca,:); ob = op.coilori(cb,:);
        a = a(:,ca); b = b(:,cb);
        nominal = isfield(op,'allow_nominal_ctf_coils') && op.allow_nominal_ctf_coils;
        if nominal
            assert(isfield(sens,'type') && strncmpi(sens.type,'ctf',3), ...
                'Nominal CTF pickup coils require a CTF recipient.');
        end
        done_a = false(size(pa,1),1); done_b = false(size(pb,1),1);
        for i=1:size(pa,1)
            if done_a(i), continue; end
            ia = ~done_a & max(abs(bsxfun(@minus,pa,pa(i,:))),[],2) <= 1e-7;
            delta = bsxfun(@minus,pb,pa(i,:));
            if nominal
                % MNE's nominal CTF baselines can differ from measured centers. Each
                % nominal pickup must be within 0.15 mm of exactly one measured group.
                ib = sum(delta.^2,2) <= (0.15e-3)^2;
                assert(~any(done_b & ib),'Ambiguous nominal CTF pickup position.');
            else
                ib = ~done_b & max(abs(delta),[],2) <= 1e-7;
            end
            assert(any(ib),'MEG coil positions differ.');
            ta = a(:,ia)*oa(ia,:); tb = b(:,ib)*ob(ib,:);
            if nominal
                % Compare each channel's full signed vector, including reference weights.
                agrees = sqrt(sum((ta-tb).^2,2)) <= 1e-9 + 1e-5*sqrt(sum(ta.^2,2));
            else
                agrees = abs(ta(:)-tb(:)) <= 1e-9 + 1e-7*abs(ta(:));
            end
            assert(all(agrees),'MEG coil orientations or channel compensation differ.');
            done_a(ia)=true; done_b(ib)=true;
        end
        assert(all(done_b),'Unexpected active coils in external gain.');
    end

    % Check the actual SPM cache, including a previously published cache for
    % the same content. A native fallback writes a different gain filename.
    disk = load(fullfile(D.path,cfg.read_cache));
    assert(isequaln(disk,op), 'Gain cache differs.');
    clear disk
    D.val = slot;
    original.gainmat = cfg.read_cache;
    % Preserve the anatomical mesh for priors. Forward calculations use the solver's
    % effective positions and the gain is already in recorded channel units per nA m.
    original.forward.mesh.vert = op.positions_m;
    original.forward.scale = 1;
    original.lameg_external_gain = struct('file',cfg.cache_name, ...
        'provenance',op.provenance,'format_version',1,'sha256',cfg.file_digest, ...
        'units','channel units / nA m');
    % Estimates and contrasts in this slot were fitted with the previous gain.
    for field={'inverse','contrast'}
        if isfield(original,field{1}), original=rmfield(original,field{1}); end
    end
    D.inv{slot} = original;
    % Exercise SPM's own loader before committing. Verify it returned every signed
    % coefficient in sensor order and kept the selected cache as its gain source.
    [L,D] = spm_eeg_lgainmat(D, [], labels);
    assert(strcmp(D.inv{slot}.gainmat,cfg.read_cache) && isequal(L,op.G(rows,:)), ...
        'SPM did not use the supplied fixed-orientation gain.');
    D.inv{slot}.gainmat = cfg.cache_name;
    clear op L
    % Same struct representation as @meeg/save; stage it before replacing
    % the header. No dataset samples are copied or edited.
    D = struct(D);
    save(cfg.staged_header,'D',spm_get_defaults('mat.format'));
    message = '';
catch err
    message = err.message;
end
save(cfg.status_file,'message','-v7');
"""


def _strings(values):
    return np.asarray(tuple(values), dtype=object).reshape(-1, 1)


def _finite_real(value, name):
    array = np.asarray(value)
    if (array.dtype.kind not in 'iuf' or not np.isfinite(array).all()):
        raise ValueError(f'{name} must contain finite real numeric values.')
    return np.asarray(array, dtype=np.float64)


def _gain_payload(gain, channel_names, source_positions_m, source_normals, channel_units,
                  requested_positions_m, sensor_geometry, provenance,
                  allow_nominal_ctf_coils=False):
    names, units = tuple(channel_names), tuple(channel_units)
    if (not names or len(set(names)) != len(names) or len(units) != len(names)
            or not all(isinstance(x, str) and x.strip() for x in (*names, *units))):
        raise ValueError('Expected unique channel names and one unit per channel.')
    matrix = _finite_real(gain, 'gain')
    effective = _finite_real(source_positions_m, 'source positions')
    normal = _finite_real(source_normals, 'source normals')
    requested = (effective if requested_positions_m is None else
                 _finite_real(requested_positions_m, 'requested source positions'))
    if matrix.ndim != 2 or not matrix.shape[1] or matrix.shape[0] != len(names):
        raise ValueError('Expected a nonempty matrix with one row per channel.')
    if not all(arr.shape == (matrix.shape[1], 3) for arr in (effective, normal, requested)):
        raise ValueError('The gain must have one row per channel and one column per source.')
    size = np.linalg.norm(normal, axis=1)
    if not np.all((size == 0) | (np.abs(size-1) <= 1e-5)):
        raise ValueError('Nonzero source normals must be unit vectors.')
    meta = json.dumps(provenance if provenance is not None else {}, sort_keys=True,
                      allow_nan=False)
    fields = {'G': matrix, 'positions_m': effective, 'requested_m': requested, 'normals': normal}
    if sensor_geometry is not None:
        for key in ('coilpos', 'coilori', 'tra'):
            fields[key] = _finite_real(sensor_geometry[key], key)
        n_coils = len(fields['coilpos'])
        if (fields['coilpos'].shape != (n_coils, 3) or fields['coilori'].shape != (n_coils, 3)
                or fields['tra'].shape != (len(names), n_coils) or not n_coils):
            raise ValueError('Invalid MEG coil geometry or channel transfer matrix.')
    if not isinstance(allow_nominal_ctf_coils, (bool, np.bool_)):
        raise ValueError('allow_nominal_ctf_coils must be a boolean.')
    if allow_nominal_ctf_coils:
        if sensor_geometry is None:
            raise ValueError('Nominal CTF pickup matching requires coil geometry.')
        fields['allow_nominal_ctf_coils'] = np.array([[1]], dtype=np.uint8)
    # Name the cache by the physical operator. The same numbers with different units,
    # vertex positions or channel ordering describe a different forward model.
    digest = hashlib.sha256(json.dumps([names, units, meta]).encode())
    for key, array in fields.items():
        digest.update(key.encode())
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    fields.update(label=_strings(names), channel_units=_strings(units),
                  provenance=meta, format_version=1)
    return fields, f'lameg_gain_{digest.hexdigest()}.mat'


def validate_inversion_forward(forward_operator, mesh_fname, layer_names, inversion_idx, *,
                               priors=()):
    """Validate the source space and any zero-based MSP priors before installing the gain."""
    validate_slot_index(inversion_idx)
    if not isinstance(forward_operator, ForwardOperator):
        raise TypeError("forward_operator must be a ForwardOperator.")
    # A single-surface inversion needs one block of gain columns. A multilayer
    # inversion also depends on the order of blocks when pairing vertices across depths.
    layers = tuple(layer_names)
    selected = forward_operator.layer(layers[0]) if len(layers) == 1 else forward_operator
    if layers != selected.layer_names:
        raise ValueError("The forward operator must match the combined mesh's layer order.")
    selected.validate_mesh(None, mesh_fname)
    # Callers pass zero-based indices in the selected mesh (after any vertex exclusions).
    # On a combined mesh a prior may address a column in any layer block.
    indices = np.asarray(priors)
    if indices.size:
        if (indices.dtype.kind not in 'iu' or np.any(indices < 0)
                or np.any(indices >= selected.gain.shape[1])):
            raise ValueError("An MSP prior is outside the requested mesh.")
        if not np.all(selected.orientation_defined[indices]):
            raise ValueError("An MSP prior has an undefined direction in the forward operator.")
    return selected


def attach_forward_operator(data_fname, forward_operator, inversion_idx, spm_instance,
                            mesh_fname=None):
    """Install the selected operator on a registered dataset with matching source geometry."""
    # Requested points start in native MRI millimetres. Effective points are already
    # in head millimetres. SPM expects both sets in head metres at attachment.
    transform = forward_operator.mri_to_head_mm
    requested = forward_operator.requested_mri_mm @ transform[:3, :3].T + transform[:3, 3]
    return attach_spm_gain(
        data_fname, forward_operator.gain, forward_operator.channel_names,
        forward_operator.effective_head_mm / 1000, forward_operator.orientations_head,
        forward_operator.channel_units, inversion_idx=inversion_idx, spm_instance=spm_instance,
        requested_positions_m=requested / 1000, sensor_geometry=forward_operator.sensor_geometry,
        provenance=forward_operator.provenance, mesh_fname=mesh_fname,
        allow_nominal_ctf_coils=forward_operator.allow_nominal_ctf_coils)


def attach_spm_gain(data_fname, gain, channel_names, source_positions_m, source_normals,
                    channel_units, *, inversion_idx=0, spm_instance=None,
                    requested_positions_m=None, sensor_geometry=None, provenance=None,
                    allow_nominal_ctf_coils=False, mesh_fname=None):
    """Select an external MEG gain on an already coregistered SPM source space.

    Columns follow the target mesh. Positions and signed unit directions are in
    the SPM MEG head frame. Explicit zero directions identify undefined columns.
    Rows are named good MEG channels, in their native data units per nA m. When
    a solver relocated sources, supply its effective positions and the original
    registered positions separately. The sensor transfer matrix rows correspond
    to channel_names. The operation returns the persistent cache path and
    invalidates any earlier inversion results for this slot.
    """
    path = Path(data_fname).resolve(strict=True)
    slot = operator.index(inversion_idx)
    if slot < 0:
        raise ValueError('inversion_idx must be nonnegative.')
    fields, name = _gain_payload(gain, channel_names, source_positions_m, source_normals,
                                 channel_units, requested_positions_m, sensor_geometry,
                                 provenance, allow_nominal_ctf_coils)
    cache = path.with_name(name)
    before = file_identity(path)
    with TemporaryDirectory(dir=path.parent) as temp:
        folder = Path(temp)
        staged = folder/name
        header = folder/'header.mat'
        status = folder/'status.mat'
        config = folder/'settings.mat'
        savemat(staged, fields)
        # MATLAB files can encode the same arrays with different bytes. If a cache
        # already exists, validate its arrays against the staged payload in MATLAB and
        # record the checksum of the actual file SPM reads.
        actual = cache if cache.is_file() else staged
        file_digest = cache_hash(actual, file_identity(actual))
        # MATLAB indexes D.inv cells from one. read_cache locates the file for the
        # trial load, while cache_name is the basename to persist in the dataset.
        payload = {'dataset': str(path), 'cache_name': cache.name, 'slot': slot + 1,
                   'read_cache': str(actual.relative_to(path.parent)), 'file_digest': file_digest,
                   'staged_gain': str(staged), 'staged_header': str(header),
                   'status_file': str(status)}
        if mesh_fname is not None:
            payload['input_mesh'] = str(Path(mesh_fname).resolve(strict=True))
        savemat(config, payload)
        config_literal = str(config).replace("'", "''")
        with spm_context(spm_instance) as spm:
            spm.spm_standalone('eval', f"config_file='{config_literal}';\n{_ATTACH}", nargout=0)
        if not status.is_file():
            raise RuntimeError('SPM did not complete external gain validation.')
        error = np.asarray(loadmat(status, squeeze_me=True)['message'])
        if error.size:
            raise ValueError(f'SPM external gain validation failed: {error.item()}')
        # The validated header was derived from the initial dataset. Discard it if
        # the on-disk header has changed while MATLAB was working.
        if file_identity(path) != before:
            raise RuntimeError('SPM dataset changed while attaching the gain.')
        os.chmod(header, path.stat().st_mode)
        if actual == staged:
            # Publish the exact file that SPM read; errors above leave no artefact.
            try:
                os.link(staged, cache)
            except FileExistsError as exc:
                if cache_hash(cache, file_identity(cache)) != file_digest:
                    raise RuntimeError('Conflicting gain cache publication.') from exc
        # Install the header last so that its selected gain file already exists.
        os.replace(header, path)
    return cache


@lru_cache(maxsize=128)
def _gain_channel_labels(cache, _file_identity):
    # Only read the small MATLAB cell array. The separately verified SHA-256
    # binds those labels to the complete physical gain that native SPM read.
    cells = loadmat(cache, variable_names=['label'])['label']
    return frozenset(cell_strings(cells))


@lru_cache(maxsize=64)
def _gain_registration(path, _file_identity, inversion_idx):
    """Read only slot metadata from HDF5. Cache legacy MAT's monolithic header read."""
    with open_spm_header(path) as header:
        def read_registration(index):
            return gain_reference(inversion_slot(header, index), external_only=True)

        active = active_slot_index(header)
        registration = read_registration(active)
        # Spatial-mode preparation reads the active slot in SPM. Using it with an
        # external gain assigned to another slot could mix two forward models.
        if inversion_idx is not None and inversion_idx != active:
            if registration is not None or read_registration(inversion_idx) is not None:
                raise ValueError('The active SPM slot differs from the requested external slot.')
        if registration is None:
            return None
        return registration + (good_meg_channels(header),)


def check_spm_gain_cache(data_fname, inversion_idx=None):
    """Reject missing, unselected or altered external gains before SPM prepares modes."""
    path = Path(data_fname).resolve(strict=True)
    registration = _gain_registration(path, file_identity(path), inversion_idx)
    if registration is None:
        return
    recorded, digest, good_channels = registration
    cache = path.parent / recorded
    with verified_gain(cache, digest) as identity:
        missing = set(good_channels).difference(_gain_channel_labels(cache, identity))
    if missing:
        raise ValueError(f'Good MEG channels missing from the external gain: {sorted(missing)}')
