"""Attach a validated, fixed-orientation gain to an SPM MEG inversion slot."""
from functools import lru_cache
import hashlib
import json
import operator
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from scipy.io import loadmat, savemat

from ._gain_cache import (active_slot_index, cache_hash, cell_strings, file_identity,
                          gain_reference, inversion_slot, mat_string, open_spm_header,
                          verified_gain)
from .util import spm_context


# Keep this in the package so the same code runs in SPM standalone and MATLAB.
# The gain cache lives next to D; patch smoothing continues to use the original
# per-layer GIfTI. SPM's physical gains are divided by forward.scale in simulation.
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
    channels = D.indchantype({'MEG','MEGPLANAR'},'GOOD');
    labels = D.chanlabels(channels);
    units = D.units(channels);
    [found,rows] = ismember(labels(:), op.label(:));
    assert(all(found) && numel(labels)==numel(op.label), 'Good MEG channels differ.');
    assert(isequal(units(:), op.channel_units(rows)), 'MEG channel units differ.');

    ctx = gifti(original.mesh.tess_ctx);
    M = fw.fromMNI * original.mesh.Affine;
    requested = spm_eeg_inv_transform_points(M, double(ctx.vertices));
    assert(isequal(size(requested),size(op.requested_m)), 'Different source count.');
    assert(all(sqrt(sum((requested-op.requested_m).^2,2)) <= 1e-6), ...
        'Registered source positions or vertex order differ (metres).');
    assert(isequal(double(ctx.faces), double(fw.mesh.face)), 'Surface topology differs.');
    if isfield(ctx, 'normals')
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
        ca = any(a~=0,1); cb = any(b~=0,1);
        pa = sens.coilpos(ca,:); pb = op.coilpos(cb,:);
        oa = sens.coilori(ca,:); ob = op.coilori(cb,:);
        a = a(:,ca); b = b(:,cb);
        done_a = false(size(pa,1),1); done_b = false(size(pb,1),1);
        for i=1:size(pa,1)
            if done_a(i), continue; end
            ia = ~done_a & max(abs(bsxfun(@minus,pa,pa(i,:))),[],2) <= 1e-7;
            ib = ~done_b & max(abs(bsxfun(@minus,pb,pa(i,:))),[],2) <= 1e-7;
            assert(any(ib),'MEG coil positions differ.');
            ta = a(:,ia)*oa(ia,:); tb = b(:,ib)*ob(ib,:);
            assert(all(abs(ta(:)-tb(:)) <= 1e-9 + 1e-7*abs(ta(:))), ...
                'MEG coil orientations or channel compensation differ.');
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
    original.forward.mesh.vert = op.positions_m;
    original.forward.scale = 1;
    original.lameg_external_gain = struct('file',cfg.cache_name, ...
        'provenance',op.provenance,'format_version',1,'sha256',cfg.file_digest, ...
        'units','channel units / nA m');
    for field={'inverse','contrast'}
        if isfield(original,field{1}), original=rmfield(original,field{1}); end
    end
    D.inv{slot} = original;
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
                  requested_positions_m, sensor_geometry, provenance):
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
    digest = hashlib.sha256(json.dumps([names, units, meta]).encode())
    for key, array in fields.items():
        digest.update(key.encode())
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    fields.update(label=_strings(names), channel_units=_strings(units),
                  provenance=meta, format_version=1)
    return fields, f'lameg_gain_{digest.hexdigest()}.mat'


def attach_spm_gain(data_fname, gain, channel_names, source_positions_m, source_normals,
                    channel_units, *, inversion_idx=0, spm_instance=None,
                    requested_positions_m=None, sensor_geometry=None, provenance=None):
    """Select an external MEG gain on an already coregistered, single-layer SPM dataset.

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
                                 provenance)
    cache = path.with_name(name)
    before = file_identity(path)
    with TemporaryDirectory(dir=path.parent) as temp:
        folder = Path(temp)
        staged = folder/name
        header = folder/'header.mat'
        status = folder/'status.mat'
        config = folder/'settings.mat'
        savemat(staged, fields)
        actual = cache if cache.is_file() else staged
        file_digest = cache_hash(actual, file_identity(actual))
        payload = {'dataset': str(path), 'cache_name': cache.name, 'slot': slot + 1,
                   'read_cache': str(actual.relative_to(path.parent)), 'file_digest': file_digest,
                   'staged_gain': str(staged), 'staged_header': str(header),
                   'status_file': str(status)}
        savemat(config, payload)
        config_literal = str(config).replace("'", "''")
        with spm_context(spm_instance) as spm:
            spm.spm_standalone('eval', f"config_file='{config_literal}';\n{_ATTACH}", nargout=0)
        if not status.is_file():
            raise RuntimeError('SPM did not complete external gain validation.')
        error = np.asarray(loadmat(status, squeeze_me=True)['message'])
        if error.size:
            raise ValueError(f'SPM external gain validation failed: {error.item()}')
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
        os.replace(header, path)
    return cache


_MEG_TYPES = {'MEG', 'MEGMAG', 'MEGGRAD', 'MEGPLANAR'}


def _h5_good_labels(header):
    """Return labels selected by SPM for magnetic channels, including an active montage."""
    channels = header['D/channels']
    montage = int(np.asarray(header.get('D/montage/Mind', 0)).item())
    if montage:
        channels = header['D/montage/M/channels']
        if isinstance(channels, h5py.Dataset):
            channels = header[channels[()].ravel()[montage - 1]]

    def values(key, read):
        field = channels[key]
        if h5py.check_dtype(ref=field.dtype) is not None:
            return [read(header[ref]) for ref in field[()].ravel()]
        return [read(field)]

    names = values('label', mat_string)
    types = values('type', mat_string)
    bad = values('bad', lambda item: item[()].item())
    return tuple(name for name, kind, flag in zip(names, types, bad)
                 if not flag and kind.upper() in _MEG_TYPES)


def _legacy_good_labels(dataset):
    montage = getattr(dataset, 'montage', None)
    active = int(getattr(montage, 'Mind', 0))
    model = np.atleast_1d(montage.M).ravel()[active - 1] if active else dataset
    return tuple(row.label for row in np.atleast_1d(model.channels).ravel()
                 if not row.bad and row.type.upper() in _MEG_TYPES)


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
        if inversion_idx is not None and inversion_idx != active:
            if registration is not None or read_registration(inversion_idx) is not None:
                raise ValueError('The active SPM slot differs from the requested external slot.')
        if registration is None:
            return None
        read_labels = _h5_good_labels if isinstance(header, h5py.File) else _legacy_good_labels
        return registration + (read_labels(header),)


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
