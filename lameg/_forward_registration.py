"""Read MEG channel geometry and MRI registration from the recording."""
from os import PathLike

import h5py
import numpy as np
from scipy.sparse import issparse

from ._gain_cache import (_field, _text_field, cell_strings, good_meg_channels, inversion_slot,
                          mat_string, open_spm_header, validate_slot_index)
from ._mat_sparse import h5_to_csc


def _registration_matrix(struct, field):
    value = _field(struct, field)
    if value is None or np.shape(value) != (4, 4):
        raise ValueError(f"Missing or invalid {field} in SPM registration.")
    # h5py exposes the dimensions of MATLAB v7.3 arrays in reverse order. The
    # conventional MAT reader has already restored the matrix to MATLAB orientation.
    return value[()].T if isinstance(value, h5py.Dataset) else np.asarray(value)


def _spm_reference(header, inversion_idx):
    """Read a slot's forward structure, active recording units and MRI registration."""
    slot = inversion_slot(header, inversion_idx)
    forward = _field(slot, 'forward')
    registration = _field(slot, 'datareg')
    if (_text_field(forward, 'modality') != 'MEG' or
            _text_field(registration, 'modality') != 'MEG'):
        raise ValueError(f"Coregister MEG data in SPM inversion slot {inversion_idx} first.")
    # Use the active montage's good channels in recording order. The forward
    # sensor table can include other channels or list them in a different order.
    pairs = good_meg_channels(header, with_units=True)
    if not pairs or len(dict(pairs)) != len(pairs):
        raise ValueError("Expected unique good MEG channel labels in the SPM registration.")
    # The mesh affine takes native MRI points into MNI coordinates; fromMNI
    # continues into this recording's head frame. The rightmost map acts first.
    affine = (_registration_matrix(registration, 'fromMNI')
              @ _registration_matrix(_field(slot, 'mesh'), 'Affine'))
    return forward, dict(pairs), affine


def mat_strings(cells):
    """Read the ordered strings in a MATLAB cellstr from either MAT format."""
    if isinstance(cells, h5py.Dataset):
        if h5py.check_dtype(ref=cells.dtype) is None:
            return (mat_string(cells),)
        return tuple(mat_string(cells.file[ref]) for ref in cells[()].ravel())
    return cell_strings(np.atleast_1d(cells))


def index_labels(available, required):
    """Match physical channel rows by label without accepting duplicate entries."""
    lookup = {name: i for i, name in enumerate(available)}
    if len(lookup) != len(available) or not set(required).issubset(lookup):
        raise ValueError("Channel labels are duplicated or missing in the SPM files.")
    return [lookup[name] for name in required]


def _mat_matrix(node):
    """Return the logical MATLAB rows and columns for a dense or sparse matrix."""
    if isinstance(node, h5py.Dataset):
        return node[()].T
    if isinstance(node, h5py.Group):
        return h5_to_csc(node).toarray()
    return node.toarray() if issparse(node) else np.asarray(node)


def spm_sensor_geometry(forward, labels):
    """Read measured head-frame pickup coils and signed channel weights in SI."""
    sensors = _field(forward, 'sensors')
    if _text_field(sensors, 'unit') != 'm':
        raise ValueError("Expected SPM's physical sensor coil coordinates in metres.")
    sensor_labels = mat_strings(_field(sensors, 'label'))
    rows = index_labels(sensor_labels, labels)
    sensor_units = mat_strings(_field(sensors, 'chanunit'))
    if (len(sensor_units) != len(sensor_labels) or
            any(sensor_units[i] not in ('T', 'T/m') for i in rows)):
        raise ValueError("Expected SI-calibrated MEG sensor coil weights.")

    pos = np.atleast_2d(_mat_matrix(_field(sensors, 'coilpos')))
    ori = np.atleast_2d(_mat_matrix(_field(sensors, 'coilori')))
    transfer = _mat_matrix(_field(sensors, 'tra'))
    shape = (len(sensor_labels), len(pos))
    # scipy.io.loadmat squeezes singleton axes in classic MAT headers. Restore
    # the channel-by-coil axes for a lone channel or pickup, including a 1x1.
    if transfer.ndim < 2 and 1 in shape and transfer.size == shape[0] * shape[1]:
        transfer = transfer.reshape(shape)
    if pos.shape[1] != 3 or ori.shape != pos.shape or transfer.shape != shape:
        raise ValueError("SPM coil coordinates, orientations and channel weights must agree.")
    return {'coilpos': pos, 'coilori': ori, 'tra': transfer[rows, :]}


def fem_recording_geometry(reference, inversion_idx=0):
    """Read the FEM's measured MEG geometry directly after SPM coregistration."""
    validate_slot_index(inversion_idx)
    with open_spm_header(str(reference)) as header:
        forward, units, affine = _spm_reference(header, inversion_idx)
        if np.asarray(_field(forward, 'siunits')).item() != 1:
            raise ValueError("A single MEG registration with physical sensor units is required.")
        labels = tuple(units)
        # These fields are all available before spm_eeg_lgainmat is called:
        # channel order and units from the recording, coil weights from the slot.
        return labels, tuple(units.values()), spm_sensor_geometry(forward, labels), affine


def _mne_head_transform(info, frame):
    """Convert FIFF's CTF-head to MNE-head transform to the reverse direction in mm."""
    from mne.io.constants import FIFF  # pylint: disable=import-outside-toplevel

    ctf = None if info is None else info.get('ctf_head_t')
    # A sensor array may come from a CTF instrument even when its file uses another
    # head convention. Require the named SPM frame and both FIFF frame IDs.
    if (frame != 'ctf' or ctf is None or ctf['from'] != FIFF.FIFFV_MNE_COORD_CTF_HEAD
            or ctf['to'] != FIFF.FIFFV_COORD_HEAD):
        raise ValueError("Supply mne_head_to_head_mm for this registration. Automatic alignment "
                         "requires an SPM CTF head frame and the acquisition's ctf_head_t.")
    # FIFF stores CTF head -> MNE head in metres. We need the inverse for points
    # expressed in mm, so only the translation changes scale. The rotation is unitless.
    head = np.linalg.inv(ctf['trans'])
    head[:3, 3] *= 1000
    return head


def mne_registration(reference, info, mri_to_head_mm, mne_head_to_head_mm, channel_units, *,
                     inversion_idx=0):
    """Resolve and cross-check registration from an SPM header, operator, or explicit geometry."""
    validate_slot_index(inversion_idx)
    frame = None
    if reference is not None:
        if isinstance(reference, (str, PathLike)):
            with open_spm_header(str(reference)) as header:
                forward, units, affine = _spm_reference(header, inversion_idx)
                frame = _text_field(_field(forward, 'sensors'), 'coordsys').lower()
        else:
            # The reference operator supplies row units and the registered MRI affine.
            # Deriving the MNE head transform separately requires a named head frame.
            try:
                units = dict(zip(reference.channel_names, reference.channel_units))
                affine = reference.mri_to_head_mm
            except AttributeError as exc:
                raise TypeError("reference must be a registered SPM dataset or ForwardOperator.") \
                    from exc
        # Inputs supplied alongside a reference must describe the same registration.
        # Matching labels by name allows the caller to choose a different row order.
        if channel_units is not None and dict(channel_units) != units:
            raise ValueError("Requested channel labels or units differ from the reference.")
        if mri_to_head_mm is not None and not np.allclose(
                mri_to_head_mm, affine, rtol=0, atol=1e-6):
            raise ValueError("The supplied MRI-to-head transform differs from the reference.")
        channel_units = units if channel_units is None else channel_units
        mri_to_head_mm = affine
    if channel_units is None or mri_to_head_mm is None:
        raise ValueError("Supply a registered reference, or channel_units and mri_to_head_mm.")
    if mne_head_to_head_mm is None:
        mne_head_to_head_mm = _mne_head_transform(info, frame)
    return channel_units, mri_to_head_mm, mne_head_to_head_mm
