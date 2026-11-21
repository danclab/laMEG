"""Read the recording geometry needed to align an MNE forward with SPM."""
from os import PathLike

import h5py
from mne.io.constants import FIFF
import numpy as np

from ._gain_cache import (_field, _text_field, good_meg_channels, inversion_slot,
                          open_spm_header, validate_slot_index)


def _registration_matrix(struct, field):
    value = _field(struct, field)
    if value is None or np.shape(value) != (4, 4):
        raise ValueError(f"Missing or invalid {field} in SPM registration.")
    return value[()].T if isinstance(value, h5py.Dataset) else np.asarray(value)


def _spm_reference(reference, inversion_idx):
    """Obtain units and registration from the header, including active montage channels."""
    with open_spm_header(str(reference)) as header:
        slot = inversion_slot(header, inversion_idx)
        forward = _field(slot, 'forward')
        registration = _field(slot, 'datareg')
        if (_text_field(forward, 'modality') != 'MEG' or
                _text_field(registration, 'modality') != 'MEG'):
            raise ValueError(f"Coregister MEG data in SPM inversion slot {inversion_idx} first.")
        pairs = good_meg_channels(header, with_units=True)
        if not pairs or len(dict(pairs)) != len(pairs):
            raise ValueError("Expected unique good MEG channel labels in the SPM registration.")
        affine = (_registration_matrix(registration, 'fromMNI')
                  @ _registration_matrix(_field(slot, 'mesh'), 'Affine'))
        frame = _text_field(_field(forward, 'sensors'), 'coordsys').lower()
        return dict(pairs), affine, frame


def _mne_head_transform(info, frame):
    """Convert FIFF's CTF-head to MNE-head transform to the reverse direction in mm."""
    ctf = None if info is None else info.get('ctf_head_t')
    if (frame != 'ctf' or ctf is None or ctf['from'] != FIFF.FIFFV_MNE_COORD_CTF_HEAD
            or ctf['to'] != FIFF.FIFFV_COORD_HEAD):
        raise ValueError("Supply mne_head_to_head_mm for this registration. Automatic alignment "
                         "requires an SPM CTF head frame and the acquisition's ctf_head_t.")
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
            units, affine, frame = _spm_reference(reference, inversion_idx)
        else:
            try:
                units = dict(zip(reference.channel_names, reference.channel_units))
                affine = reference.mri_to_head_mm
            except AttributeError as exc:
                raise TypeError("reference must be a registered SPM dataset or ForwardOperator.") \
                    from exc
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
