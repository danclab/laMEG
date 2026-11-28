"""Sensor rows, physical units and pickup coils for MNE-Python MEG forwards."""

import numpy as np
from mne.forward._make_forward import _create_meg_coils
from mne.io.compensator import get_current_comp, make_compensator
from mne.io.constants import FIFF, FWD

from ._forward_units import si_gain_scale


def _row_scale(channel, unit):
    """Convert SI gain per A m to the requested channel unit per nA m."""
    scale = si_gain_scale(unit)
    # Use the recorded physical unit to distinguish field from field gradient. CTF axial
    # gradiometers can report the difference between pickups in field units.
    expected = FIFF.FIFF_UNIT_T_M if '/' in unit else FIFF.FIFF_UNIT_T
    if channel['unit'] != expected:
        raise ValueError(f"Unit {unit!r} differs in dimension from {channel['ch_name']}.")
    return scale


def _device_to_head(info):
    transform = info.get('dev_head_t')
    if (transform is None or transform['from'] != FIFF.FIFFV_COORD_DEVICE
            or transform['to'] != FIFF.FIFFV_COORD_HEAD):
        raise ValueError("An MNE device-to-head transform is required for MEG pickup coils.")
    return transform


def _compare_infos(recorded, acquisition, names):
    """Check geometry and calibration shared by a light forward Info and acquisition Info."""
    # Identical labels can occur in different recordings. Check the head pose as well as
    # each primary coil before using reference metadata from the supplied acquisition.
    if not np.allclose(_device_to_head(recorded)['trans'], _device_to_head(acquisition)['trans'],
                       rtol=0, atol=1e-7):
        raise ValueError("Acquisition Info and forward have different device-to-head transforms.")
    for name in names:
        if name not in acquisition['ch_names']:
            raise ValueError(f"The acquisition Info lacks MEG channel {name!r}.")
        old = recorded['chs'][recorded['ch_names'].index(name)]
        new = acquisition['chs'][acquisition['ch_names'].index(name)]
        if (any(old[key] != new[key] for key in ('coil_type', 'unit', 'kind', 'coord_frame'))
                or not np.allclose(old['loc'], new['loc'], rtol=0, atol=1e-7)):
            raise ValueError(f"Acquisition Info disagrees with the forward's MEG coil {name!r}.")


def _compensated_rows(recorded, acquisition, rows, grade, ignore_ref):
    """Reconstruct MNE's actual reference montage using calibrated acquisition coefficients."""
    # MNE keeps the compensation grade on the channel even when ignore_ref=True.
    # The forward file omits that solver setting and its reference channel Info.
    if ignore_ref is not False:
        raise ValueError("Compensated MNE data require a forward computed with reference "
                         "compensation. Specify its original setting, ignore_ref=False.")
    if not any(ch['kind'] == FIFF.FIFFV_REF_MEG_CH for ch in acquisition['chs']):
        raise ValueError("A compensated forward requires its original acquisition Info "
                         "with reference MEG channels and compensation coefficients.")
    if get_current_comp(acquisition) != grade:
        raise ValueError("Acquisition Info and forward have different compensation grades.")
    # Each selected row maps the full set of uncompensated acquisition channels to one
    # compensated signal. Keep the reference columns and their signed coefficients.
    weights = make_compensator(acquisition, 0, grade)[rows]
    # In-memory forwards can retain some coefficients; FIF generally drops them.
    for comp in recorded.get('comps', ()):
        if comp['kind'] == grade:
            actual = comp['data']
            idx = [acquisition['ch_names'].index(n) for n in actual['col_names']]
            for row, selected in enumerate(rows):
                name = acquisition['ch_names'][selected]
                if name in actual['row_names']:
                    coeff = actual['data'][actual['row_names'].index(name)]
                    # MNE applies I - C in physical channel units.
                    expected = (np.asarray(idx) == selected).astype(float) - coeff
                    if not np.allclose(weights[row, idx], expected, rtol=1e-7, atol=1e-12):
                        raise ValueError("Acquisition compensation differs from the MNE forward.")
    return weights


def _require_ctf_coils(acquisition, indices):
    """Limit nominal-to-measured pickup matching to CTF primaries and references."""
    supported = {
        (FIFF.FIFFV_MEG_CH, FIFF.FIFFV_COIL_CTF_GRAD),
        (FIFF.FIFFV_REF_MEG_CH, FIFF.FIFFV_COIL_CTF_REF_MAG),
        (FIFF.FIFFV_REF_MEG_CH, FIFF.FIFFV_COIL_CTF_REF_GRAD),
        (FIFF.FIFFV_REF_MEG_CH, FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD)}
    for idx in indices:
        channel = acquisition['chs'][idx]
        # FIFF stores the active compensation grade in the upper 16 bits.
        if (channel['kind'], int(channel['coil_type']) & 0xffff) not in supported:
            raise ValueError(f"Nominal CTF coil matching requires CTF sensors; "
                             f"{channel['ch_name']!r} has a different coil type.")


def mne_channel_geometry(fwd, channel_names, channel_units, head_transform, info=None,
                         ignore_ref=None, allow_nominal_ctf_coils=False):
    """Return gain row indices, numeric gain scales and physical coil geometry.

    head_transform maps MNE head metres to operator head metres. MNE's
    point-accuracy coil definitions represent the physical pickup loops for
    registration, independent of the quadrature used to calculate the gain.
    For compensated data, pass the complete Info and ignore_ref=False as used
    for the original solve. The matrix alone cannot verify this setting.
    """
    recorded = fwd['info']
    acquisition = recorded if info is None else info
    # The gain, the acquisition Info and the recipient recording may have different
    # channel orders. Gain rows are selected by label in the recipient's order.
    row_names = list(fwd['sol']['row_names'])
    if len(row_names) != fwd['sol']['data'].shape[0]:
        raise ValueError("The MNE gain row count differs from its channel labels.")
    if len(set(row_names)) != len(row_names):
        raise ValueError("The MNE gain contains duplicate channel labels.")
    rows = []
    scales = []
    for name, unit in zip(channel_names, channel_units):
        if name not in row_names or name not in recorded['ch_names']:
            raise ValueError(f"MEG channel {name!r} is absent from the MNE forward.")
        channel = recorded['chs'][recorded['ch_names'].index(name)]
        if (channel['kind'] != FIFF.FIFFV_MEG_CH or name in recorded.get('bads', ())
                or name in acquisition.get('bads', ())):
            raise ValueError(f"Channel {name!r} must be a good primary MEG channel.")
        scales.append(_row_scale(channel, unit))
        rows.append(row_names.index(name))
    if info is not None:
        _compare_infos(recorded, acquisition, channel_names)
    # Compensation coefficients use acquisition indices. The rows returned to the caller
    # index the gain instead, so look up the same channels in both places.
    selected = [acquisition['ch_names'].index(name) for name in channel_names]
    grade = get_current_comp(recorded)
    if grade:
        mixing = _compensated_rows(recorded, acquisition, selected, grade, ignore_ref)
    else:
        # At grade zero each output takes its own primary channel with weight one.
        mixing = np.zeros((len(rows), len(acquisition['chs'])))
        mixing[np.arange(len(rows)), selected] = 1

    # Stable order across FIF files with the same hardware but different row order.
    active = sorted(np.flatnonzero(np.any(mixing != 0, axis=0)),
                    key=acquisition['ch_names'].__getitem__)
    if allow_nominal_ctf_coils:
        _require_ctf_coils(acquisition, set(selected).union(active))
    channels = [acquisition['chs'][idx] for idx in active]
    if any(ch['coord_frame'] != FIFF.FIFFV_COORD_DEVICE for ch in channels):
        raise ValueError("Expected MEG coil locations in MNE device coordinates.")
    # Expand each active channel into its physical pickup centres and axes in MNE head
    # coordinates. A single gradiometer channel, for example, contributes two pickups.
    coils = _create_meg_coils(channels, FWD.COIL_ACCURACY_POINT, _device_to_head(acquisition))
    rotation = head_transform[:3, :3]
    # Translate only the positions; a coil's signed sensitivity axis undergoes rotation.
    # For each pickup, multiply its within-channel weight by that channel's contribution
    # to each recorded signal. The resulting tra has one row per output and one column per pickup.
    geometry = {
        'coilpos': np.vstack([c['rmag'] for c in coils]) @ rotation.T + head_transform[:3, 3],
        'coilori': np.vstack([c['cosmag'] for c in coils]) @ rotation.T,
        'tra': np.hstack([mixing[:, idx, None] * coil['w'] for idx, coil in zip(active, coils)])
    }
    return np.asarray(rows), np.asarray(scales), geometry
