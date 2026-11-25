"""Physical channel units used for MEG forward gains."""
import re


MEG_UNIT = re.compile(r'([fpnuµμm]?)T(?:/(m|cm|mm))?')


def si_gain_scale(unit):
    """Scale a gain in T or T/m per A m to the recorded unit per nA m."""
    match = MEG_UNIT.fullmatch(unit)
    if match is None:
        raise ValueError(f"Unsupported MEG channel unit {unit!r}.")
    prefix, length = match.groups()
    field_unit = {'': 1, 'm': 1e-3, 'u': 1e-6, 'µ': 1e-6, 'μ': 1e-6,
                  'n': 1e-9, 'p': 1e-12, 'f': 1e-15}[prefix]
    length_unit = {None: 1, 'm': 1, 'cm': 1e-2, 'mm': 1e-3}[length]
    # A 1 nA m source has a moment of 1e-9 A m. For example, fT/cm scales
    # the field by 1e15 and the gradient denominator by 0.01 m per cm.
    return 1e-9 * length_unit / field_unit
