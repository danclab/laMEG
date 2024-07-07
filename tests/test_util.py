"""
This module contains the unit tests for the `utils` module from the `lameg` package.
"""
from lameg.util import check_many

def test_check_many():
    """
    Test the `check_many` function to verify its response to different scenarios and parameters.

    The function is tested to:
    - Throw a ValueError when `target` contains characters not in `multiple` (when applicable).
    - Correctly return True if any or all elements in `multiple` are in `target` based on the
      `func` parameter.
    - Correctly return False if not all or none of the elements in `multiple` are in `target`,
      based on the `func` parameter.

    Tests include:
    - Single element in `multiple` that is part of `target`.
    - Multiple elements in `multiple` with partial inclusion in `target`.
    - Multiple elements in `multiple` fully included in `target`.
    - No elements from `multiple` included in `target`.

    Args:
        None

    Returns:
        None
    """

    multiple=['x']
    target='xy'
    val_error=False
    try:
        check_many(multiple,target)
    except ValueError:
        val_error=True
    assert(val_error)

    multiple=['x','y']
    target='x'
    assert check_many(multiple, target, func='any')

    multiple = ['x','y']
    target = 'z'
    assert not check_many(multiple, target, func='any')

    multiple = ['x', 'x']
    target = 'x'
    assert check_many(multiple, target, func='all')

    multiple = ['x', 'y']
    target = 'x'
    assert not check_many(multiple, target, func='all')

    multiple = ['x', 'y']
    target = 'z'
    assert not check_many(multiple, target, func='all')
