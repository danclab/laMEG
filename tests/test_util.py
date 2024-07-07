from lameg.util import *
def test_check_many():
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
    assert(check_many(multiple, target, func='any'))

    multiple = ['x','y']
    target = 'z'
    assert (not check_many(multiple, target, func='any'))

    multiple = ['x', 'x']
    target = 'x'
    assert (check_many(multiple, target, func='all'))

    multiple = ['x', 'y']
    target = 'x'
    assert (not check_many(multiple, target, func='all'))

    multiple = ['x', 'y']
    target = 'z'
    assert (not check_many(multiple, target, func='all'))