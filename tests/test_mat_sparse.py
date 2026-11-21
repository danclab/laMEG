"""MATLAB v7.3 sparse shape, data layout and row-selection regressions."""
import h5py
import numpy as np
import pytest
from scipy.sparse import csc_matrix, isspmatrix_csc

from lameg import _mat_sparse
from lameg._source_io import (
    _h5_matrix_from_object, _h5_matrix_rows_from_object, _h5_matrix_shape_from_object,
)
from lameg.invert import _read_inversion_results
from lameg.forward import _mat_matrix


def _store(parent, name, array, declared=True):
    """Store the row count and CSC vectors as written by MATLAB."""
    matrix = csc_matrix(array)
    group = parent.create_group(name)
    # jc gives the boundaries between columns, including empty columns. Only
    # the MATLAB_sparse attribute records the full number of rows.
    group['jc'] = matrix.indptr
    if declared:
        group.attrs['MATLAB_sparse'] = np.uint64(matrix.shape[0])
    # MATLAB can omit both coefficient arrays when the matrix is all zero.
    if matrix.nnz or not declared:
        group['data'], group['ir'] = matrix.data, matrix.indices
    return group


def test_native_inverse_and_coils(tmp_path):
    """Native sparse M, U and coil transfer retain trailing rows and physical axes."""
    path = tmp_path / 'inverse.mat'
    # M has two silent sources at the end and U has an unused final mode.
    # Losing either dimension would give the wrong shape for M @ U.
    weighting = np.array([[0, 2, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    reduction = np.array([[1, 2], [3, 4], [0, 0]])
    with h5py.File(path, 'w') as file:
        slot = file.create_group('slot')
        slots = file.create_dataset('D/other/inv', (1, 1), dtype=h5py.ref_dtype)
        slots[0, 0] = slot.ref
        inverse = slot.create_group('inverse')
        inverse['crossF'], inverse['crosserr'] = 5., 1.
        _store(inverse, 'M', weighting)
        modes = inverse.create_dataset('U', (1, 1), dtype=h5py.ref_dtype)
        modes[0, 0] = _store(file, 'reduction', reduction).ref
        # The same coil transfer should decode identically from sparse or
        # dense MATLAB storage. h5py exposes the dense array axes in reverse.
        sparse_tra = _store(file, 'sparse_tra', reduction)
        dense_tra = file.create_dataset('dense_tra', data=reduction.T)
        np.testing.assert_array_equal(_mat_matrix(sparse_tra), _mat_matrix(dense_tra))
    inverse_op = _read_inversion_results(path, return_mu_matrix=True)[-1]
    assert isspmatrix_csc(inverse_op)
    np.testing.assert_array_equal(inverse_op.toarray(), weighting @ reduction)


@pytest.mark.parametrize('n_rows,n_cols', [(5, 3), (4, 0), (0, 3)])
def test_native_empty_sparse(tmp_path, n_rows, n_cols):
    """MATLAB omits data and ir for an all-zero sparse matrix, including empty axes."""
    with h5py.File(tmp_path / 'empty.mat', 'w') as file:
        group = _store(file, 'all_zero', np.zeros((n_rows, n_cols)))
        group.attrs['MATLAB_class'] = np.bytes_('double')
        assert set(group) == {'jc'}
        shape = (n_rows, n_cols)
        assert _h5_matrix_shape_from_object(group, n_cols) == shape
        matrix = _h5_matrix_from_object(group)
        assert isspmatrix_csc(matrix) and matrix.shape == shape and matrix.nnz == 0
        np.testing.assert_array_equal(_mat_matrix(group), np.zeros(shape))
        if n_rows:
            selected = _h5_matrix_rows_from_object(group, [n_rows - 1], n_cols)
            assert selected.shape == (1, n_cols) and selected.nnz == 0


def test_malformed_sparse(tmp_path):
    """Missing coefficients must not silently produce an all-zero matrix."""
    with h5py.File(tmp_path / 'bad.mat', 'w') as file:
        group = _store(file, 'sparse', [[0, 5], [0, 0]])
        # This group promises one coefficient in jc. Neither an impossible
        # row count nor missing coefficient arrays describes a valid zero matrix.
        group.attrs['MATLAB_sparse'] = 0
        with pytest.raises(ValueError):
            _mat_sparse.h5_to_csc(group)
        del group['data']
        with pytest.raises(ValueError, match='Incomplete'):
            _mat_sparse.h5_to_csc(group)
        del group['ir']
        with pytest.raises(ValueError, match='Incomplete'):
            _mat_sparse.h5_to_csc(group)


def test_legacy_missing_dimensions(tmp_path):
    """For older groups the largest stored row remains the recoverable row count."""
    with h5py.File(tmp_path / 'legacy.mat', 'w') as file:
        for array in ([[1., 0], [0, 2], [0, 0]], [[0., 0], [0, 0], [0, 0]]):
            group = _store(file, 'matrix', array, declared=False)
            # Without a declared row count we can only recover rows up to the
            # last stored nonzero. The column boundaries still give both columns.
            rows = 2 if np.any(array) else 0
            assert _mat_sparse.sparse_shape(group) == (rows, 2)
            np.testing.assert_array_equal(_mat_sparse.h5_to_csc(group).toarray(),
                                          np.asarray(array)[:rows])
            del file['matrix']


@pytest.mark.parametrize('transpose', [False, True])
def test_boundaries_and_selection(tmp_path, monkeypatch, transpose):
    """Shape lookup stays lazy and repeated source indices preserve order in both layouts."""
    original = np.arange(10.).reshape(5, 2)
    original[-2:] = 0
    storage = original.T if transpose else original
    # Request a silent vertex and a repeated vertex, out of order. Sparse
    # and dense reads must both return them in exactly this sequence.
    rows = [4, 0, 2, 0]
    with h5py.File(tmp_path / 'window.mat', 'w') as file:
        sparse = _store(file, 'M', storage)
        dense = file.create_dataset('dense', data=storage)
        read = h5py.Dataset.__getitem__
        reads = []
        def track_read(dataset, selection):
            reads.append((dataset.name, selection))
            return read(dataset, selection)

        monkeypatch.setattr(h5py.Dataset, '__getitem__', track_read)
        # A shape lookup should touch only metadata. A dense selection should
        # read just the unique vertices in one operation, on the correct axis.
        assert _h5_matrix_shape_from_object(sparse, 2) == original.shape
        assert not reads
        selected = _h5_matrix_rows_from_object(dense, rows, 2)
        np.testing.assert_array_equal(selected, original[rows])
        assert len(reads) == 1
        _, selection = reads[0]
        np.testing.assert_array_equal(selection[int(transpose)], np.unique(rows))
        selected = _h5_matrix_rows_from_object(sparse, rows, 2)
        assert isspmatrix_csc(selected)
        np.testing.assert_array_equal(selected.toarray(), original[rows])
        with pytest.raises(IndexError):
            _h5_matrix_rows_from_object(sparse, [5], 2)
