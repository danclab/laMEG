"""Read MATLAB v7.3 sparse matrices in their logical row-by-column orientation."""
import h5py
import numpy as np
from scipy.sparse import csc_matrix


def is_h5_sparse(node):
    """Recognize native sparse groups and legacy groups without the size attribute."""
    return (isinstance(node, h5py.Group) and 'jc' in node
            and ('MATLAB_sparse' in node.attrs or ('data' in node and 'ir' in node)))


def sparse_shape(group, indices=None):
    """Use the declared size, or infer legacy row counts from nonzero indices."""
    n_columns = group['jc'].size - 1
    if 'MATLAB_sparse' in group.attrs:
        return int(np.asarray(group.attrs['MATLAB_sparse']).item()), n_columns
    if indices is None:
        indices = group['ir'][()]
    return (int(indices.max()) + 1 if indices.size else 0), n_columns


def h5_to_csc(group):
    """Decode CSC storage, including MATLAB's column-pointer-only zero matrices."""
    pointers = group['jc'][()].ravel()
    if 'data' not in group or 'ir' not in group:
        if 'data' in group or 'ir' in group or not pointers.size or np.any(pointers):
            raise ValueError("Incomplete MATLAB sparse coefficient or row index arrays.")
        cls = group.attrs.get('MATLAB_class', b'double')
        dtype = np.bool_ if cls in (b'logical', 'logical') else np.dtype(cls)
        return csc_matrix(sparse_shape(group), dtype=dtype)
    indices = group['ir'][()].ravel()
    data = group['data'][()].ravel()
    matrix = csc_matrix((data, indices, pointers), shape=sparse_shape(group, indices))
    # Check indices against the declared dimensions before consumers densify the matrix.
    matrix.check_format(full_check=True)
    return matrix
