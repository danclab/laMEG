# csurf.pyx
import numpy as np
cimport numpy as cnp
from scipy.sparse import coo_matrix

cdef extern from "float.h":
    double DBL_MAX

cdef void fill_adjacency_matrix(cnp.ndarray[int, ndim=2] faces, int num_vertices, cnp.ndarray[int, ndim=1] rows, cnp.ndarray[int, ndim=1] cols):
    cdef int idx = 0
    for i in range(faces.shape[0]):
        for j in range(3):
            rows[idx] = faces[i, j]
            cols[idx] = faces[i, (j + 1) % 3]
            idx += 1
            rows[idx] = faces[i, j]
            cols[idx] = faces[i, (j + 2) % 3]
            idx += 1

cdef void relax_edges(int u, cnp.ndarray[double, ndim=1] dist, cnp.ndarray[int, ndim=1] indices, cnp.ndarray[int, ndim=1] indptr, cnp.ndarray[double, ndim=1] data, int[:] remaining):
    cdef int v
    cdef double alt
    for v in indices[indptr[u]:indptr[u + 1]]:
        alt = dist[u] + data[indptr[u]:indptr[u + 1]][indices[indptr[u]:indptr[u + 1]] == v]
        if alt < dist[v]:
            dist[v] = alt

cdef cnp.ndarray[double, ndim=1] compute_geodesic_distances_internal(object D_csr, cnp.ndarray[int, ndim=1] source_indices, double max_dist):
    cdef int num_vertices = D_csr.shape[0]
    cdef cnp.ndarray[double, ndim=1] dist = np.full(num_vertices, np.inf)
    cdef cnp.ndarray[int, ndim=1] D_indices = D_csr.indices
    cdef cnp.ndarray[int, ndim=1] D_indptr = D_csr.indptr
    cdef cnp.ndarray[double, ndim=1] D_data = D_csr.data

    cdef int[:] remaining = np.arange(num_vertices, dtype=np.int32)
    cdef int remaining_size = remaining.size
    cdef int[:] new_remaining
    cdef int u, v, idx

    for src in source_indices:
        dist[src] = 0.0

    while remaining_size > 0:
        u = remaining[np.argmin(dist[remaining])]
        if dist[u] > max_dist:
            break
        relax_edges(u, dist, D_indices, D_indptr, D_data, remaining)

        # Efficiently remove 'u' from 'remaining'
        new_remaining = np.empty(remaining_size - 1, dtype=np.int32)
        idx = 0
        for v in remaining:
            if v != u:
                new_remaining[idx] = v
                idx += 1
        remaining = new_remaining
        remaining_size -= 1

    return dist


def compute_mesh_adjacency(faces):
    num_vertices = np.max(faces) + 1
    rows = np.empty(6 * faces.shape[0], dtype=np.int32)
    cols = np.empty(6 * faces.shape[0], dtype=np.int32)
    fill_adjacency_matrix(faces, num_vertices, rows, cols)
    data = np.ones(len(rows), dtype=np.int32)
    return coo_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices)).tocsr()

def compute_geodesic_distances(object D_csr, source_indices, max_dist=np.inf):
    return compute_geodesic_distances_internal(D_csr, np.array(source_indices, dtype=np.int32), max_dist)