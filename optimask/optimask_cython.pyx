# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np

cpdef np.ndarray[np.int32_t] groupby_max(np.ndarray[np.int32_t, ndim=1] a, np.ndarray[np.int32_t, ndim=1] b):
    cdef int n = a.max()
    cdef np.ndarray[np.int32_t] ret = np.zeros(n + 1, dtype=np.int32)
    cdef int i
    for i in range(a.shape[0]):
        ret[a[i]] = max(ret[a[i]], b[i])
    return (ret + 1).astype(np.int32)

cpdef bint is_decreasing(np.ndarray[np.int32_t, ndim=1] x):
    cdef int n = x.shape[0]
    cdef int i
    for i in range(n - 1):
        if x[i] < x[i + 1]:
            return False
    return True

def permutation_index(np.ndarray[np.int32_t, ndim=1] p):
    cdef int i
    cdef int[:] s = np.empty(p.size, dtype=np.int32)
    for i in range(p.size):
        s[p[i]] = i
    return np.array(s)
