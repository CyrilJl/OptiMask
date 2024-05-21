# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np

ctypedef np.int64_t DTYPE_INT_t

cpdef np.ndarray[DTYPE_INT_t, ndim=1] groupby_max(np.ndarray[DTYPE_INT_t, ndim=1] a, np.ndarray[DTYPE_INT_t, ndim=1] b):
    cdef int n = a.max()
    cdef np.ndarray[DTYPE_INT_t, ndim=1] ret = np.zeros(n + 1, dtype=np.int64)
    cdef int i
    for i in range(a.shape[0]):
        ret[a[i]] = max(ret[a[i]], b[i] + 1)
    return ret

cpdef bint is_decreasing(np.ndarray[DTYPE_INT_t, ndim=1] x):
    cdef int n = x.shape[0]
    cdef int i
    for i in range(n - 1):
        if x[i] < x[i + 1]:
            return False
    return True

cpdef np.ndarray[DTYPE_INT_t, ndim=1] permutation_index(np.ndarray[DTYPE_INT_t, ndim=1] p):
    cdef int i
    cdef np.ndarray[DTYPE_INT_t, ndim=1] s = np.empty(p.size, dtype=np.int64)
    for i in range(p.size):
        s[p[i]] = i
    return s
