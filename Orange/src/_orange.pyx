#cython: embedsignature=True

import numpy
cimport numpy as np
import cython

from numpy cimport NPY_FLOAT64 as NPY_float64

@cython.boundscheck(False)
@cython.wraparound(False)
def valuecount(np.ndarray[np.float64_t, ndim=2] a):
    '''Count the occurrences of each value'''
    cdef np.npy_intp *dim
    dim = np.PyArray_DIMS(a)
    if dim[0] != 2:
        raise ValueError("value count requires array with shape (2, N)")
    cdef Py_ssize_t N = dim[1]

    cdef Py_ssize_t src
    for src in range(1, N):
        if a[0, src] == a[0, src-1]:
            break
    else:
        return a
    cdef int dst = src - 1
    for src in range(src, N):
        if a[0, src] != a[0, src]:
            break
        if a[0, src] == a[0, dst]:
            a[1, dst] += a[1, src]
        else:
            dst += 1
            a[0, dst] = a[0, src]
            a[1, dst] = a[1, src]
    return a[:, :dst + 1]
