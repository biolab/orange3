#cython: embedsignature=True

import numpy
cimport numpy as np
import cython

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

from numpy cimport NPY_FLOAT64 as NPY_float64

@cython.boundscheck(False)
@cython.wraparound(False)
def valuecount(np.ndarray[np.float64_t, ndim=2] a not None):
    """Count the occurrences of each value.

    It does so in-place, on a 2-d array of shape (2, N); the first row
    contains values and the second contains weights (1's, if unweighted).
    The array should be sorted by the elements in the first row (e.g.
    a.sort(axis=0)). The function 'compresses' the array by merging
    consecutive columns with the same value in the first row, and adding
    the corresponding weights in the second row.

    Examples
    --------
    >>> a = np.array([[1, 1, 2, 3, 3], [0.1, 0.2, 0.3, 0.4, 0.5]])
    >>> _orange.valuecount(a)
    [[ 1.   2.   3. ]
     [ 0.3  0.3  0.9]]

    """
    cdef np.npy_intp *dim
    dim = np.PyArray_DIMS(a)
    if dim[0] != 2:
        raise ValueError("valuecount expects an array with shape (2, N)")
    cdef Py_ssize_t N = dim[1]

    cdef Py_ssize_t src

    if N == 0 or npy_isnan(a[0, 0]):
        return a[:, :0]

    for src in range(1, N):
        if a[0, src] == a[0, src - 1] or npy_isnan(a[0, src]):
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
