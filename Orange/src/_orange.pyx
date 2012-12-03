#cython: embedsignature=True

import numpy
cimport numpy as np
import cython

from numpy cimport NPY_FLOAT64 as NPY_float64

@cython.boundscheck(False)
@cython.wraparound(False)
def valuecount(np.ndarray[np.float64_t, ndim=2] a):
    """
    Count the occurrences of each value. Function requires a 2xN array
    of type float64; the first row must contain *sorted values* and the
    second contains their weights. The function 'compresses' the array
    by merging consecutive columns with the same value in the first row,
    and adding the corresponding weights in the second row.

    >>> a = np.array([[1, 1, 2, 3, 3], [0.1, 0.2, 0.3, 0.4, 0.5]])
    >>> _orange.valuecount(a)
    [[ 1.   2.   3. ]
     [ 0.3  0.3  0.9]]
    """
    cdef np.npy_intp *dim
    dim = np.PyArray_DIMS(a)
    if dim[0] != 2:
        raise ValueError("value count requires an array with shape (2, N)")
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
