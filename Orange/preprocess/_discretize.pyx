#cython: embedsignature=True
#cython: language_level=3

import numpy
cimport numpy as np
import cython
from libc.math cimport log
from numpy cimport NPY_FLOAT64 as NPY_float64

@cython.boundscheck(False)
@cython.wraparound(False)
def split_eq_freq(np.ndarray[np.float64_t, ndim=2] dist not None, int n):

    cdef int llen = dist.shape[1]

    if n >= llen: #n is greater than distributions
        return [(v1+v2)/2 for v1,v2 in zip(dist[0], dist[0][1:])]

    cdef np.float64_t N = dist[1].sum()
    cdef int toGo = n
    cdef np.float64_t inthis = 0
    cdef np.float64_t prevel = -1
    cdef np.float64_t inone = N/toGo
    points = []

    cdef Py_ssize_t i
    cdef np.float64_t v
    cdef np.float64_t k
    cdef np.float64_t vn

    for i in range(llen):
        v = dist[0,i]
        k = dist[1,i]
        if toGo <= 1:
            break
        inthis += k
        if inthis < inone or i == 0: 
            prevel = v
        else: #current count exceeded
            if i < llen - 1 and inthis - inone < k / 2:
                #exceeded for less than half the current count:
                #split after current
                vn = dist[0,i+1]
                points.append((vn + v)/2)
                N -= inthis
                inthis = 0
                prevel = vn
            else:
                #split before the current value
                points.append((prevel + v)/2)
                N -= inthis - k
                inthis = k
                prevel = v
            toGo -= 1
            if toGo:
                inone = N/toGo
    return points


@cython.wraparound(False)
@cython.boundscheck(False)
def entropy_normalized1(np.ndarray[np.float64_t, ndim=1] D):
    """
    Compute entropy of distribution in `D` (must be normalized).
    """
    cdef np.float64_t R = 0.
    cdef Py_ssize_t j
    cdef np.float64_t t
    cdef np.float64_t log2 = 1./log(2.)
    for j in range(D.shape[0]):
        t = D[j]
        if t > 0.:
            if t > 1.0: t = 1.0
            R -= t*log(t)*log2
    return R


@cython.wraparound(False)
@cython.boundscheck(False)
def entropy_normalized2(np.ndarray[np.float64_t, ndim=2] D):
    """
    Compute entropy of distributions in `D`.
    Rows in `D` must be a distribution (i.e. sum to 1.0 over `axis`).
    """
    cdef np.ndarray[np.float64_t, ndim=1] R = numpy.zeros(D.shape[0])
    cdef Py_ssize_t i,j
    cdef np.float64_t t
    cdef np.float64_t log2 = 1./log(2.)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            t = D[i,j]
            if t > 0.:
                if t > 1.0: t = 1.0
                R[i] -= t*log(t)*log2
    return R
