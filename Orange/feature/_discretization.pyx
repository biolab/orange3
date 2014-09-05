#cython: embedsignature=True

import numpy
cimport numpy as np
import cython

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
def join_contingency(contingency):
    """
    Join contingency list into a single ordered distribution.
    """
    cdef Py_ssize_t k = len(contingency)
    cdef np.ndarray[np.float64_t, ndim=1] values = numpy.r_[tuple(contingency[i][0] for i in range(k))]
    cdef np.ndarray[np.float64_t, ndim=2] I = numpy.zeros((len(values), k), dtype=numpy.float64)
    cdef Py_ssize_t i, j, start, span, pos
    start = 0
    
    for i in range(k):
        counts = contingency[i][1]
        span = len(counts)
        I[start: start + span, i] = contingency[i][1]
        start += span

    cdef np.ndarray[np.int_t, ndim=1] sort_ind = values.argsort()

    #indexing operations are slower here
    #cdef np.ndarray[np.float64_t, ndim=1] values2 = values[sort_ind]
    cdef np.ndarray[np.float64_t, ndim=1] values2 = numpy.zeros(sort_ind.shape[0], dtype=numpy.float64)
    for i in range(sort_ind.shape[0]):
        values2[i] = values[sort_ind[i]]
    
    #cdef np.ndarray[np.float64_t, ndim=2] I2 = I[sort_ind, :]
    cdef np.ndarray[np.float64_t, ndim=2] I2 = numpy.zeros((sort_ind.shape[0], k), dtype=numpy.float64)
    for i in range(sort_ind.shape[0]):
        for j in range(k):
            I2[i, j] = I[sort_ind[i], j]

    cdef np.float64_t last = float("NaN")
    cdef Py_ssize_t iv = -1
    for i in range(values2.shape[0]):
        if last != values2[i]:
            iv += 1
            last = values2[i]
            values2[iv] = last
            #I2[iv,:] = I2[i,:] #expanded for speed
            for j in range(k):
                I2[iv,j] = I2[i,j]
        else:
            #I2[iv] += I2[i] #expanded for speed
            for j in range(k):
                I2[iv,j] += I2[i,j]

    return values2[:iv+1],I2[:iv+1]
        
