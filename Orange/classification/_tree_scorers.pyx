#cython: embedsignature=True

import numpy
cimport numpy as np
import cython

from libc.stdlib cimport calloc, free

cdef extern from "math.h":
    double log(double x)

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
def find_threshold_entropy(np.ndarray[np.float64_t, ndim=1] x,
                           np.ndarray[np.float64_t, ndim=1] y,
                           np.ndarray[np.int64_t, ndim=1] idx,
                           int n_classes, int min_leaf):
    cdef int *distr = <int *>calloc(2 * n_classes, sizeof(int))
    cdef Py_ssize_t i, j
    cdef float entro, class_entro, best_entro
    cdef int p, curr_y
    cdef int best_idx = 0
    cdef int N = idx.shape[0]

    # Initial split (min_left on the left)
    for i in range(min_leaf - 1):  # one will be added in the loop
        # without temporary int, cython uses PyObjects here
        curr_y = int(y[idx[i]])
        distr[n_classes + curr_y] += 1
    for i in range(min_leaf - 1, N):
        distr[int(y[idx[i]])] += 1

    # Compute class entropy
    class_entro = N * log(N)
    for j in range(n_classes):
        p = distr[j] + distr[j + n_classes]
        if p:
            class_entro -= p * log(p)
    best_entro = class_entro

    # Loop through
    for i in range(min_leaf - 1, N - min_leaf):
        curr_y = int(y[idx[i]])
        distr[curr_y] -= 1
        distr[n_classes + curr_y] += 1
        if curr_y != y[idx[i + 1]] and x[idx[i]] != x[idx[i + 1]]:
            entro = (i + 1) * log(i + 1) + (N - i - 1) * log(N - i - 1)
            for j in range(2 * n_classes):
                if distr[j]:
                    entro -= distr[j] * log(distr[j])
            if entro < best_entro:
                best_entro = entro
                best_idx = i
    free(distr)
    return (class_entro - best_entro) / N / log(2), x[idx[best_idx]]


@cython.boundscheck(False)
@cython.wraparound(False)
def find_binarization_entropy(np.ndarray[np.float64_t, ndim=2] cont,
                              np.ndarray[np.float64_t, ndim=1] class_distr,
                              np.ndarray[np.float64_t, ndim=1] val_distr,
                              int min_leaf):
    cdef int n_classes = cont.shape[0]
    cdef int n_values = cont.shape[1]
    cdef float *distr = <float *>calloc(2 * n_classes, sizeof(float))
    cdef float *mfrom
    cdef float *mto
    cdef float left, right
    cdef int i, change, to_right, allowed, m
    cdef int best_mapping, move, mapping, previous
    cdef float entro, class_entro, best_entro
    cdef float N = 0

    class_entro = 0
    for i in range(n_classes):
        distr[i + n_classes] = 0
        distr[i] = class_distr[i]
        if class_distr[i] > 0:
            N += class_distr[i]
            class_entro -= class_distr[i] * log(class_distr[i])
    class_entro += N * log(N)
    best_entro = class_entro
    left = N
    right = 0

    previous = 0
    # Gray code
    for m in range(1, 1 << (n_values - 1)):
        # What moves where
        mapping = m ^ (m >> 1)
        change = mapping ^ previous
        to_right = change & mapping
        for move in range(n_values):
            if change & 1:
                break
            change = change >> 1
        previous = mapping

        if to_right:
            left -= val_distr[move]
            right += val_distr[move]
            mfrom = distr
            mto = distr + n_classes
        else:
            left += val_distr[move]
            right -= val_distr[move]
            mfrom = distr + n_classes
            mto = distr

        allowed = left >= min_leaf and right >= min_leaf
        # Move distribution to the other side and
        # compute entropy by the way, if the split is allowed
        entro = 0
        for i in range(n_classes):
            mfrom[i] -= cont[i, move]
            mto[i] += cont[i, move]
            if allowed:
                if mfrom[i]:
                    entro -= mfrom[i] * log(mfrom[i])
                if mto[i]:
                    entro -= mto[i] * log(mto[i])

        if allowed:
            entro += left * log(left) + right * log(right)
            if entro < best_entro:
                best_entro = entro
                best_mapping = mapping
    free(distr)
    return (class_entro - best_entro) / N / log(2), best_mapping


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_predictions(np.ndarray[np.float64_t, ndim=2] X,
                        np.ndarray[np.int32_t, ndim=1] code,
                        np.ndarray[np.float64_t, ndim=2] class_distrs,
                        np.ndarray[np.float64_t, ndim=1] thresholds):
    cdef int node_ptr, node_idx, i, val_idx
    cdef np.float64_t val
    cdef np.ndarray[np.float64_t, ndim=2] predictions = \
        numpy.empty((X.shape[0], class_distrs.shape[1]), dtype=numpy.float64)

    for i in range(X.shape[0]):
        node_ptr = 0
        while code[node_ptr]:
            val = X[i, code[node_ptr + 2]]
            if npy_isnan(val):
                break
            if code[node_ptr] == 3:
                node_idx = code[node_ptr + 1]
                val_idx = int(val > thresholds[node_idx])
            else:
                val_idx = int(val)
            node_ptr = code[node_ptr + 3 + val_idx]
        node_idx = code[node_ptr + 1]
        predictions[i] = class_distrs[node_idx]
    return predictions
