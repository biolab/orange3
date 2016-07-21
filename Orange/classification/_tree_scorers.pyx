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
    cdef unsigned int *distr = <unsigned int *>calloc(2 * n_classes, sizeof(int))
    cdef Py_ssize_t i, j
    cdef double entro, class_entro, best_entro
    cdef unsigned int p, curr_y
    cdef unsigned int best_idx = 0
    cdef unsigned int N = idx.shape[0]

    # Initial split (min_leaf on the left)
    if N <= min_leaf:
        return 0, 0
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
    cdef unsigned int n_classes = cont.shape[0]
    cdef unsigned int n_values = cont.shape[1]
    cdef double *distr = <double *>calloc(2 * n_classes, sizeof(double))
    cdef double *mfrom
    cdef double *mto
    cdef double left, right
    cdef unsigned int i, change, to_right, allowed, m
    cdef unsigned int best_mapping, move, mapping, previous
    cdef double entro, class_entro, best_entro
    cdef double N = 0

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
def find_threshold_MSE(np.ndarray[np.float64_t, ndim=1] x,
                       np.ndarray[np.float64_t, ndim=1] y,
                       np.ndarray[np.int64_t, ndim=1] idx,
                       int min_leaf):
    cdef double sleft = 0, sum, inter, best_inter
    cdef unsigned int i, best_idx = 0
    cdef unsigned int N = idx.shape[0]

    # Initial split (min_leaf on the left)
    if N <= min_leaf:
        return 0, 0
    sum = 0
    for i in range(min_leaf - 1):  # one will be added in the loop
        sum += y[idx[i]]
    sleft = sum
    for i in range(min_leaf - 1, N):
        sum += y[idx[i]]

    best_inter = (sum * sum) / N
    for i in range(min_leaf - 1, N - min_leaf):
        sleft += y[idx[i]]
        if x[idx[i]] == x[idx[i + 1]]:
            continue
        inter = sleft * sleft / (i + 1) + (sum - sleft) * (sum - sleft) / (N - i - 1)
        if inter > best_inter:
            best_inter = inter
            best_idx = i
    # punishment for missing values is delivered outside
    return (best_inter - (sum * sum) / N) / N, x[idx[best_idx]]


@cython.boundscheck(False)
@cython.wraparound(False)
def find_binarization_MSE(np.ndarray[np.float64_t, ndim=1] x,
                          np.ndarray[np.float64_t, ndim=1] y,
                          int n_values, int min_leaf):
    cdef double sleft, sum, val
    cdef unsigned int left
    cdef unsigned int i, change, to_right, allowed, m
    cdef unsigned int best_mapping, move, mapping, previous
    cdef double inter, best_inter, start_inter
    cdef unsigned int N

    cdef np.ndarray[np.int32_t, ndim=1] group_sizes = \
        numpy.zeros(n_values, dtype=numpy.int32)
    cdef np.ndarray[np.float64_t, ndim=1] group_sums = numpy.zeros(n_values)

    N = 0
    for i in range(x.shape[0]):
        val = x[i]
        if not npy_isnan(val):
            group_sizes[int(val)] += 1
            group_sums[int(val)] += y[i]
            N += 1
    if N == 0:
        return 0, 0
    left = N
    sleft = sum = numpy.sum(group_sums)
    best_inter = start_inter = (sum * sum) / N

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
            left -= group_sizes[move]
            sleft -= group_sums[move]
        else:
            left += group_sizes[move]
            sleft += group_sums[move]

        if left >= min_leaf and (N - left) >= min_leaf:
            inter = sleft * sleft / left + (sum - sleft) * (sum - sleft) / (N - left)
            if inter > best_inter:
                best_inter = inter
                best_mapping = mapping
    # factor N / x.shape[0] is the punishment for missing values
    # return (best_inter - start_inter) / N * (N / x.shape[0]), best_mapping
    return (best_inter - start_inter) / x.shape[0], best_mapping


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_grouped_MSE(np.ndarray[np.float64_t, ndim=1] x,
                        np.ndarray[np.float64_t, ndim=1] y,
                        int n_values, int min_leaf):
    cdef int i, n
    cdef double sum = 0, inter, tx

    cdef np.ndarray[np.int32_t, ndim=1] group_sizes = numpy.zeros(n_values, dtype=numpy.int32)
    cdef np.ndarray[np.float64_t, ndim=1] group_sums = numpy.zeros(n_values)

    for i in range(x.shape[0]):
        tx = x[i]
        if not npy_isnan(tx):
            group_sizes[int(tx)] += 1
            group_sums[int(tx)] += y[i]
    inter = 0
    n = 0
    for i in range(n_values):
        if group_sizes[i] < min_leaf:
            # We don't construct nodes with less than min_leaf instances
            # If there is only one non-null node, the split will yield a
            # score of 0
            continue
        inter += group_sums[i] * group_sums[i] / group_sizes[i]
        sum += group_sums[i]
        n += group_sizes[i]
    if n < 2:
        return 0
    # factor n / x.shape[0] is the punishment for missing values
    #return (inter - sum * sum / n) / n * n / x.shape[0]
    return (inter - sum * sum / n) / x.shape[0]


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_predictions(np.ndarray[np.float64_t, ndim=2] X,
                        np.ndarray[np.int32_t, ndim=1] code,
                        np.ndarray[np.float64_t, ndim=2] class_distrs,
                        np.ndarray[np.float64_t, ndim=1] thresholds):
    cdef unsigned int node_ptr, node_idx, i, val_idx
    cdef signed int next_node_ptr
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
            next_node_ptr = code[node_ptr + 3 + val_idx]
            if next_node_ptr == -1:
                break
            node_ptr = next_node_ptr
        node_idx = code[node_ptr + 1]
        predictions[i] = class_distrs[node_idx]
    return predictions
