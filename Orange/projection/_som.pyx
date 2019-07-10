#cython: embedsignature=True
#cython: infer_types=True
#cython: cdivision=True
#cython: boundscheck=True
#cython: wraparound=False
#cython: language_level=3


# TODO: disable boundscheck
# TODO: add nogil

import numpy as np
cimport numpy as np

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x) nogil

cdef extern from "math.h":
    double exp(double x) nogil


cpdef void get_winner(np.ndarray[np.float64_t, ndim=3] weights,
                      np.ndarray[np.float64_t, ndim=1] row,
                      np.ndarray[np.int16_t, ndim=1] winner,
                      int hex):
    cdef:
        int x, y, best_x, best_y
        double diff, smallest_diff = 1e30

    for y in range(weights.shape[0]):
        for x in range(weights.shape[1] - hex * (y % 2)):
            diff = 0
            for col in range(weights.shape[2]):
                diff += (row[col] - weights[y, x, col]) ** 2
            if diff < smallest_diff:
                best_x = x
                best_y = y
                smallest_diff = diff
    winner[0] = best_x
    winner[1] = best_y

def update(np.ndarray[np.float64_t, ndim=3] weights,
           np.ndarray[np.float64_t, ndim=2] X,
           double eta, double sigma):
    cdef:
        int rowi, x, y, win_x, win_y
        np.ndarray[np.int16_t, ndim=1] winner = np.empty(2, dtype=np.int16)
        int max_dist = weights.shape[0] ** 2 + weights.shape[1] ** 2
        double[:] w_lookup = np.empty(max_dist + 1)
        double d = 6.28 * sigma * sigma
        double w

    for x in range(max_dist + 1):
        w_lookup[x] = eta * exp(-x / d)

    for rowi in range(X.shape[0]):
        get_winner(weights, X[rowi], winner, 0)
        win_x = winner[0]
        win_y = winner[1]
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                w = w_lookup[(y - win_y) ** 2 + (x - win_x) ** 2]
                for col in range(weights.shape[2]):
                    weights[y, x, col] += w * (X[rowi, col] - weights[y, x, col])


def update_hex(np.ndarray[np.float64_t, ndim=3] weights,
               np.ndarray[np.float64_t, ndim=2] X,
               double eta, double sigma):
    cdef:
        int rowi, x, y, win_x, win_y
        np.ndarray[np.int16_t, ndim=1] winner = np.empty(2, dtype=np.int16)
        int max_dist = weights.shape[0] ** 2 + weights.shape[1] ** 2
        double d = 6.28 * sigma * sigma
        double dist, dy
        double w

    for rowi in range(X.shape[0]):
        get_winner(weights, X[rowi], winner, 1)
        win_x = winner[0]
        win_y = winner[1]
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                dy = y - win_y
                dist = dy ** 2 * 3 / 4 + (x - win_x + (dy % 2) / 2) ** 2
                w = eta * exp(-dist / d)
                for col in range(weights.shape[2]):
                    weights[y, x, col] += w * (X[rowi, col] - weights[y, x, col])


cpdef void get_winner_sparse(np.ndarray[np.float64_t, ndim=3] weights,
                             np.ndarray[np.float64_t, ndim=2] ssumweights,
                             np.int32_t[:] columns,
                             np.float64_t[:] data,
                             np.ndarray[np.int16_t, ndim=1] winner,
                             int hex):
    cdef:
        int x, y, best_x, best_y, i, col
        double diff, smallest_diff = 1e30

    for y in range(weights.shape[0]):
        for x in range(weights.shape[1] - hex * (y % 2)):
            diff = ssumweights[y, x]  # First assume that all values are zero
            for i in range(columns.shape[0]):
                col = columns[i]
                diff += (data[i] - weights[y, x, col]) ** 2 \
                        - weights[y, x, col] ** 2
            if diff < smallest_diff:
                best_x = x
                best_y = y
                smallest_diff = diff
    winner[0] = best_x
    winner[1] = best_y


def update_sparse(np.ndarray[np.float64_t, ndim=3] weights,
                  np.ndarray[np.float64_t, ndim=2] ssumweights,
                  X,
                  double eta, double sigma):
    cdef:
        int rowi, x, y, win_x, win_y, i, col
        np.ndarray[np.int16_t, ndim=1] winner = np.empty(2, dtype=np.int16)
        int max_dist = weights.shape[0] ** 2 + weights.shape[1] ** 2
        double[:] w_lookup = np.empty(max_dist + 1)
        double d = 6.28 * sigma * sigma
        double w
        np.float64_t[:] data = X.data
        np.int32_t[:] indices = X.indices
        np.int32_t[:] indptr = X.indptr
        np.float64_t[:] row
        np.int32_t[:] columns

    for x in range(max_dist + 1):
        w_lookup[x] = eta * exp(-x / d)

    for rowi in range(X.shape[0]):
        columns = indices[indptr[rowi]:indptr[rowi + 1]]
        row = data[indptr[rowi]:indptr[rowi + 1]]
        get_winner_sparse(weights, ssumweights, columns, row, winner, 0)
        win_x = winner[0]
        win_y = winner[1]
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1] - y % 2):
                w = w_lookup[(y - win_y) ** 2 + (x - win_x) ** 2]
                for i in range(columns.shape[0]):
                    col = columns[i]
                    ssumweights[y, x] -= weights[y, x, col] ** 2
                    weights[y, x, col] += w * (row[i] - weights[y, x, col])
                    ssumweights[y, x] += weights[y, x, col] ** 2


def update_sparse_hex(np.ndarray[np.float64_t, ndim=3] weights,
                  np.ndarray[np.float64_t, ndim=2] ssumweights,
                  X,
                  double eta, double sigma):
    cdef:
        int rowi, x, y, win_x, win_y, i, col
        np.ndarray[np.int16_t, ndim=1] winner = np.empty(2, dtype=np.int16)
        double d = 6.28 * sigma * sigma
        double w
        np.float64_t[:] data = X.data
        np.int32_t[:] indices = X.indices
        np.int32_t[:] indptr = X.indptr
        np.float64_t[:] row
        np.int32_t[:] columns

    for rowi in range(X.shape[0]):
        columns = indices[indptr[rowi]:indptr[rowi + 1]]
        row = data[indptr[rowi]:indptr[rowi + 1]]
        get_winner_sparse(weights, ssumweights, columns, row, winner, 0)
        win_x = winner[0]
        win_y = winner[1]
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1] - y % 2):
                dy = y - win_y
                dist = dy ** 2 * 3 / 4 + (x - win_x + (dy % 2) / 2) ** 2
                w = eta * exp(-dist / d)
                for i in range(columns.shape[0]):
                    col = columns[i]
                    ssumweights[y, x] -= weights[y, x, col] ** 2
                    weights[y, x, col] += w * (row[i] - weights[y, x, col])
                    ssumweights[y, x] += weights[y, x, col] ** 2
