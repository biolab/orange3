#cython: embedsignature=True
#cython: infer_types=True
#cython: cdivision=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

import numpy as np
cimport numpy as np

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x) nogil

cdef extern from "math.h":
    double exp(double x) nogil


def get_winners(np.float64_t[:, :, :] weights, np.float64_t[:, :] X, int hex):
    cdef:
        int x, y, win_x, win_y
        double diff, min_diff
        np.float64_t[:] row
        np.ndarray[np.int16_t, ndim=2] winners = \
            np.empty((X.shape[0], 2), dtype=np.int16)
        int nrows = X.shape[0]

    with nogil:
        for rowi in range(nrows):
            row = X[rowi]

            min_diff = 1e30
            for y in range(weights.shape[0]):
                for x in range(weights.shape[1] - hex * (y % 2)):
                    diff = 0
                    for col in range(weights.shape[2]):
                        diff += (row[col] - weights[y, x, col]) ** 2
                    if diff < min_diff:
                        win_x = x
                        win_y = y
                        min_diff = diff
            winners[rowi, 0] = win_x
            winners[rowi, 1] = win_y

    return winners


def update(np.float64_t[:, :, :] weights,
           np.float64_t[:, :] X,
           double eta, double sigma):
    cdef:
        int rowi, x, y, win_x, win_y
        np.float64_t[:] row
        int max_dist = weights.shape[0] ** 2 + weights.shape[1] ** 2
        double[:] w_lookup = np.empty(max_dist + 1)
        double d = 6.28 * sigma * sigma
        double w, diff, min_diff

    with nogil:
        for x in range(max_dist + 1):
            w_lookup[x] = eta * exp(-x / d)

        for rowi in range(X.shape[0]):
            row = X[rowi]

            min_diff = 1e30
            for y in range(weights.shape[0]):
                for x in range(weights.shape[1]):
                    diff = 0
                    for col in range(weights.shape[2]):
                        diff += (row[col] - weights[y, x, col]) ** 2
                    if diff < min_diff:
                        win_x = x
                        win_y = y
                        min_diff = diff

            for y in range(weights.shape[0]):
                for x in range(weights.shape[1]):
                    w = w_lookup[(y - win_y) ** 2 + (x - win_x) ** 2]
                    for col in range(weights.shape[2]):
                        weights[y, x, col] += w * (row[col] - weights[y, x, col])


def update_hex(np.float64_t[:, :, :] weights,
               np.float64_t[:, :] X,
               double eta, double sigma):
    cdef:
        int rowi, x, y, win_x, win_y
        np.float64_t[:] row
        double d = 6.28 * sigma * sigma
        double dist, dy
        double w, diff, min_diff

    with nogil:
        for rowi in range(X.shape[0]):
            row = X[rowi]

            min_diff = 1e30
            for y in range(weights.shape[0]):
                for x in range(weights.shape[1] - y % 2):
                    diff = 0
                    for col in range(weights.shape[2]):
                        diff += (row[col] - weights[y, x, col]) ** 2
                    if diff < min_diff:
                        win_x = x
                        win_y = y
                        min_diff = diff

            for y in range(weights.shape[0]):
                for x in range(weights.shape[1]):
                    dy = y - win_y
                    dist = dy ** 2 * 3 / 4 + (x - win_x + (dy % 2) / 2) ** 2
                    w = eta * exp(-dist / d)
                    for col in range(weights.shape[2]):
                        weights[y, x, col] += w * (row[col] - weights[y, x, col])


def get_winners_sparse(np.float64_t[:, :, :] weights,
                       np.float64_t[:, :] ssumweights,
                       X, int hex):
    cdef:
        int x, y, best_x, best_y, i, col
        double diff, min_diff
        np.float64_t[:] data = X.data
        np.int32_t[:] indices = X.indices
        np.int32_t[:] indptr = X.indptr
        np.int32_t[:] columns,
        np.float64_t[:] row,
        np.ndarray[np.int16_t, ndim=2] winners = \
            np.empty((X.shape[0], 2), dtype=np.int16)
        int nrows = X.shape[0]

    with nogil:
        for rowi in range(nrows):
            columns = indices[indptr[rowi]:indptr[rowi + 1]]
            row = data[indptr[rowi]:indptr[rowi + 1]]

            min_diff = 1e30
            for y in range(weights.shape[0]):
                for x in range(weights.shape[1] - hex * (y % 2)):
                    diff = ssumweights[y, x]  # First assume that all values are zero
                    for i in range(columns.shape[0]):
                        col = columns[i]
                        diff += (data[i] - weights[y, x, col]) ** 2 \
                                - weights[y, x, col] ** 2
                    if diff < min_diff:
                        win_x = x
                        win_y = y
                        min_diff = diff

            winners[rowi, 0] = win_x
            winners[rowi, 1] = win_y
    return winners


def update_sparse(np.ndarray[np.float64_t, ndim=3] weights,
                  np.ndarray[np.float64_t, ndim=2] ssumweights,
                  X,
                  double eta, double sigma):
    cdef:
        int rowi, x, y, win_x, win_y, i, col
        int max_dist = weights.shape[0] ** 2 + weights.shape[1] ** 2
        double[:] w_lookup = np.empty(max_dist + 1)
        double d = 6.28 * sigma * sigma
        double w, diff, min_diff
        np.float64_t[:] data = X.data
        np.int32_t[:] indices = X.indices
        np.int32_t[:] indptr = X.indptr
        np.float64_t[:] row
        np.int32_t[:] columns
        int nrows = X.shape[0]

    with nogil:
        for x in range(max_dist + 1):
            w_lookup[x] = eta * exp(-x / d)

        for rowi in range(nrows):
            columns = indices[indptr[rowi]:indptr[rowi + 1]]
            row = data[indptr[rowi]:indptr[rowi + 1]]

            min_diff = 1e30
            for y in range(weights.shape[0]):
                for x in range(weights.shape[1]):
                    diff = ssumweights[y, x]
                    for i in range(columns.shape[0]):
                        col = columns[i]
                        diff += (row[i] - weights[y, x, col]) ** 2 \
                                - weights[y, x, col] ** 2
                    if diff < min_diff:
                        win_x = x
                        win_y = y
                        min_diff = diff

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
        double w, diff, min_diff
        np.float64_t[:] data = X.data
        np.int32_t[:] indices = X.indices
        np.int32_t[:] indptr = X.indptr
        np.float64_t[:] row
        np.int32_t[:] columns
        int ncols = X.shape[0]

    with nogil:
        for rowi in range(ncols):
            columns = indices[indptr[rowi]:indptr[rowi + 1]]
            row = data[indptr[rowi]:indptr[rowi + 1]]

            min_diff = 1e30
            for y in range(weights.shape[0]):
                for x in range(weights.shape[1]):
                    diff = ssumweights[y, x]  # First assume that all values are zero
                    for i in range(columns.shape[0]):
                        col = columns[i]
                        diff += (row[i] - weights[y, x, col]) ** 2 \
                                - weights[y, x, col] ** 2
                    if diff < min_diff:
                        win_x = x
                        win_y = y
                        min_diff = diff

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
