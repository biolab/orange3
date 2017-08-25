#cython: embedsignature=True
#cython: infer_types=True
#cython: cdivision=True
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x) nogil

cdef extern from "math.h":
    double fabs(double x) nogil
    double sqrt(double x) nogil


cpdef void lower_to_symmetric(double [:, :] distances):
    cdef int row1, row2
    for row1 in range(distances.shape[0]):
        for row2 in range(row1):
            distances[row2, row1] = distances[row1, row2]


def euclidean_rows_discrete(np.ndarray[np.float64_t, ndim=2] distances,
                            np.ndarray[np.float64_t, ndim=2] x1,
                            np.ndarray[np.float64_t, ndim=2] x2,
                            double[:, :] dist_missing,
                            np.ndarray[np.float64_t, ndim=1] dist_missing2,
                            char two_tables):
    cdef:
        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d
        int ival1, ival2

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0]
    with nogil:
        for row1 in range(n_rows1):
            for row2 in range(n_rows2 if two_tables else row1):
                d = 0
                for col in range(n_cols):
                    val1, val2 = x1[row1, col], x2[row2, col]
                    ival1, ival2 = int(val1), int(val2)
                    if npy_isnan(val1):
                        if npy_isnan(val2):
                            d += dist_missing2[col]
                        else:
                            d += dist_missing[col, ival2]
                    elif npy_isnan(val2):
                        d += dist_missing[col, ival1]
                    elif ival1 != ival2:
                        d += 1
                distances[row1, row2] += d


def fix_euclidean_rows(
    np.ndarray[np.float64_t, ndim=2] distances,
    np.ndarray[np.float64_t, ndim=2] x1,
    np.ndarray[np.float64_t, ndim=2] x2,
    np.ndarray[np.float64_t, ndim=1] means,
    np.ndarray[np.float64_t, ndim=1] vars,
    np.ndarray[np.float64_t, ndim=1] dist_missing2,
    char two_tables):
    cdef:
        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0]
    with nogil:
        for row1 in range(n_rows1):
            for row2 in range(n_rows2 if two_tables else row1):
                if npy_isnan(distances[row1, row2]):
                    d = 0
                    for col in range(n_cols):
                        val1, val2 = x1[row1, col], x2[row2, col]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                d += dist_missing2[col]
                            else:
                                d += (val2 - means[col]) ** 2 + vars[col]
                        elif npy_isnan(val2):
                            d += (val1 - means[col]) ** 2 + vars[col]
                        else:
                            d += (val1 - val2) ** 2
                    distances[row1, row2] = d
                    if not two_tables:
                        distances[row2, row1] = d


def fix_euclidean_rows_normalized(
    np.ndarray[np.float64_t, ndim=2] distances,
    np.ndarray[np.float64_t, ndim=2] x1,
    np.ndarray[np.float64_t, ndim=2] x2,
    np.ndarray[np.float64_t, ndim=1] means,
    np.ndarray[np.float64_t, ndim=1] vars,
    np.ndarray[np.float64_t, ndim=1] dist_missing2,
    char two_tables):
    cdef:
        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0]
    with nogil:
        for row1 in range(n_rows1):
            for row2 in range(n_rows2 if two_tables else row1):
                if npy_isnan(distances[row1, row2]):
                    d = 0
                    for col in range(n_cols):
                        val1, val2 = x1[row1, col], x2[row2, col]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                d += dist_missing2[col]
                            else:
                                d += val2 ** 2 + 0.5
                        elif npy_isnan(val2):
                            d += val1 ** 2 + 0.5
                        else:
                            d += (val1 - val2) ** 2
                    distances[row1, row2] = d
                    if not two_tables:
                        distances[row2, row1] = d


def fix_euclidean_cols(
    np.ndarray[np.float64_t, ndim=2] distances,
    np.ndarray[np.float64_t, ndim=2] x,
    double[:] means,
    double[:] vars):
    cdef:
        int n_rows, n_cols, col1, col2, row
        double val1, val2, d

    n_rows, n_cols = x.shape[0], x.shape[1]
    with nogil:
        for col1 in range(n_cols):
            for col2 in range(col1):
                if npy_isnan(distances[col1, col2]):
                    d = 0
                    for row in range(n_rows):
                        val1, val2 = x[row, col1], x[row, col2]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                d += vars[col1] + vars[col2] \
                                     + (means[col1] - means[col2]) ** 2
                            else:
                                d += (val2 - means[col1]) ** 2 + vars[col1]
                        elif npy_isnan(val2):
                            d += (val1 - means[col2]) ** 2 + vars[col2]
                        else:
                            d += (val1 - val2) ** 2
                    distances[col1, col2] = distances[col2, col1] = d


def fix_euclidean_cols_normalized(
    np.ndarray[np.float64_t, ndim=2] distances,
    np.ndarray[np.float64_t, ndim=2] x,
    double[:] means,
    double[:] vars):
    cdef:
        int n_rows, n_cols, col1, col2, row
        double val1, val2, d

    n_rows, n_cols = x.shape[0], x.shape[1]
    with nogil:
        for col1 in range(n_cols):
            for col2 in range(col1):
                if npy_isnan(distances[col1, col2]):
                    d = 0
                    for row in range(n_rows):
                        val1, val2 = x[row, col1], x[row, col2]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                d += 1
                            else:
                                d += val2 ** 2 + 0.5
                        elif npy_isnan(val2):
                            d += val1 ** 2 + 0.5
                        else:
                            d += (val1 - val2) ** 2
                    distances[col1, col2] = distances[col2, col1] = d


def manhattan_rows_cont(np.ndarray[np.float64_t, ndim=2] x1,
                        np.ndarray[np.float64_t, ndim=2] x2,
                        char two_tables):
    cdef:
        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d
        np.ndarray[np.float64_t, ndim=2] distances

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0]
    distances = np.zeros((n_rows1, n_rows2), dtype=float)
    with nogil:
        for row1 in range(n_rows1):
            for row2 in range(n_rows2 if two_tables else row1):
                d = 0
                for col in range(n_cols):
                    d += fabs(x1[row1, col] - x2[row2, col])
                distances[row1, row2] = d
    return distances

def fix_manhattan_rows(np.ndarray[np.float64_t, ndim=2] distances,
                       np.ndarray[np.float64_t, ndim=2] x1,
                       np.ndarray[np.float64_t, ndim=2] x2,
                       np.ndarray[np.float64_t, ndim=1] medians,
                       np.ndarray[np.float64_t, ndim=1] mads,
                       np.ndarray[np.float64_t, ndim=1] dist_missing2_cont,
                       char two_tables):
    cdef:
        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0] if two_tables else 0
    with nogil:
        for row1 in range(n_rows1):
            for row2 in range(n_rows2 if two_tables else row1):
                if npy_isnan(distances[row1, row2]):
                    d = 0
                    for col in range(n_cols):
                        val1, val2 = x1[row1, col], x2[row2, col]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                d += dist_missing2_cont[col]
                            else:
                                d += fabs(val2 - medians[col]) + mads[col]
                        elif npy_isnan(val2):
                            d += fabs(val1 - medians[col]) + mads[col]
                        else:
                            d += fabs(val1 - val2)
                    distances[row1, row2] = d
    return distances


def fix_manhattan_rows_normalized(np.ndarray[np.float64_t, ndim=2] distances,
                                  np.ndarray[np.float64_t, ndim=2] x1,
                                  np.ndarray[np.float64_t, ndim=2] x2,
                                  char two_tables):
    cdef:
        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0] if two_tables else 0
    with nogil:
        for row1 in range(n_rows1):
            for row2 in range(n_rows2 if two_tables else row1):
                if npy_isnan(distances[row1, row2]):
                    d = 0
                    for col in range(n_cols):
                        val1, val2 = x1[row1, col], x2[row2, col]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                d += 1
                            else:
                                d += fabs(val2) + 0.5
                        elif npy_isnan(val2):
                            d += fabs(val1) + 0.5
                        else:
                            d += fabs(val1 - val2)
                    distances[row1, row2] = d
    return distances


def manhattan_cols(np.ndarray[np.float64_t, ndim=2] x,
                   np.ndarray[np.float64_t, ndim=1] medians,
                   np.ndarray[np.float64_t, ndim=1] mads,
                   char normalize):
    cdef:
        int n_rows, n_cols, col1, col2, row
        double val1, val2, d
        double [:, :] distances

    n_rows, n_cols = x.shape[0], x.shape[1]
    distances = np.zeros((n_cols, n_cols), dtype=float)
    with nogil:
        for col1 in range(n_cols):
            for col2 in range(col1):
                d = 0
                for row in range(n_rows):
                    val1, val2 = x[row, col1], x[row, col2]
                    if npy_isnan(val1):
                        if npy_isnan(val2):
                            if normalize:
                                d += 1
                            else:
                                d += mads[col1] + mads[col2] \
                                    + fabs(medians[col1] - medians[col2])
                        else:
                            if normalize:
                                d += fabs(val2) + 0.5
                            else:
                                d += fabs(val2 - medians[col1]) + mads[col1]
                    else:
                        if npy_isnan(val2):
                            if normalize:
                                d += fabs(val1) + 0.5
                            else:
                                d += fabs(val1 - medians[col2]) + mads[col2]
                        else:
                            d += fabs(val1 - val2)
                distances[col1, col2] = distances[col2, col1] = d
    return distances


def p_nonzero(np.ndarray[np.float64_t, ndim=1] x):
    cdef:
        int row, nonzeros, nonnans
        double val

    nonzeros = nonnans = 0
    for row in range(len(x)):
        val = x[row]
        if not npy_isnan(val):
            nonnans += 1
            if val != 0:
                nonzeros += 1
    return float(nonzeros) / nonnans

def any_nan_row(np.ndarray[np.float64_t, ndim=2] x):
    cdef:
        int row, n_cols, n_rows
        np.ndarray[np.int8_t, ndim=1] flags

    n_rows, n_cols = x.shape[0], x.shape[1]
    flags = np.zeros(x.shape[0], dtype=np.int8)
    with nogil:
        for row in range(n_rows):
            for col in range(n_cols):
                if npy_isnan(x[row, col]):
                    flags[row] = 1
                    break
    return flags


def jaccard_rows(np.ndarray[np.int8_t, ndim=2] nonzeros1,
                 np.ndarray[np.int8_t, ndim=2] nonzeros2,
                 np.ndarray[np.float64_t, ndim=2] x1,
                 np.ndarray[np.float64_t, ndim=2] x2,
                 np.ndarray[np.int8_t, ndim=1] nans1,
                 np.ndarray[np.int8_t, ndim=1] nans2,
                 np.ndarray[np.float64_t, ndim=1] ps,
                 char two_tables):
    cdef:
        int n_rows1, n_rows2, n_cols, row1, row2, col
        np.float64_t val1, val2, intersection, union
        int ival1, ival2
        double [:, :] distances

    n_rows1, n_cols = x1.shape[0], x2.shape[1]
    n_rows2 = x2.shape[0]
    distances = np.zeros((n_rows1, n_rows2), dtype=float)
    with nogil:
        for row1 in range(n_rows1):
            if nans1[row1]:
                for row2 in range(n_rows2 if two_tables else row1):
                    union = intersection = 0
                    for col in range(n_cols):
                        val1, val2 = x1[row1, col], x2[row2, col]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                intersection += ps[col] ** 2
                                union += 1 - (1 - ps[col]) ** 2
                            elif val2 != 0:
                                intersection += ps[col]
                                union += 1
                            else:
                                union += ps[col]
                        elif npy_isnan(val2):
                            if val1 != 0:
                                intersection += val1 * ps[col]
                                union += 1
                            else:
                                union += ps[col]
                        else:
                            ival1 = nonzeros1[row1, col]
                            ival2 = nonzeros2[row2, col]
                            union += ival1 | ival2
                            intersection += ival1 & ival2
                    if union != 0:
                        distances[row1, row2] = 1 - intersection / union
            else:
                for row2 in range(n_rows2 if two_tables else row1):
                    union = intersection = 0
                    # This case is slightly different since val1 can't be nan
                    if nans2[row2]:
                        for col in range(n_cols):
                            val2 = x2[row2, col]
                            if nonzeros1[row1, col] != 0:
                                union += 1
                                if npy_isnan(val2):
                                    intersection += ps[col]
                                elif val2 != 0:
                                    intersection += 1
                            elif npy_isnan(val2):
                                union += ps[col]
                            elif val2 != 0:
                                union += 1
                    else:
                        for col in range(n_cols):
                            ival1 = nonzeros1[row1, col]
                            ival2 = nonzeros2[row2, col]
                            union += ival1 | ival2
                            intersection += ival1 & ival2
                    if union != 0:
                        distances[row1, row2] = 1 - intersection / union
    if not two_tables:
        lower_to_symmetric(distances)
    return distances


def jaccard_cols(np.ndarray[np.int8_t, ndim=2] nonzeros,
                 np.ndarray[np.float64_t, ndim=2] x,
                 np.ndarray[np.int8_t, ndim=1] nans,
                 np.ndarray[np.float64_t, ndim=1] ps):
    cdef:
        int n_rows, n_cols, col1, col2, row
        double val1, val2, intersection, union
        np.int8_t ival1, ival2
        int in_both, in_any, in1_unk2, unk1_in2, unk1_unk2, unk1_not2, not1_unk2
        double [:, :] distances

    n_rows, n_cols = x.shape[0], x.shape[1]
    distances = np.zeros((n_cols, n_cols), dtype=float)
    with nogil:
        for col1 in range(n_cols):
            if nans[col1]:
                for col2 in range(col1):
                    in_both = in_any = 0
                    in1_unk2 = unk1_in2 = unk1_unk2 = unk1_not2 = not1_unk2 = 0
                    for row in range(n_rows):
                        val1, val2 = x[row, col1], x[row, col2]
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                unk1_unk2 += 1
                            elif val2 != 0:
                                unk1_in2 += 1
                            else:
                                unk1_not2 += 1
                        elif npy_isnan(val2):
                            if val1 != 0:
                                in1_unk2 += 1
                            else:
                                not1_unk2 += 1
                        else:
                            ival1 = nonzeros[row, col1]
                            ival2 = nonzeros[row, col2]
                            in_both += ival1 & ival2
                            in_any += ival1 | ival2
                    union = (in_any + unk1_in2 + in1_unk2
                             + ps[col1] * unk1_not2
                             + ps[col2] * not1_unk2
                             + (1 - (1 - ps[col1]) * (1 - ps[col2])) * unk1_unk2)
                    if union != 0:
                        intersection = (in_both
                                        + ps[col1] * unk1_in2 +
                                        + ps[col2] * in1_unk2 +
                                        + ps[col1] * ps[col2] * unk1_unk2)
                        distances[col1, col2] = distances[col2, col1] = \
                            1 - intersection / union
            else:
                for col2 in range(col1):
                    if nans[col2]:
                        in_both = in_any = 0
                        in1_unk2 = unk1_in2 = unk1_unk2 = unk1_not2 = not1_unk2 = 0
                        for row in range(n_rows):
                            ival1 = nonzeros[row, col1]
                            val2 = x[row, col2]
                            if npy_isnan(val2):
                                if ival1:
                                    in1_unk2 += 1
                                else:
                                    not1_unk2 += 1
                            else:
                                ival2 = nonzeros[row, col2]
                                in_both += ival1 & ival2
                                in_any += ival1 | ival2
                        distances[col1, col2] = distances[col2, col1] = \
                            1 - float(in_both
                                      + ps[col1] * unk1_in2 +
                                      + ps[col2] * in1_unk2 +
                                      + ps[col1] * ps[col2] * unk1_unk2) / \
                            (in_any + unk1_in2 + in1_unk2 +
                             + ps[col1] * unk1_not2
                             + ps[col2] * not1_unk2
                             + (1 - (1 - ps[col1]) * (1 - ps[col2])) * unk1_unk2)
                    else:
                        in_both = in_any = 0
                        for row in range(n_rows):
                            ival1 = nonzeros[row, col1]
                            ival2 = nonzeros[row, col2]
                            in_both += ival1 & ival2
                            in_any += ival1 | ival2
                        if in_any != 0:
                            distances[col1, col2] = distances[col2, col1] = \
                                1 - float(in_both) / in_any

    return distances
