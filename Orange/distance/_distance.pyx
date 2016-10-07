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


# This function is unused, but kept here for any future use
cdef _check_division_by_zero(double[:, :] x, double[:] dividers):
    cdef int col
    for col in range(dividers.shape[0]):
        if dividers[col] == 0 and not x[:, col].isnan().all():
            raise ValueError("cannot normalize: the data has no variance")


cdef _lower_to_symmetric(double [:, :] distances):
    cdef int row1, row2
    for row1 in range(distances.shape[0]):
        for row2 in range(row1):
            distances[row2, row1] = distances[row1, row2]


def euclidean_rows(np.ndarray[np.float64_t, ndim=2] x1,
                   np.ndarray[np.float64_t, ndim=2] x2,
                   fit_params):
    cdef:
        double [:] vars = fit_params["vars"]
        double [:] means = fit_params["means"]
        double [:, :] dist_missing = fit_params["dist_missing"]
        double [:] dist_missing2 = fit_params["dist_missing2"]
        char normalize = fit_params["normalize"]

        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d
        int ival1, ival2
        double [:, :] distances
        char same = x1 is x2

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0]
    assert n_cols == x2.shape[1] == len(vars) == len(means) \
            == len(dist_missing) == len(dist_missing2)
    distances = np.zeros((n_rows1, n_rows2), dtype=float)

    with nogil:
        for row1 in range(n_rows1):
            for row2 in range(row1 if same else n_rows2):
                d = 0
                for col in range(n_cols):
                    if vars[col] == -2:
                        continue
                    val1, val2 = x1[row1, col], x2[row2, col]
                    if npy_isnan(val1) and npy_isnan(val2):
                        d += dist_missing2[col]
                    elif vars[col] == -1:
                        ival1, ival2 = int(val1), int(val2)
                        if npy_isnan(val1):
                            d += dist_missing[col, ival2]
                        elif npy_isnan(val2):
                            d += dist_missing[col, ival1]
                        elif ival1 != ival2:
                            d += 1
                    elif normalize:
                        if npy_isnan(val1):
                            d += (val2 - means[col]) ** 2 / vars[col] / 2 + 0.5
                        elif npy_isnan(val2):
                            d += (val1 - means[col]) ** 2 / vars[col] / 2 + 0.5
                        else:
                            d += ((val1 - val2) ** 2 / vars[col]) / 2
                    else:
                        if npy_isnan(val1):
                            d += (val2 - means[col]) ** 2 + vars[col]
                        elif npy_isnan(val2):
                            d += (val1 - means[col]) ** 2 + vars[col]
                        else:
                            d += (val1 - val2) ** 2
                distances[row1, row2] = d
    if same:
        _lower_to_symmetric(distances)
    return np.sqrt(distances)


def euclidean_cols(np.ndarray[np.float64_t, ndim=2] x, fit_params):
    cdef:
        double [:] means = fit_params["means"]
        double [:] vars = fit_params["vars"]
        char normalize = fit_params["normalize"]

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
                    if normalize:
                        val1 = (val1 - means[col1]) / sqrt(2 * vars[col1])
                        val2 = (val2 - means[col2]) / sqrt(2 * vars[col2])
                        if npy_isnan(val1):
                            if npy_isnan(val2):
                                d += 1
                            else:
                                d += val2 ** 2 + 0.5
                        elif npy_isnan(val2):
                            d += val1 ** 2 + 0.5
                        else:
                            d += (val1 - val2) ** 2
                    else:
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
    return np.sqrt(distances)


def manhattan_rows(np.ndarray[np.float64_t, ndim=2] x1,
                   np.ndarray[np.float64_t, ndim=2] x2,
                   fit_params):
    cdef:
        double [:] medians = fit_params["medians"]
        double [:] mads = fit_params["mads"]
        double [:, :] dist_missing = fit_params["dist_missing"]
        double [:] dist_missing2 = fit_params["dist_missing2"]
        char normalize = fit_params["normalize"]
        char same = x1 is x2

        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, d
        int ival1, ival2
        double [:, :] distances

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0]
    assert n_cols == x2.shape[1] == len(mads) == len(medians) \
            == len(dist_missing) == len(dist_missing2)

    distances = np.zeros((n_rows1, n_rows2), dtype=float)
    for row1 in range(n_rows1):
        for row2 in range(row1 if same else n_rows2):
            d = 0
            for col in range(n_cols):
                if mads[col] == -2:
                    continue

                val1, val2 = x1[row1, col], x2[row2, col]
                if npy_isnan(val1) and npy_isnan(val2):
                    d += dist_missing2[col]
                elif mads[col] == -1:
                    ival1, ival2 = int(val1), int(val2)
                    if npy_isnan(val1):
                        d += dist_missing[col, ival2]
                    elif npy_isnan(val2):
                        d += dist_missing[col, ival1]
                    elif ival1 != ival2:
                        d += 1
                elif normalize:
                    if npy_isnan(val1):
                        d += fabs(val2 - medians[col]) / mads[col] / 2 + 0.5
                    elif npy_isnan(val2):
                        d += fabs(val1 - medians[col]) / mads[col] / 2 + 0.5
                    else:
                        d += fabs(val1 - val2) / mads[col] / 2
                else:
                    if npy_isnan(val1):
                        d += fabs(val2 - medians[col]) + mads[col]
                    elif npy_isnan(val2):
                        d += fabs(val1 - medians[col]) + mads[col]
                    else:
                        d += fabs(val1 - val2)

            distances[row1, row2] = d

    if same:
        _lower_to_symmetric(distances)
    return distances


def manhattan_cols(np.ndarray[np.float64_t, ndim=2] x, fit_params):
    cdef:
        double [:] medians = fit_params["medians"]
        double [:] mads = fit_params["mads"]
        char normalize = fit_params["normalize"]

        int n_rows, n_cols, col1, col2, row
        double val1, val2, d
        double [:, :] distances

    n_rows, n_cols = x.shape[0], x.shape[1]
    distances = np.zeros((n_cols, n_cols), dtype=float)
    for col1 in range(n_cols):
        for col2 in range(col1):
            d = 0
            for row in range(n_rows):
                val1, val2 = x[row, col1], x[row, col2]
                if normalize:
                    val1 = (val1 - medians[col1]) / (2 * mads[col1])
                    val2 = (val2 - medians[col2]) / (2 * mads[col2])
                    if npy_isnan(val1):
                        if npy_isnan(val2):
                            d += 1
                        else:
                            d += fabs(val2) + 0.5
                    elif npy_isnan(val2):
                        d += fabs(val1) + 0.5
                    else:
                        d += fabs(val1 - val2)
                else:
                    if npy_isnan(val1):
                        if npy_isnan(val2):
                            d += mads[col1] + mads[col2] \
                                 + fabs(medians[col1] - medians[col2])
                        else:
                            d += fabs(val2 - medians[col1]) + mads[col1]
                    elif npy_isnan(val2):
                        d += fabs(val1 - medians[col2]) + mads[col2]
                    else:
                        d += fabs(val1 - val2)
            distances[col1, col2] = distances[col2, col1] = d
    return distances


def fit_jaccard(np.ndarray[np.float64_t, ndim=2] x, *_):
    cdef:
        int row, n_cols, nonzeros, nonnans
        double val
        double [:] ps

    n_cols = x.shape[1]
    ps = np.empty(n_cols, dtype=np.double)
    for col in range(n_cols):
        nonzeros = nonnans = 0
        for row in range(len(x)):
            val = x[row, col]
            if not npy_isnan(val):
                nonnans += 1
                if val != 0:
                    nonzeros += 1
        ps[col] = float(nonzeros) / nonnans
    return {"ps": ps}


def jaccard_rows(np.ndarray[np.float64_t, ndim=2] x1,
                 np.ndarray[np.float64_t, ndim=2] x2,
                 fit_params):
    cdef:
        double [:] ps = fit_params["ps"]
        char same = x1 is x2

        int n_rows1, n_rows2, n_cols, row1, row2, col
        double val1, val2, intersection, union
        int ival1, ival2
        double [:, :] distances

    n_rows1, n_cols = x1.shape[0], x1.shape[1]
    n_rows2 = x2.shape[0]
    assert n_cols == x2.shape[1] == ps.shape[0]

    distances = np.ones((n_rows1, n_rows2), dtype=float)
    for row1 in range(n_rows1):
        for row2 in range(row1 if same else n_rows2):
            intersection = union = 0
            for col in range(n_cols):
                val1, val2 = x1[row1, col], x1[row2, col]
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
                        intersection += ps[col]
                        union += 1
                    else:
                        union += ps[col]
                else:
                    if val1 != 0 and val2 != 0:
                        intersection += 1
                    if val1 != 0 or val2 != 0:
                        union += 1
            if union != 0:
                distances[row1, row2] = 1 - intersection / union

    if same:
        _lower_to_symmetric(distances)
    return distances


def jaccard_cols(np.ndarray[np.float64_t, ndim=2] x, fit_params):
    cdef:
        double [:] ps = fit_params["ps"]

        int n_rows, n_cols, col1, col2, row
        double val1, val2
        int in_both, in_one, in1_unk2, unk1_in2, unk1_unk2, unk1_not2, not1_unk2
        double [:, :] distances

    n_rows, n_cols = x.shape[0], x.shape[1]
    distances = np.ones((n_cols, n_cols), dtype=float)
    for col1 in range(n_cols):
        for col2 in range(col1):
            in_both = in_one = 0
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
                    if val1 != 0 and val2 != 0:
                        in_both += 1
                    elif val1 != 0 or val2 != 0:
                        in_one += 1
            distances[col1, col2] = distances[col2, col1] = \
                1 - float(in_both
                          + ps[col1] * unk1_in2 +
                          + ps[col2] * in1_unk2 +
                          + ps[col1] * ps[col2] * unk1_unk2) / \
                (in_both + in_one + unk1_in2 + in1_unk2 +
                 + ps[col1] * unk1_not2
                 + ps[col2] * not1_unk2
                 + (1 - (1 - ps[col1]) * (1 - ps[col2])) * unk1_unk2)
    return distances
