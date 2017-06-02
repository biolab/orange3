"""
This module provides alternatives for the few additional functions found in
and once used from the bottlechest package (fork of bottleneck).

It also patches bottleneck to contain these functions.
"""
import numpy as np
import scipy.sparse as sp
import bottleneck as bn


def _count_nans_per_row_sparse(X, weights):
    """ Count the number of nans (undefined) values per row. """
    items_per_row = 1 if X.ndim == 1 else X.shape[1]
    counts = np.ones(X.shape[0]) * items_per_row
    nnz_per_row = np.bincount(X.indices, minlength=len(counts))
    counts -= nnz_per_row
    if weights is not None:
        counts *= weights
    return np.sum(counts)


def bincount(X, max_val=None, weights=None, minlength=None):
    """Return counts of values in array X.

    Works kind of like np.bincount(), except that it also supports floating
    arrays with nans.
    """
    if sp.issparse(X):
        minlength = max_val + 1
        bin_weights = weights[X.indices] if weights is not None else None
        return (np.bincount(X.data.astype(int),
                            weights=bin_weights,
                            minlength=minlength, ),
                _count_nans_per_row_sparse(X, weights))

    X = np.asanyarray(X)
    if X.dtype.kind == 'f' and bn.anynan(X):
        nonnan = ~np.isnan(X)
        X = X[nonnan]
        if weights is not None:
            nans = (~nonnan * weights).sum(axis=0)
            weights = weights[nonnan]
        else:
            nans = (~nonnan).sum(axis=0)
    else:
        nans = 0. if X.ndim == 1 else np.zeros(X.shape[1], dtype=float)
    if minlength is None and max_val is not None:
        minlength = max_val + 1
    bc = np.array([]) if minlength is not None and minlength <= 0 else \
        np.bincount(X.astype(np.int32, copy=False),
                    weights=weights, minlength=minlength).astype(float)
    return bc, nans


def countnans(X, weights=None, axis=None, dtype=None, keepdims=False):
    """
    Count the undefined elements in arr along given axis.

    Parameters
    ----------
    X : array_like
    weights : array_like
        Weights to weight the nans with, before or after counting (depending
        on the weights shape).

    Returns
    -------
    counts
    """
    if not sp.issparse(X):
        X = np.asanyarray(X)
        isnan = np.isnan(X)
        if weights is not None and weights.shape == X.shape:
            isnan = isnan * weights
        counts = isnan.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        if weights is not None and weights.shape != X.shape:
            counts = counts * weights
    else:
        if any(attr is not None for attr in [axis, dtype]) or \
                        keepdims is not False:
            raise ValueError('Arguments axis, dtype and keepdims'
                             'are not yet supported on sparse data!')

        counts = _count_nans_per_row_sparse(X, weights)
    return counts


def contingency(X, y, max_X=None, max_y=None, weights=None, mask=None):
    """
    Compute the contingency matrices for each column of X (excluding the masked)
    versus the vector y.

    If the array is 1-dimensional, a 2d contingency matrix is returned. If the
    array is 2d, the function returns a 3d array, with the first dimension
    corresponding to column index (variable in the input array).

    The rows of contingency matrix correspond to values of variables, the
    columns correspond to values in vector `y`.
    (??? isn't it the other way around ???)

    Rows in the input array can be weighted (argument `weights`). A subset of
    columns can be selected by additional argument `mask`.

    The function also returns a count of NaN values per each value of `y`.

    Parameters
    ----------
    X : array_like
        With values in columns.
    y : 1d array
        Vector of true values.
    max_X : int
        The maximal value in the array
    max_y : int
        The maximal value in `y`
    weights : ...
    mask : sequence
        Discrete columns of X.

    Returns
    -------
    contingencies: (m × ny × nx) array
        m number of masked (used) columns (all if mask=None), i.e.
        for each column of X;
        ny number of uniques in y,
        nx number of uniques in column of X.
    nans : array_like
        Number of nans in each column of X for each unique value of y.
    """
    if weights is not None and np.any(weights) and np.unique(weights)[0] != 1:
        raise ValueError('weights not yet supported')

    was_1d = False
    if X.ndim == 1:
        X = X[..., np.newaxis]
        was_1d = True

    contingencies, nans = [], []
    ny = np.unique(y).size if max_y is None else max_y + 1
    for i in range(X.shape[1]):
        if mask is not None and not mask[i]:
            contingencies.append(np.zeros((ny, max_X + 1)))
            nans.append(np.zeros(ny))
            continue
        col = X[..., i]
        nx = np.unique(col[~np.isnan(col)]).size if max_X is None else max_X + 1
        if sp.issparse(col):
            col = np.ravel(col.todense())
        contingencies.append(
            bincount(y + ny * col,
                     minlength=ny * nx)[0].reshape(nx, ny).T)
        nans.append(
            bincount(y[np.isnan(col)], minlength=ny)[0])
    if was_1d:
        return contingencies[0], nans[0]
    return np.array(contingencies), np.array(nans)


def stats(X, weights=None, compute_variance=False):
    """
    Compute min, max, #nans, mean and variance.

    Result is a tuple (min, max, mean, variance, #nans, #non-nans) or an
    array of shape (len(X), 6).

    The mean and the number of nans and non-nans are weighted.

    Computation of variance requires an additional pass and is not enabled
    by default. Zeros are filled in instead of variance.

    Parameters
    ----------
    X : array_like, 1 or 2 dimensions
        Input array.
    weights : array_like, optional
        Weights, array of the same length as `x`.
    compute_variance : bool, optional
        If set to True, the function also computes variance.

    Returns
    -------
    out : a 6-element tuple or an array of shape (len(x), 6)
        Computed (min, max, mean, variance or 0, #nans, #non-nans)

    Raises
    ------
    ValueError
        If the length of the weight vector does not match the length of the
        array
    """
    is_numeric = np.issubdtype(X.dtype, np.number)
    is_sparse = sp.issparse(X)
    weighted = weights is not None and X.dtype != object

    def weighted_mean():
        if is_sparse:
            w_X = X.multiply(sp.csr_matrix(np.c_[weights] / sum(weights)))
            return np.asarray(w_X.sum(axis=0)).ravel()
        else:
            return np.nansum(X * np.c_[weights] / sum(weights), axis=0)

    if X.size and is_numeric and not is_sparse:
        nans = np.isnan(X).sum(axis=0)
        return np.column_stack((
            np.nanmin(X, axis=0),
            np.nanmax(X, axis=0),
            np.nanmean(X, axis=0) if not weighted else weighted_mean(),
            np.nanvar(X, axis=0) if compute_variance else np.zeros(X.shape[1]),
            nans,
            X.shape[0] - nans))
    elif is_sparse and X.size:
        if compute_variance:
            raise NotImplementedError

        non_zero = np.bincount(X.nonzero()[1], minlength=X.shape[1])
        X = X.tocsc()
        return np.column_stack((
            nanmin(X, axis=0),
            nanmax(X, axis=0),
            nanmean(X, axis=0) if not weighted else weighted_mean(),
            np.zeros(X.shape[1]),      # variance not supported
            X.shape[0] - non_zero,
            non_zero))
    else:
        nans = (~X.astype(bool)).sum(axis=0) if X.size else np.zeros(X.shape[1])
        return np.column_stack((
            np.tile(np.inf, X.shape[1]),
            np.tile(-np.inf, X.shape[1]),
            np.zeros(X.shape[1]),
            np.zeros(X.shape[1]),
            nans,
            X.shape[0] - nans))


def _sparse_has_zeros(x):
    """ Check if sparse matrix contains any implicit zeros. """
    return np.prod(x.shape) != x.nnz


def _nan_min_max(x, func, axis=0):
    if not sp.issparse(x):
        return func(x, axis=axis)
    if axis is None:
        extreme = func(x.data, axis=axis) if x.nnz else float('nan')
        if _sparse_has_zeros(x):
            extreme = func([0, extreme])
        return extreme
    if axis == 0:
        x = x.T
    else:
        assert axis == 1

    # TODO check & transform to correct format
    r = []
    for row in x:
        values = row.data
        extreme = func(values) if values.size else float('nan')
        if _sparse_has_zeros(row):
            extreme = func([0, extreme])
        r.append(extreme)
    return np.array(r)


def nanmin(x, axis=None):
    """ Equivalent of np.nammin that supports sparse or dense matrices. """
    return _nan_min_max(x, np.nanmin, axis)


def nanmax(x, axis=None):
    """ Equivalent of np.nammax that supports sparse or dense matrices. """
    return _nan_min_max(x, np.nanmax, axis)


def mean(x):
    """ Equivalent of np.mean that supports sparse or dense matrices. """
    if not sp.issparse(x):
        return np.mean(x)

    n_values = np.prod(x.shape)
    return np.sum(x.data) / n_values

def nanmean(x, axis=None):
    """ Equivalent of np.nanmean that supports sparse or dense matrices. """
    def nanmean_sparse(x):
        n_values = np.prod(x.shape) - np.sum(np.isnan(x.data))
        return np.nansum(x.data) / n_values

    if not sp.issparse(x):
        return np.nanmean(x, axis=axis)
    if axis is None:
        return nanmean_sparse(x)
    if axis in [0, 1]:
        arr = x if axis == 1 else x.T
        arr = arr.tocsr()
        return np.array([nanmean_sparse(row) for row in arr])
    else:
        raise NotImplementedError

def unique(x, return_counts=False):
    """ Equivalent of np.unique that supports sparse or dense matrices. """
    if not sp.issparse(x):
        return np.unique(x, return_counts=return_counts)

    implicit_zeros = np.prod(x.shape) - x.nnz
    explicit_zeros = not np.all(x.data)
    r = np.unique(x.data, return_counts=return_counts)
    if not implicit_zeros:
        return r
    if return_counts:
        if explicit_zeros:
            r[1][r[0] == 0.] += implicit_zeros
            return r
        return np.insert(r[0], 0, 0), np.insert(r[1], 0, implicit_zeros)
    else:
        if explicit_zeros:
            return r
        return np.insert(r, 0, 0)
