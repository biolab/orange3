"""
This module provides alternatives for the few additional functions found in
and once used from the bottlechest package (fork of bottleneck).

It also patches bottleneck to contain these functions.
"""
from warnings import warn

import bottleneck as bn
import numpy as np
from scipy import sparse as sp


def _count_nans_per_row_sparse(X, weights, dtype=None):
    """ Count the number of nans (undefined) values per row. """
    if weights is not None:
        X = X.tocoo(copy=False)
        nonzero_mask = np.isnan(X.data)
        nan_rows, nan_cols = X.row[nonzero_mask], X.col[nonzero_mask]

        if weights.ndim == 1:
            data_weights = weights[nan_rows]
        else:
            data_weights = weights[nan_rows, nan_cols]

        w = sp.coo_matrix((data_weights, (nan_rows, nan_cols)), shape=X.shape)
        w = w.tocsr()

        return np.fromiter((np.sum(row.data) for row in w), dtype=dtype)

    return np.fromiter((np.isnan(row.data).sum() for row in X), dtype=dtype)


def sparse_count_implicit_zeros(x):
    """ Count the number of implicit zeros in a sparse matrix. """
    if not sp.issparse(x):
        raise TypeError('The matrix provided was not sparse.')
    return np.prod(x.shape) - x.nnz


def sparse_has_implicit_zeros(x):
    """ Check if sparse matrix contains any implicit zeros. """
    if not sp.issparse(x):
        raise TypeError('The matrix provided was not sparse.')
    return np.prod(x.shape) != x.nnz


def sparse_implicit_zero_weights(x, weights):
    """ Extract the weight values of all zeros in a sparse matrix. """
    if not sp.issparse(x):
        raise TypeError('The matrix provided was not sparse.')

    if weights.ndim == 1:
        # Match weights and x axis so `indices` will be set appropriately
        if x.shape[0] == weights.shape[0]:
            x = x.tocsc()
        elif x.shape[1] == weights.shape[0]:
            x = x.tocsr()
        n_items = np.prod(x.shape)
        zero_indices = np.setdiff1d(np.arange(n_items), x.indices, assume_unique=True)
        return weights[zero_indices]
    else:
        # Can easily be implemented using a coo_matrix
        raise NotImplementedError(
            'Computing zero weights on ndimensinal weight matrix is not implemented'
        )


def bincount(x, weights=None, max_val=None, minlength=None):
    """Return counts of values in array X.

    Works kind of like np.bincount(), except that it also supports floating
    arrays with nans.

    Parameters
    ----------
    x : array_like, 1 dimension, nonnegative ints
        Input array.
    weights : array_like, optional
        Weights, array of the same shape as x.
    max_val : int, optional
        Indicates the maximum value we expect to find in X and sets the result
        array size accordingly. E.g. if we set `max_val=2` yet the largest
        value in X is 1, the result will contain a bin for the value 2, and
        will be set to 0. See examples for usage.
    minlength : int, optional
        A minimum number of bins for the output array. See numpy docs for info.

    Returns
    -------
    Tuple[np.ndarray, int]
        Returns the bincounts and the number of NaN values.

    Examples
    --------
    In case `max_val` is provided, the return shape includes bins for these
    values as well, even if they do not appear in the data. However, this will
    not truncate the bincount if values larger than `max_count` are found.
    >>> bincount([0, 0, 1, 1, 2], max_val=4)
    (array([ 2.,  2.,  1.,  0.,  0.]), 0.0)
    >>> bincount([0, 1, 2, 3, 4], max_val=2)
    (array([ 1.,  1.,  1.,  1.,  1.]), 0.0)

    """
    # Store the original matrix before any manipulation to check for sparse
    x_original = x
    if sp.issparse(x):
        if weights is not None:
            # Match weights and x axis so `indices` will be set appropriately
            if x.shape[0] == weights.shape[0]:
                x = x.tocsc()
            elif x.shape[1] == weights.shape[0]:
                x = x.tocsr()

            zero_weights = sparse_implicit_zero_weights(x, weights).sum()
            weights = weights[x.indices]
        else:
            zero_weights = sparse_count_implicit_zeros(x)

        x = x.data

    x = np.asanyarray(x)
    if x.dtype.kind == 'f' and bn.anynan(x):
        nonnan = ~np.isnan(x)
        x = x[nonnan]
        if weights is not None:
            nans = (~nonnan * weights).sum(axis=0)
            weights = weights[nonnan]
        else:
            nans = (~nonnan).sum(axis=0)
    else:
        nans = 0. if x.ndim == 1 else np.zeros(x.shape[1], dtype=float)

    if minlength is None and max_val is not None:
        minlength = max_val + 1

    if minlength is not None and minlength <= 0:
        bc = np.array([])
    else:
        bc = np.bincount(
            x.astype(np.int32, copy=False), weights=weights, minlength=minlength
        ).astype(float)
        # Since `csr_matrix.values` only contain non-zero values or explicit
        # zeros, we must count implicit zeros separately and add them to the
        # explicit ones found before
        if sp.issparse(x_original):
            # If x contains only NaNs, then bc will be an empty array
            if zero_weights and bc.size == 0:
                bc = [zero_weights]
            elif zero_weights:
                bc[0] += zero_weights

    return bc, nans


def countnans(x, weights=None, axis=None, dtype=None, keepdims=False):
    """
    Count the undefined elements in an array along given axis.

    Parameters
    ----------
    x : array_like
    weights : array_like, optional
        Weights to weight the nans with, before or after counting (depending
        on the weights shape).
    axis : int, optional
    dtype : dtype, optional
        The data type of the returned array.

    Returns
    -------
    Union[np.ndarray, float]

    """
    if not sp.issparse(x):
        x = np.asanyarray(x)
        isnan = np.isnan(x)
        if weights is not None and weights.shape == x.shape:
            isnan = isnan * weights

        counts = isnan.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        if weights is not None and weights.shape != x.shape:
            counts = counts * weights
    else:
        assert axis in [None, 0, 1], 'Only axis 0 and 1 are currently supported'
        # To have consistent behaviour with dense matrices, raise error when
        # `axis=1` and the array is 1d (e.g. [[1 2 3]])
        if x.shape[0] == 1 and axis == 1:
            raise ValueError('Axis %d is out of bounds' % axis)

        arr = x if axis == 1 else x.T

        if weights is not None:
            weights = weights if axis == 1 else weights.T

        arr = arr.tocsr()
        counts = _count_nans_per_row_sparse(arr, weights, dtype=dtype)

        # We want a scalar value if `axis=None` or if the sparse matrix is
        # actually a vector (e.g. [[1 2 3]]), but has `ndim=2` due to scipy
        # implementation
        if axis is None or x.shape[0] == 1:
            counts = counts.sum(dtype=dtype)

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
    weights : array_like
        Row weights. When not None, contingencies contain weighted counts
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
                     minlength=ny * nx,
                     weights=weights)[0].reshape(nx, ny).T)
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


def _nan_min_max(x, func, axis=0):
    if not sp.issparse(x):
        return func(x, axis=axis)
    if axis is None:
        extreme = func(x.data, axis=axis) if x.nnz else float('nan')
        if sparse_has_implicit_zeros(x):
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
        if sparse_has_implicit_zeros(row):
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
    m = (np.sum(x.data) / np.prod(x.shape)
         if sp.issparse(x) else
         np.mean(x))
    if np.isnan(m):
        warn('mean() resulted in nan. If input can contain nan values, perhaps '
             'you meant nanmean?', stacklevel=2)
    return m


def _apply_func(x, dense_func, sparse_func, axis=None):
    """ General wrapper for a function depending on sparse or dense matrices. """
    if not sp.issparse(x):
        return dense_func(x, axis=axis)
    if axis is None:
        return sparse_func(x)
    if axis in [0, 1]:
        arr = x if axis == 1 else x.T
        arr = arr.tocsr()
        return np.fromiter((sparse_func(row) for row in arr),
                           dtype=np.double, count=arr.shape[0])
    else:
        raise NotImplementedError


def nansum(x, axis=None):
    """ Equivalent of np.nansum that supports sparse or dense matrices. """
    def nansum_sparse(x):
        return np.nansum(x.data)

    return _apply_func(x, np.nansum, nansum_sparse, axis=axis)


def nanmean(x, axis=None):
    """ Equivalent of np.nanmean that supports sparse or dense matrices. """
    def nanmean_sparse(x):
        n_values = np.prod(x.shape) - np.sum(np.isnan(x.data))
        return np.nansum(x.data) / n_values

    return _apply_func(x, np.nanmean, nanmean_sparse, axis=axis)


def nanvar(x, axis=None):
    """ Equivalent of np.nanvar that supports sparse or dense matrices. """
    def nanvar_sparse(x):
        n_values = np.prod(x.shape) - np.sum(np.isnan(x.data))
        mean = np.nansum(x.data) / n_values
        return np.nansum((x.data - mean) ** 2) / n_values

    return _apply_func(x, np.nanvar, nanvar_sparse, axis=axis)


def nanmedian(x, axis=None):
    """ Equivalent of np.nanmedian that supports sparse or dense matrices. """
    def nanmedian_sparse(x):
        nz = np.logical_not(np.isnan(x.data))
        n_nan = sum(np.isnan(x.data))
        n_nonzero = sum(x.data[nz] != 0)
        n_zeros = np.prod(x.shape) - n_nonzero - n_nan
        if n_zeros > n_nonzero:
            # Typical case if use of sparse matrices make sense
            return 0
        else:
            # Possibly contains NaNs and
            # more nz values than zeros, so allocating memory should not be too problematic
            return np.nanmedian(x.toarray())

    return _apply_func(x, np.nanmedian, nanmedian_sparse, axis=axis)


def unique(x, return_counts=False):
    """ Equivalent of np.unique that supports sparse or dense matrices. """
    if not sp.issparse(x):
        return np.unique(x, return_counts=return_counts)

    implicit_zeros = sparse_count_implicit_zeros(x)
    explicit_zeros = not np.all(x.data)
    r = np.unique(x.data, return_counts=return_counts)
    if not implicit_zeros:
        return r

    if return_counts:
        zero_index = np.searchsorted(r[0], 0)
        if explicit_zeros:
            r[1][r[0] == 0.] += implicit_zeros
            return r
        return np.insert(r[0], zero_index, 0), np.insert(r[1], zero_index, implicit_zeros)
    else:
        if explicit_zeros:
            return r
        zero_index = np.searchsorted(r, 0)
        return np.insert(r, zero_index, 0)


def nanunique(*args, **kwargs):
    """ Return unique values while disregarding missing (np.nan) values.
    Supports sparse or dense matrices. """
    result = unique(*args, **kwargs)

    if isinstance(result, tuple):
        result, counts = result
        non_nan_mask = ~np.isnan(result)
        return result[non_nan_mask], counts[non_nan_mask]

    return result[~np.isnan(result)]


def digitize(x, bins, right=False):
    """Equivalent of np.digitize that supports sparse and dense matrices.

    If a sparse matrix is provided and the '0's belong to the '0'th bin, then
    a sparse matrix is returned.

    Because this can return both sparse and dense matrices, we must keep the
    return shape consistent. Since sparse matrices don't support 1d matrices,
    we reshape any returned 1d numpy array to a 2d matrix, with the first
    dimension shape being 1. This is equivalent to the behaviour of sparse
    matrices.

    Parameters
    ----------
    x : Union[np.ndarry, sp.csr_matrix, sp.csc_matrix]
    bins : np.ndarray
    right : Optional[bool]

    Returns
    -------
    Union[np.ndarray, sp.csr_matrix]

    """
    if not sp.issparse(x):
        # TODO Remove reshaping logic when support for numpy==1.9 is dropped
        original_shape = x.shape
        x = x.flatten()
        result = np.digitize(x, bins, right)
        result = result.reshape(original_shape)
        # In order to keep the return shape consistent, and sparse matrices
        # don't support 1d matrices, make sure to convert 1d to 2d matrices
        if result.ndim == 1:
            result = result.reshape(((1,) + result.shape))
        return result

    # Find the bin where zeros belong, depending on the `right` parameter
    zero_bin = np.searchsorted(bins, 0, side=['right', 'left'][right])

    if zero_bin == 0:
        r = sp.lil_matrix(x.shape, dtype=np.int64)
    else:
        r = zero_bin * np.ones(x.shape, dtype=np.int64)

    for idx, row in enumerate(x.tocsr()):
        # TODO Remove this check when support for numpy==1.9 is dropped
        if row.nnz > 0:
            r[idx, row.indices] = np.digitize(row.data, bins, right)

    # Orange mainly deals with `csr_matrix`, but `lil_matrix` is more efficient
    # for incremental building
    if sp.issparse(r):
        r = r.tocsr()

    return r


def var(x, axis=None):
    """ Equivalent of np.var that supports sparse and dense matrices. """
    if not sp.issparse(x):
        return np.var(x, axis)

    result = x.multiply(x).mean(axis) - np.square(x.mean(axis))
    result = np.squeeze(np.asarray(result))
    return result
