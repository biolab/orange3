"""
This module provides alternatives for the few additional functions found in
and once used from the bottlechest package (fork of bottleneck).

It also patches bottleneck to contain these functions.
"""
import numpy as np
from scipy.sparse import issparse
import bottleneck as bn


def bincount(X, max_val=None, weights=None, minlength=None):
    """Return counts of values in array X.

    Works kind of like np.bincount(), except that it also supports floating
    arrays with nans.
    """
    X = np.asanyarray(X)
    if X.dtype.kind == 'f' and bn.anynan(X):
        nonnan = ~np.isnan(X)
        nans = (~nonnan).sum(axis=0)
        X = X[nonnan]
        if weights is not None:
            weights = weights[nonnan]
    else:
        nans = 0. if X.ndim == 1 else np.zeros(X.shape[1], dtype=float)
    if minlength is None and max_val is not None:
        minlength = max_val + 1
    return (np.bincount(X.astype(np.int32, copy=False),
                        weights=weights,
                        minlength=minlength).astype(float),
            nans)


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
    X = np.asanyarray(X)
    isnan = np.isnan(X)
    if weights is not None and weights.shape == X.shape:
        isnan = isnan * weights
    counts = isnan.sum(axis=axis, dtype=dtype, keepdims=keepdims)
    if weights is not None and weights.shape != X.shape:
        counts = counts * weights
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
        if issparse(col):
            col = np.ravel(col.todense())
        contingencies.append(
            bincount(y + ny * col,
                     minlength=ny * nx)[0].reshape(nx, ny).T)
        nans.append(
            bincount(y[np.isnan(col)], minlength=ny)[0])
    if was_1d:
        return contingencies[0], nans[0]
    return np.array(contingencies), np.array(nans)
