"""
Data-manipulation utilities.
"""
import numpy as np
import bottleneck as bn
from scipy import sparse as sp


def one_hot(values, dtype=float):
    """Return a one-hot transform of values

    Parameters
    ----------
    values : 1d array
        Integer values (hopefully 0-max).

    Returns
    -------
    result
        2d array with ones in respective indicator columns.
    """
    if not len(values):
       return np.zeros((0, 0), dtype=dtype)
    return np.eye(int(np.max(values) + 1), dtype=dtype)[np.asanyarray(values, dtype=int)]


def scale(values, min=0, max=1):
    """Return values scaled to [min, max]"""
    if not len(values):
        return np.array([])
    minval = np.float_(bn.nanmin(values))
    ptp = bn.nanmax(values) - minval
    if ptp == 0:
        return np.clip(values, min, max)
    return (-minval + values) / ptp * (max - min) + min


class SharedComputeValue:
    """A base class that separates compute_value computation
    for different variables into shared and specific parts.

    Parameters
    ----------
    compute_shared: Callable[[Orange.data.Table], object]
        A callable that performs computation that is shared between
        multiple variables. Variables sharing computation need to set
        the same instance.
    variable: Orange.data.Variable
        The original variable on which this compute value is set.
    """

    def __init__(self, compute_shared, variable=None):
        self.compute_shared = compute_shared
        self.variable = variable

    def __call__(self, data, shared_data=None):
        """Fallback if common parts are not passed."""
        if shared_data is None:
            shared_data = self.compute_shared(data)
        return self.compute(data, shared_data)

    def compute(self, data, shared_data):
        """Given precomputed shared data, perform variable-specific
        part of computation and return new variable values."""
        raise NotImplementedError


def vstack(arrays):
    """vstack that supports sparse and dense arrays

    If all arrays are dense, result is dense. Otherwise,
    result is a sparse (csr) array.
    """
    if any(sp.issparse(arr) for arr in arrays):
        arrays = [sp.csr_matrix(arr) for arr in arrays]
        return sp.vstack(arrays)
    else:
        return np.vstack(arrays)


def hstack(arrays):
    """hstack that supports sparse and dense arrays

    If all arrays are dense, result is dense. Otherwise,
    result is a sparse (csc) array.
    """
    arrays = [a if a.ndim > 1 else a.reshape(-1, 1) for a in arrays]
    if any(arr.dtype == object for arr in arrays):
        arrays = [a.toarray() if sp.issparse(a) else a
                  for a in arrays]
        return np.hstack(arrays)
    if any(sp.issparse(arr) for arr in arrays):
        arrays = [sp.csc_matrix(arr) for arr in arrays]
        r = sp.hstack(arrays)
        density = r.nnz / np.prod(r.shape)
        print('density:', density)
        if density > 1/3:
            r = r.toarray()
        return r
    else:
        return np.hstack(arrays)
