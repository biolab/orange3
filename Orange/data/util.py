"""
Data-manipulation utilities.
"""
import re
from collections import Counter
from itertools import chain
from typing import Callable

import numpy as np
import bottleneck as bn
from scipy import sparse as sp

RE_FIND_INDEX = r"(^{} \()(\d{{1,}})(\)$)"


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
        The original variable on which this compute value is set. Optional.
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
        part of computation and return new variable values.
        Subclasses need to implement this function."""
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
    if any(sp.issparse(arr) for arr in arrays):
        arrays = [sp.csc_matrix(arr) for arr in arrays]
        return sp.hstack(arrays)
    else:
        return np.hstack(arrays)


def array_equal(a1, a2):
    """array_equal that supports sparse and dense arrays with missing values"""
    if a1.shape != a2.shape:
        return False

    if not (sp.issparse(a1) or sp.issparse(a2)):  # Both dense: just compare
        return np.allclose(a1, a2, equal_nan=True)

    v1 = np.vstack(sp.find(a1)).T
    v2 = np.vstack(sp.find(a2)).T
    if not (sp.issparse(a1) and sp.issparse(a2)):  # Any dense: order indices
        v1.sort(axis=0)
        v2.sort(axis=0)
    return np.allclose(v1, v2, equal_nan=True)


def assure_array_dense(a):
    if sp.issparse(a):
        a = a.toarray()
    return np.asarray(a)


def assure_array_sparse(a, sparse_class: Callable = sp.csc_matrix):
    if not sp.issparse(a):
        # since x can be a list, cast to np.array
        # since x can come from metas with string, cast to float
        a = np.asarray(a).astype(np.float)
    return sparse_class(a)


def assure_column_sparse(a):
    # if x of shape (n, ) is passed to csc_matrix constructor or
    # sparse matrix with shape (1, n) is passed,
    # the resulting matrix is of shape (1, n) and hence we
    # need to transpose it to make it a column
    if a.ndim == 1 or a.shape[0] == 1:
        # csr matrix becomes csc when transposed
        return assure_array_sparse(a, sparse_class=sp.csr_matrix).T
    else:
        return assure_array_sparse(a, sparse_class=sp.csc_matrix)


def assure_column_dense(a):
    a = assure_array_dense(a)
    # column assignments must be (n, )
    return a.reshape(-1)


def get_indices(names, name):
    """
    Return list of indices which occur in a names list for a given name.
    :param names: list of strings
    :param name: str
    :return: list of indices
    """
    return [int(a.group(2)) for x in names
            for a in re.finditer(RE_FIND_INDEX.format(name), x)]


def get_unique_names(names, proposed):
    """
    Returns unique names for variables

    Proposed is a list of names (or a string with a single name). If any name
    already appears in `names`, the function appends an index in parentheses,
    which is one higher than the highest index at these variables. Also, if
    `names` contains any of the names with index in parentheses, this counts
    as an occurence of the name. For instance, if `names` does not contain
    `x` but it contains `x (3)`, `get_unique_names` will replace `x` with
    `x (4)`.

    If argument `names` is domain, the method observes all variables and metas.

    Function returns a string if `proposed` is a string, and a list if it's a
    list.

    The method is used in widgets like MDS, which adds two variables (`x` and
    `y`). It is desired that they have the same index. If `x`, `x (1)` and
    `x (2)` and `y` (but no other `y`'s already exist in the domain, MDS
    should append `x (3)` and `y (3)`, not `x (3)` and y (1)`.

    Args:
        names (Domain or list of str): used names
        proposed (str or list of str): proposed name

    Return:
        str or list of str
    """
    from Orange.data import Domain  # prevent cyclic import
    if isinstance(names, Domain):
        names = [var.name for var in chain(names.variables, names.metas)]
    if isinstance(proposed, str):
        return get_unique_names(names, [proposed])[0]
    indicess = [indices
                for indices in (get_indices(names, name) for name in proposed)
                if indices]
    if not (set(proposed) & set(names) or indicess):
        return proposed
    max_index = max(map(max, indicess), default=0) + 1
    return [f"{name} ({max_index})" for name in proposed]


def get_unique_names_duplicates(proposed: list) -> list:
    """
    Returns list of unique names. If a name is duplicated, the
    function appends an index in parentheses.
    """
    counter = Counter(proposed)
    temp_counter = Counter()
    names = []
    for name in proposed:
        if counter[name] > 1:
            temp_counter.update([name])
            name = f"{name} ({temp_counter[name]})"
        names.append(name)
    return names
