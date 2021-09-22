"""
Data-manipulation utilities.
"""
import re
from collections import Counter
from itertools import chain, count
from typing import Callable, Union, List, Type

import numpy as np
import bottleneck as bn
from scipy import sparse as sp

RE_FIND_INDEX = r"(^{})( \((\d{{1,}})\))?$"


def one_hot(
        values: Union[np.ndarray, List], dtype: Type = float, dim: int = None
) -> np.ndarray:
    """Return a one-hot transform of values

    Parameters
    ----------
    values : 1d array
        Integer values (hopefully 0-max).
    dtype
        dtype of result array
    dim
        Number of columns (attributes) in the one hot encoding. This parameter
        is used when we need fixed number of columns and values does not
        reflect that number correctly, e.g. not all values from the discrete
        variable are present in values parameter.

    Returns
    -------
    result
        2d array with ones in respective indicator columns.
    """
    dim_values = int(np.max(values) + 1 if len(values) > 0 else 0)
    if dim is None:
        dim = dim_values
    elif dim < dim_values:
        raise ValueError("dim must be greater than max(values)")
    return np.eye(dim, dtype=dtype)[np.asanyarray(values, dtype=int)]


# pylint: disable=redefined-builtin
def scale(values, min=0, max=1):
    """Return values scaled to [min, max]"""
    if len(values) == 0:
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
        a = np.asarray(a).astype(float)
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
    # quick check and exit for the most common case
    if isinstance(a, np.ndarray) and len(a.shape) == 1:
        return a
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
    return [int(a.group(3) or 0) for x in filter(None, names)
            for a in re.finditer(RE_FIND_INDEX.format(re.escape(name)), x)]


def get_unique_names(names, proposed, equal_numbers=True):
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
    `y`). It is desired that they have the same index. In case when
    equal_numbers=True, if `x`, `x (1)` and `x (2)` and `y` (but no other
    `y`'s already exist in the domain, MDS should append `x (3)` and `y (3)`,
    not `x (3)` and y (1)`.

    Args:
        names (Domain or list of str): used names
        proposed (str or list of str): proposed name
        equal_numbers (bool): Add same number to all proposed names

    Return:
        str or list of str
    """
    # prevent cyclic import: pylint: disable=import-outside-toplevel
    from Orange.data import Domain
    if isinstance(names, Domain):
        names = [var.name for var in chain(names.variables, names.metas)]
    if isinstance(proposed, str):
        return get_unique_names(names, [proposed])[0]
    indices = {name: get_indices(names, name) for name in proposed}
    indices = {name: max(ind) + 1 for name, ind in indices.items() if ind}
    if not (set(proposed) & set(names) or indices):
        return proposed
    if equal_numbers:
        max_index = max(indices.values())
        return [f"{name} ({max_index})" for name in proposed]
    else:
        return [f"{name} ({indices[name]})" if name in indices else name
                for name in proposed]


def get_unique_names_duplicates(proposed: list, return_duplicated=False) -> list:
    """
    Returns list of unique names. If a name is duplicated, the
    function appends the next available index in parentheses.

    For example, a proposed list of names `x`, `x` and `x (2)`
    results in `x (3)`, `x (4)`, `x (2)`.
    """
    indices = {name: count(max(get_indices(proposed, name), default=0) + 1)
               for name, cnt in Counter(proposed).items()
               if name and cnt > 1}
    new_names = [f"{name} ({next(indices[name])})" if name in indices else name
                 for name in proposed]
    if return_duplicated:
        return new_names, list(indices)
    return new_names


def get_unique_names_domain(attributes, class_vars=(), metas=()):
    """
    Return de-duplicated names for variables for attributes, class_vars
    and metas. If a name appears more than once, the function appends
    indices in parentheses.

    Args:
        attributes (list of str): proposed names for attributes
        class_vars (list of str): proposed names for class_vars
        metas (list of str): proposed names for metas

    Returns:
        (attributes, class_vars, metas): new names
        renamed: list of names renamed variables; names appear in order of
            appearance in original lists; every name appears only once
    """
    all_names = list(chain(attributes, class_vars, metas))
    unique_names = get_unique_names_duplicates(all_names)
    # don't be smart with negative indices: they won't work for empty lists
    attributes = unique_names[:len(attributes)]
    class_vars = unique_names[len(attributes):len(attributes) + len(class_vars)]
    metas = unique_names[len(attributes) + len(class_vars):]
    # use dict, not set, to keep the order
    renamed = list(dict.fromkeys(old
                                 for old, new in zip(all_names, unique_names)
                                 if new != old))
    return (attributes, class_vars, metas), renamed


def sanitized_name(name: str) -> str:
    """
    Replace non-alphanumeric characters and leading zero with `_`.

    Args:
        name (str): proposed name

    Returns:
        name (str): new name
    """
    sanitized = re.sub(r"\W", "_", name)
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized
