"""
Data-manipulation utilities.
"""
import re
from collections import Counter, defaultdict
from itertools import chain

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
    if len(values) == 0:
        return np.zeros((0, 0), dtype=dtype)
    return np.eye(int(np.max(values) + 1), dtype=dtype)[np.asanyarray(values, dtype=int)]


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
    return a


def assure_array_sparse(a):
    if not sp.issparse(a):
        # since x can be a list, cast to np.array
        # since x can come from metas with string, cast to float
        a = np.asarray(a).astype(np.float)
        return sp.csc_matrix(a)
    return a


def assure_column_sparse(a):
    a = assure_array_sparse(a)
    # if x of shape (n, ) is passed to csc_matrix constructor,
    # the resulting matrix is of shape (1, n) and hence we
    # need to transpose it to make it a column
    if a.shape[0] == 1:
        a = a.T
    return a


def assure_column_dense(a):
    a = assure_array_dense(a)
    # column assignments must be of shape (n,) and not (n, 1)
    return np.ravel(a)


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
    # prevent cyclic import: pylint: disable=import-outside-toplevel
    from Orange.data import Domain
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
    function appends the smallest available index in parentheses.

    For example, a proposed list of names `x`, `x` and `x (2)`
    results in `x (1)`, `x (3)`, `x (2)`.
    """
    counter = Counter(proposed)
    index = defaultdict(int)
    names = []
    for name in proposed:
        if name and counter[name] > 1:
            unique_name = name
            while unique_name in counter:
                index[name] += 1
                unique_name = f"{name} ({index[name]})"
            name = unique_name
        names.append(name)
    return names


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
