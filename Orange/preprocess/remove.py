from collections import namedtuple

import numpy as np

from Orange.data import Domain, DiscreteVariable
from Orange.preprocess.transformation import Lookup
from Orange.statistics.util import nanunique
from .preprocess import Preprocess

__all__ = ["Remove"]


class Remove(Preprocess):
    """
    Construct a preprocessor for removing constant features/classes
    and unused values.
    Given a data table, preprocessor returns a new table and a list of
    results. In the new table, the constant features/classes and unused
    values are removed. The list of results consists of two dictionaries.
    The first one contains numbers of 'removed', 'reduced' and 'sorted'
    features. The second one contains numbers of 'removed', 'reduced'
    and 'sorted' features.

    Parameters
    ----------
    attr_flags : int (default: 0)
        If SortValues, values of discrete attributes are sorted.
        If RemoveConstant, unused attributes are removed.
        If RemoveUnusedValues, unused values are removed from discrete
        attributes.
        It is possible to merge operations in one by summing several types.

    class_flags: int (default: 0)
        If SortValues, values of discrete class attributes are sorted.
        If RemoveConstant, unused class attributes are removed.
        If RemoveUnusedValues, unused values are removed from discrete
        class attributes.
        It is possible to merge operations in one by summing several types.

    Examples
    --------
    >>> from Orange.data import Table
    >>> from Orange.preprocess import Remove
    >>> data = Table("zoo")[:10]
    >>> flags = sum([Remove.SortValues, Remove.RemoveConstant, Remove.RemoveUnusedValues])
    >>> remover = Remove(attr_flags=flags, class_flags=flags)
    >>> new_data = remover(data)
    >>> attr_results, class_results = remover.attr_results, remover.class_results
    """

    SortValues, RemoveConstant, RemoveUnusedValues = 1, 2, 4

    def __init__(self, attr_flags=0, class_flags=0, meta_flags=0):
        self.attr_flags = attr_flags
        self.class_flags = class_flags
        self.meta_flags = meta_flags
        self.attr_results = None
        self.class_results = None
        self.meta_results = None

    def __call__(self, data):
        """
        Removes unused features or classes from the given data. Returns a new
        data table.

        Parameters
        ----------
        data : Orange.data.Table
            A data table to remove features or classes from.

        Returns
        -------
        data : Orange.data.Table
            New data table.
        """
        if data is None:
            return None

        domain = data.domain
        attrs_state = [purge_var_M(var, data, self.attr_flags)
                       for var in domain.attributes]
        class_state = [purge_var_M(var, data, self.class_flags)
                       for var in domain.class_vars]
        metas_state = [purge_var_M(var, data, self.meta_flags)
                       for var in domain.metas]

        att_vars, self.attr_results = self.get_vars_and_results(attrs_state)
        cls_vars, self.class_results = self.get_vars_and_results(class_state)
        meta_vars, self.meta_results = self.get_vars_and_results(metas_state)

        domain = Domain(att_vars, cls_vars, meta_vars)
        return data.transform(domain)

    def get_vars_and_results(self, state):
        removed, reduced, sorted = 0, 0, 0
        vars = []
        for st in state:
            removed += is_removed(st)
            reduced += not is_removed(st) and is_reduced(st)
            sorted += not is_removed(st) and is_sorted(st)
            if not is_removed(st):
                vars.append(merge_transforms(st).var)
        res = {'removed': removed, 'reduced': reduced, 'sorted': sorted}
        return vars, res


# Define a simple Purge expression 'language'.
#: A input variable (leaf expression).
Var = namedtuple("Var", ["var"])
#: Removed variable (can only ever be present as a root node).
Removed = namedtuple("Removed", ["sub", "var"])
#: A reduced variable
Reduced = namedtuple("Reduced", ["sub", "var"])
#: A sorted variable
Sorted = namedtuple("Sorted", ["sub", "var"])
#: A general (lookup) transformed variable.
#: (this node is returned as a result of `merge` which joins consecutive
#: Removed/Reduced nodes into a single Transformed node)
Transformed = namedtuple("Transformed", ["sub", "var"])


def is_var(exp):
    """Is `exp` a `Var` node."""
    return isinstance(exp, Var)


def is_removed(exp):
    """Is `exp` a `Removed` node."""
    return isinstance(exp, Removed)


def _contains(exp, cls):
    """Does `node` contain a sub node of type `cls`"""
    if isinstance(exp, cls):
        return True
    elif isinstance(exp, Var):
        return False
    else:
        return _contains(exp.sub, cls)


def is_reduced(exp):
    """Does `exp` contain a `Reduced` node."""
    return _contains(exp, Reduced)


def is_sorted(exp):
    """Does `exp` contain a `Reduced` node."""
    return _contains(exp, Sorted)


def merge_transforms(exp):
    """
    Merge consecutive Removed, Reduced or Transformed nodes.

    .. note:: Removed nodes are returned unchanged.

    """
    if isinstance(exp, (Var, Removed)):
        return exp
    elif isinstance(exp, (Reduced, Sorted, Transformed)):
        prev = merge_transforms(exp.sub)
        if isinstance(prev, (Reduced, Sorted, Transformed)):
            B = exp.var.compute_value
            assert isinstance(B, Lookup)
            A = B.variable.compute_value
            assert isinstance(A, Lookup)

            new_var = DiscreteVariable(
                exp.var.name,
                values=exp.var.values,
                compute_value=merge_lookup(A, B),
                sparse=exp.var.sparse,
            )
            assert isinstance(prev.sub, Var)
            return Transformed(prev.sub, new_var)
        else:
            assert prev is exp.sub
            return exp
    else:
        raise TypeError


def purge_var_M(var, data, flags):
    state = Var(var)
    if flags & Remove.RemoveConstant:
        var = remove_constant(state.var, data)
        if var is None:
            return Removed(state, state.var)

    if state.var.is_discrete:
        if flags & Remove.RemoveUnusedValues:
            newattr = remove_unused_values(state.var, data)

            if newattr is not state.var:
                state = Reduced(state, newattr)

            if flags & Remove.RemoveConstant and len(state.var.values) < 2:
                return Removed(state, state.var)

        if flags & Remove.SortValues:
            newattr = sort_var_values(state.var)
            if newattr is not state.var:
                state = Sorted(state, newattr)

    return state


def has_at_least_two_values(data, var):
    ((dist, unknowns),) = data._compute_distributions([var])
    # TODO this check is suboptimal for sparse since get_column_view
    # densifies the data. Should be irrelevant after Pandas.
    _, sparse = data.get_column_view(var)
    if var.is_continuous:
        dist = dist[1, :]
    min_size = 0 if sparse and unknowns else 1
    return np.sum(dist > 0.0) > min_size


def remove_constant(var, data):
    if var.is_continuous:
        if not has_at_least_two_values(data, var):
            return None
        else:
            return var
    elif var.is_discrete:
        if len(var.values) < 2:
            return None
        else:
            return var
    else:
        return var


def remove_unused_values(var, data):
    unique = nanunique(data.get_column_view(var)[0].astype(float)).astype(int)
    if len(unique) == len(var.values):
        return var
    used_values = [var.values[i] for i in unique]
    translation_table = np.array([np.NaN] * len(var.values))
    translation_table[unique] = range(len(used_values))
    return DiscreteVariable(var.name, values=used_values, sparse=var.sparse,
                            compute_value=Lookup(var, translation_table))


def sort_var_values(var):
    newvalues = list(sorted(var.values))

    if newvalues == list(var.values):
        return var

    translation_table = np.array(
        [float(newvalues.index(value)) for value in var.values]
    )

    return DiscreteVariable(var.name, values=newvalues,
                            compute_value=Lookup(var, translation_table),
                            sparse=var.sparse)


def merge_lookup(A, B):
    """
    Merge two consecutive Lookup transforms into one.
    """
    lookup_table = np.array(A.lookup_table)
    mask = np.isfinite(lookup_table)
    indices = np.array(lookup_table[mask], dtype=int)
    lookup_table[mask] = B.lookup_table[indices]
    return Lookup(A.variable, lookup_table)
