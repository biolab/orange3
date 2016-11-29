import random
import re

from math import isnan
from numbers import Real

import numpy as np
import bottleneck as bn

from Orange.data import Instance, Storage, Variable
from Orange.misc.enum import Enum


class Filter:
    """
    The base class for filters.

    .. attribute:: negate

        Reverts the selection
    """

    def __init__(self, negate=False):
        self.negate = negate

    def __call__(self, data):
        return


class IsDefined(Filter):
    """
    Select the data instances with no undefined values. The check can be
    restricted to a subset of columns.

    The filter's behaviour may depend upon the storage implementation.

    In particular, :obj:`~Orange.data.Table` with sparse matrix representation
    will select all data instances whose values are defined, even if they are
    zero. However, if individual columns are checked, it will select all
    rows with non-zero entries for this columns, disregarding whether they
    are stored as zero or omitted.

    .. attribute:: columns

        The columns to be checked, given as a sequence of indices, names or
        :obj:`Orange.data.Variable`.
    """

    def __init__(self, columns=None, negate=False):
        super().__init__(negate)
        self.columns = columns

    def __call__(self, data):
        if isinstance(data, Instance):
            return self.negate == bn.anynan(data._x)
        if isinstance(data, Storage):
            try:
                return data._filter_is_defined(self.columns, self.negate)
            except NotImplementedError:
                pass

        r = np.fromiter((not bn.anynan(inst._x) for inst in data),
                        dtype=bool, count=len(data))
        if self.negate:
            r = np.logical_not(r)
        return data[r]


class HasClass(Filter):
    """
    Return all rows for which the class value is known.

    :obj:`Orange.data.Table` implements the filter on the sparse data so that it
    returns all rows for which all class values are defined, even if they
    equal zero.
    """

    def __call__(self, data):
        if isinstance(data, Instance):
            return self.negate == bn.anynan(data._y)
        if isinstance(data, Storage):
            try:
                return data._filter_has_class(self.negate)
            except NotImplementedError:
                pass

        r = np.fromiter((not bn.anynan(inst._y) for inst in data), bool, len(data))
        if self.negate:
            r = np.logical_not(r)
        return data[r]


class Random(Filter):
    """
    Return a random selection of data instances.

    .. attribute:: prob

        The proportion (if below 1) or the probability (if 1 or above) of
        selected instances
    """

    def __init__(self, prob=None, negate=False):
        super().__init__(negate)
        self.prob = prob

    def __call__(self, data):
        if isinstance(data, Instance):
            return self.negate != (random.random() < self.prob)
        if isinstance(data, Storage):
            try:
                return data._filter_random(self.prob, self.negate)
            except NotImplementedError:
                pass

        retain = np.zeros(len(data), dtype=bool)
        n = int(self.prob) if self.prob >= 1 else int(self.prob * len(data))
        if self.negate:
            retain[n:] = True
        else:
            retain[:n] = True
        np.random.shuffle(retain)
        return data[retain]


class SameValue(Filter):
    """
    Return the data instances with the given value in the specified column.

    .. attribute:: column

        The column, described by an index, a string or
        :obj:`Orange.data.Variable`.

    .. attribute:: value

        The reference value
    """

    def __init__(self, column, value, negate=False):
        super().__init__(negate)
        self.column = column
        self.value = value

    def __call__(self, data):
        if isinstance(data, Instance):
            return self.negate != (data[self.column] == self.value)
        if isinstance(data, Storage):
            try:
                return data._filter_same_value(self.column, self.value, self.negate)
            except NotImplementedError:
                pass

        column = data.domain.index(self.column)
        if (data.domain[column].is_primitive() and
                not isinstance(self.value, Real)):
            value = data.domain[column].to_val(self.value)
        else:
            value = self.value

        if column >= 0:
            if self.negate:
                retain = np.fromiter(
                    (inst[column] != value for inst in data),
                     bool, len(data))
            else:
                retain = np.fromiter(
                    (inst[column] == value for inst in data),
                     bool, len(data))
        else:
            column = -1 - column
            if self.negate:
                retain = np.fromiter(
                    (inst._metas[column] != value for inst in data),
                     bool, len(data))
            else:
                retain = np.fromiter(
                    (inst._metas[column] == value for inst in data),
                     bool, len(data))
        return data[retain]


class Values(Filter):
    """
    Select the data instances based on conjunction or disjunction of filters
    derived from :obj:`ValueFilter` that check values of individual features
    or another (nested) Values filter.

    .. attribute:: conditions

        A list of conditions, derived from :obj:`ValueFilter` or :obj:`Values`

    .. attribute:: conjunction

        If `True`, the filter computes a conjunction, otherwise a disjunction

    .. attribute:: negate

        Revert the selection
    """

    def __init__(self, conditions, conjunction=True, negate=False):
        super().__init__(negate)
        self.conjunction = conjunction
        if not conditions:
            raise ValueError("Filter with no conditions.")
        self.conditions = conditions

    def __call__(self, data):
        if isinstance(data, Instance):
            agg = all if self.conjunction else any
            return self.negate != agg(cond(data) for cond in self.conditions)
        if isinstance(data, Storage):
            try:
                return data._filter_values(self)
            except NotImplementedError:
                pass
        N = len(data)
        if self.conjunction:
            sel, agg = np.ones(N, bool), np.logical_and
        else:
            sel, agg = np.zeros(N, bool), np.logical_or
        for cond in self.conditions:
            sel = agg(sel, np.fromiter((cond(inst) for inst in data), bool, count=N))
        if self.negate:
            sel = np.logical_not(sel)
        return data[sel]


class ValueFilter(Filter):
    """
    The base class for subfilters that check individual values of data
    instances. Derived classes handle discrete, continuous and string
    attributes. These filters are used to compose conditions in
    :obj:`Orange.data.filter.Values`.

    The internal implementation of `filter.Values` in data storages, like
    :obj:`Orange.data.Table`, recognize these filters and retrieve their,
    attributes, like operators and reference values, but do not call them.

    The fallback implementation of :obj:`Orange.data.filter.Values` calls
    the subfilters with individual data instances, which is very inefficient.

    .. attribute:: column

        The column to which the filter applies (int, str or
        :obj:`Orange.data.Variable`).
    """

    def __init__(self, column):
        super().__init__()
        self.column = column
        self.last_domain = None

    def cache_position(self, domain):
        self.pos_cache = domain.index(self.column)
        self.last_domain = domain


class FilterDiscrete(ValueFilter):
    """
    Subfilter for discrete variables, which selects the instances whose
    value matches one of the given values.

    .. attribute:: column

        The column to which the filter applies (int, str or
        :obj:`Orange.data.Variable`).

    .. attribute:: values

        The list (or a set) of accepted values. If None, it checks whether
        the value is defined.
    """

    def __init__(self, column, values):
        super().__init__(column)
        self.values = values

    def __call__(self, inst):
        if inst.domain is not self.last_domain:
            self.cache_position(inst.domain)
        value = inst[self.pos_cache]
        if self.values is None:
            return not isnan(value)
        else:
            return value in self.values

    def __eq__(self, other):
        return isinstance(other, FilterDiscrete) and \
               self.column == other.column and self.values == other.values


class FilterContinuous(ValueFilter):
    """
    Subfilter for continuous variables.

    .. attribute:: column

        The column to which the filter applies (int, str or
        :obj:`Orange.data.Variable`).

    .. attribute:: ref

        The reference value; also aliased to `min` for operators
        `Between` and `Outside`.

    .. attribute:: max

        The upper threshold for operators `Between` and `Outside`.

    .. attribute:: oper

        The operator; should be `FilterContinuous.Equal`, `NotEqual`, `Less`,
        `LessEqual`, `Greater`, `GreaterEqual`, `Between`, `Outside` or
        `IsDefined`.
    """

    def __init__(self, position, oper, ref=None, max=None, **a):
        super().__init__(position)
        if a:
            if len(a) != 1 or "min" not in a:
                raise TypeError(
                    "FilterContinuous got unexpected keyword arguments")
            else:
                ref = a["min"]
        self.ref = ref
        self.max = max
        self.oper = oper

    @property
    def min(self):
        return self.ref

    @min.setter
    def min(self, value):
        self.ref = value

    def __call__(self, inst):
        if inst.domain is not self.last_domain:
            self.cache_position(inst.domain)
        value = inst[self.pos_cache]
        if isnan(value):
            return self.oper == self.Equal and isnan(self.ref)
        if self.oper == self.Equal:
            return value == self.ref
        if self.oper == self.NotEqual:
            return value != self.ref
        if self.oper == self.Less:
            return value < self.ref
        if self.oper == self.LessEqual:
            return value <= self.ref
        if self.oper == self.Greater:
            return value > self.ref
        if self.oper == self.GreaterEqual:
            return value >= self.ref
        if self.oper == self.Between:
            return self.ref <= value <= self.max
        if self.oper == self.Outside:
            return not self.ref <= value <= self.max
        if self.oper == self.IsDefined:
            return True
        raise ValueError("invalid operator")

    def __eq__(self, other):
        return isinstance(other, FilterContinuous) and \
               self.column == other.column and self.oper == other.oper and \
               self.ref == other.ref and self.max == other.max

    def __str__(self):
        if isinstance(self.column, str):
            column = self.column
        elif isinstance(self.column, Variable):
            column = self.column.name
        else:
            column = "feature({})".format(self.column)

        names = {self.Equal: "=", self.NotEqual: "≠",
                 self.Less: "<", self.LessEqual: "≤",
                 self.Greater: ">", self.GreaterEqual: "≥"}
        if self.oper in names:
            return "{} {} {}".format(column, names[self.oper], self.ref)
        if self.oper == self.Between:
            return "{} ≤ {} ≤ {}".format(self.min, column, self.max)
        if self.oper == self.Outside:
            return "not {} ≤ {} ≤ {}".format(self.min, column, self.max)
        if self.oper == self.IsDefined:
            return "{} is defined".format(column)
        return "invalid operator"

    __repr__ = __str__


    # For PyCharm:
    Equal = NotEqual = Less = LessEqual = Greater = GreaterEqual = 0
    Between = Outside = IsDefined = 0


Enum("Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual",
     "Between", "Outside", "IsDefined").pull_up(FilterContinuous)


class FilterString(ValueFilter):
    """
    Subfilter for string variables.

    .. attribute:: column

        The column to which the filter applies (int, str or
        :obj:`Orange.data.Variable`).

    .. attribute:: ref

        The reference value; also aliased to `min` for operators
        `Between` and `Outside`.

    .. attribute:: max

        The upper threshold for operators `Between` and `Outside`.

    .. attribute:: oper

        The operator; should be `FilterString.Equal`, `NotEqual`, `Less`,
        `LessEqual`, `Greater`, `GreaterEqual`, `Between`, `Outside`,
        `Contains`, `StartsWith`, `EndsWith` or `IsDefined`.

    .. attribute:: case_sensitive

        Tells whether the comparisons are case sensitive
    """

    def __init__(self, position, oper, ref=None, max=None,
                 case_sensitive=True, **a):
        super().__init__(position)
        if a:
            if len(a) != 1 or "min" not in a:
                raise TypeError(
                    "FilterContinuous got unexpected keyword arguments")
            else:
                ref = a["min"]
        self.ref = ref
        self.max = max
        self.oper = oper
        self.case_sensitive = case_sensitive

    @property
    def min(self):
        return self.ref

    @min.setter
    def min(self, value):
        self.ref = value

    def __call__(self, inst):
        if inst.domain is not self.last_domain:
            self.cache_position(inst.domain)
        value = inst[self.pos_cache]
        if self.oper == self.IsDefined:
            return not np.isnan(value)
        if self.case_sensitive:
            value = str(value)
            refval = str(self.ref)
        else:
            value = str(value).lower()
            refval = str(self.ref).lower()
        if self.oper == self.Equal:
            return value == refval
        if self.oper == self.NotEqual:
            return value != refval
        if self.oper == self.Less:
            return value < refval
        if self.oper == self.LessEqual:
            return value <= refval
        if self.oper == self.Greater:
            return value > refval
        if self.oper == self.GreaterEqual:
            return value >= refval
        if self.oper == self.Contains:
            return refval in value
        if self.oper == self.StartsWith:
            return value.startswith(refval)
        if self.oper == self.EndsWith:
            return value.endswith(refval)
        high = self.max if self.case_sensitive else self.max.lower()
        if self.oper == self.Between:
            return refval <= value <= high
        if self.oper == self.Outside:
            return not refval <= value <= high
        raise ValueError("invalid operator")

    # For PyCharm:
    Equal = NotEqual = Less = LessEqual = Greater = GreaterEqual = 0
    Between = Outside = Contains = StartsWith = EndsWith = IsDefined = 0


Enum("Equal", "NotEqual",
     "Less", "LessEqual", "Greater", "GreaterEqual",
     "Between", "Outside",
     "Contains", "StartsWith", "EndsWith",
     "IsDefined").pull_up(FilterString)


class FilterStringList(ValueFilter):
    """
    Subfilter for strings variables which checks whether the value is in the
    given list of accepted values.

    .. attribute:: column

        The column to which the filter applies (int, str or
        :obj:`Orange.data.Variable`).

    .. attribute:: values

        The list (or a set) of accepted values.

    .. attribute:: case_sensitive

        Tells whether the comparisons are case sensitive
    """

    def __init__(self, column, values, case_sensitive=True):
        super().__init__(column)
        self.values = values
        self.case_sensitive = case_sensitive

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values
        self.values_lower = [x.lower() for x in values]

    def __call__(self, inst):
        if inst.domain is not self.last_domain:
            self.cache_position(inst.domain)
        value = inst[self.pos_cache]
        if self.case_sensitive:
            return value in self._values
        else:
            return value.lower() in self.values_lower


class FilterRegex(ValueFilter):
    """Filter that checks whether the values match the regular expression."""
    def __init__(self, column, pattern, flags=0):
        super().__init__(column)
        self._re = re.compile(pattern, flags)

    def __call__(self, inst):
        return bool(self._re.search(inst or ''))
