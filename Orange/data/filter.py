from numbers import Real
import random

from ..misc.enum import Enum
import numpy as np
import bottleneck as bn
from Orange.data import Instance, Storage

class Filter:
    """
    Instances of classes derived from `Filter` are used for filtering the data.

    When called with an individual data instance (:obj:Orange.data.Instance),
    they accept or reject the instance by returning either `True` or `False`.

    When called with a data storage (e.g. an instance of
    :obj:Orange.data.Table) they check whether the corresponding class
    provides the method that implements the particular filter. If so, the
    method is called and the result should be of the same type as the
    storage; e.g., filter methods of :obj:Orange.data.Table return new
    instances of :obj:Orange.data.Table, and filter methods of SQL proxies
    return new SQL proxies.

    If the class corresponding to the storage does not implement a particular
    filter, the fallback computes the indices of the rows to be selected and


    .. attribute:: negate

        If `True`, it reverts the selection
    """
    def __init__(self, negate=False):
        self.negate = negate

    def __call__(self, data):
        return


class IsDefined(Filter):
    def __init__(self, columns=None, negate=False):
        super().__init__(negate)
        self.columns = columns

    def __call__(self, data):
        if isinstance(data, Instance):
            return self.negate != bn.anynan(data._values)
        if isinstance(data, Storage):
            try:
                return data._filter_is_defined(self.columns, self.negate)
            except NotImplementedError:
                pass

        r = np.fromiter((bn.anynan(inst._values) for inst in data),
                        dtype=bool, count=len(data))
        if self.negate:
            r = np.logical_not(r)
        return data[r]


class HasClass(Filter):
    def __call__(self, data):
        if isinstance(data, Instance):
            return self.negate != bn.anynan(data._y)
        if isinstance(data, Storage):
            try:
                return data._filter_has_class(self.negate)
            except NotImplementedError:
                pass

        r = np.fromiter((bn.anynan(inst._y) for inst in data), bool, len(data))
        if self.negate:
            r = np.logical_not(r)
        return data[r]


class Random(Filter):
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
        n = self.prob if self.prob >= 1 else self.prob*len(data)
        if self.negate:
            retain[n:] = True
        else:
            retain[:n] = True
        np.random.shuffle(retain)
        return data[retain]


class SameValue(Filter):
    def __init__(self, column, value, negate=False):
        super().__init__(negate)
        self.column = column
        self.value = value

    def __call__(self, data):
        if isinstance(data, Instance):
            return self.negate != (data[self.column] == self.value)
        if isinstance(data, Storage):
            try:
                data._filter_same_value(self.column, self.value, self.negate)
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
                    (inst._values[column] != value for inst in data),
                     bool, len(data))
            else:
                retain = np.fromiter(
                    (inst._values[column] == value for inst in data),
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
    def __init__(self, conditions=[], conjunction=True, negate=False):
        super().__init__(negate)
        self.conjunction = conjunction
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



# TODO Implement __call__ methods for filters below

class ValueFilter:
    Operator = Enum("Equal", "NotEqual",
                    "Less", "LessEqual", "Greater", "GreaterEqual",
                    "Between", "Outside",
                    "Contains", "BeginsWith", "EndsWith")

    def __init__(self, position):
        self.position = position


class FilterDiscrete(ValueFilter):
    def __init__(self, position, values):
        super().__init__(position)
        self.values = values


class FilterContinuous(ValueFilter):
    def __init__(self, position, oper, min=None, max=None,
                 case_sensitive=True, **a):
        super().__init__(position)
        if a:
            if len(a) != 1 or "ref" not in a:
                raise TypeError(
                    "FilterContinuous got unexpected keyword arguments")
            else:
                min = a["ref"]
        self.min = min
        self.max = max
        self.oper = oper
        self.case_sensitive = True

    def get_ref(self):
        return self.min

    def set_ref(self, value):
        self.min = value

    ref = property(get_ref, set_ref)


class FilterString(ValueFilter):
    def __init__(self, position, oper, min=None, max=None,
                 case_sensitive=True, **a):
        super().__init__(position)
        if a:
            if len(a) != 1 or "ref" not in a:
                raise TypeError(
                    "FilterContinuous got unexpected keyword arguments")
            else:
                min = a["ref"]
        self.min = min
        self.max = max
        self.oper = oper
        self.case_sensitive = True

    def get_ref(self):
        return self.min

    def set_ref(self, value):
        self.min = value

    ref = property(get_ref, set_ref)


class FilterStringList(ValueFilter):
    def __init__(self, position, values, case_sensitive=True):
        super().__init__(position)
        self.values = values
        self.case_sensitive = case_sensitive

"""

from orange import\
    Filter as Filter,\
    FilterList as FilterList,\
    Filter_random as Random,\
    Filter_isDefined as IsDefined,\
    Filter_hasClassValue as HasClassValue,\
    Filter_hasMeta as HasMeta,\
    Filter_sameValue as SameValue,\
    Filter_values as Values,\
    Filter_hasSpecial as HasSpecial,\
    Filter_conjunction as Conjunction,\
    Filter_disjunction as Disjunction
"""
