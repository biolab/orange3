import copy

import numpy

import Orange.data
from Orange.statistics import distribution, basic_stats
from .transformation import Transformation

__all__ = ["ReplaceUnknowns", "Average"]


class ReplaceUnknowns(Transformation):
    def __init__(self, variable, value=0):
        super().__init__(variable)
        self.value = value

    def transform(self, c):
        return numpy.where(numpy.isnan(c), self.value, c)


class Average:
    def __call__(self, data, variable, value=None):
        variable = data.domain[variable]
        if value is None:
            if variable.is_continuous:
                stats = basic_stats.BasicStats(data, variable)
                value = stats.mean
            elif variable.is_discrete:
                dist = distribution.get_distribution(data, variable)
                value = dist.modus()
            else:
                raise TypeError("Variable must be continuous or discrete")

        a = variable.copy(compute_value=ReplaceUnknowns(variable, value))
        a.to_sql = ImputeSql(variable, value)
        return a


class ImputeSql:
    def __init__(self, var, default):
        self.var = var
        self.default = default

    def __call__(self):
        return 'coalesce(%s, %s)' % (self.var.to_sql(), str(self.default))
