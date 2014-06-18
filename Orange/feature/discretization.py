import numpy as np
import Orange.statistics.distribution

from Orange.feature.transformation import ColumnTransformation
from Orange.data.sql.table import SqlTable

from ..feature import _discretization

def _split_eq_width(dist, n):
    min = dist[0][0]
    max = dist[0][-1]
    dif = (max-min)/n
    return [ min + (i+1)*dif for i in range(n-1) ]

def _split_eq_width_fixed(min, max, n):
    dif = (max-min)/n
    return [ min + (i+1)*dif for i in range(n-1) ]


class Discretizer(ColumnTransformation):
    """Interval discretizer.
    The lower limits are inclusive, the upper exclusive.
    """
    def __init__(self, variable, points):
        super().__init__(variable)
        self.points = points

    def _transform(self, c):
        return np.where(np.isnan(c), np.NaN, np.digitize(c, self.points))


def _discretized_var(data, var, points):
    name = "D_" + data.domain[var].name
    var = data.domain[var]

    if len(points) >= 1:
        values = ["<%f" % points[0]] \
            + ["[%f, %f)" % (p1, p2) for p1, p2 in zip(points, points[1:])] \
            + [">=%f" % points[-1]]
        def discretized_attribute():
            return 'bin(%s, ARRAY%s)' % (var.to_sql(), str(points))
    else:
        values = ["single_value"]
        def discretized_attribute():
            return "'%s'" % values[0]

    dvar = Orange.data.variable.DiscreteVariable(name=name, values=values)
    dvar.get_value_from = Discretizer(var, points)
    dvar.to_sql = discretized_attribute
    return dvar


class Discretization:
    """Base class for discretization classes."""
    pass


class EqualFreq(Discretization):
    """Discretizes the feature by spliting its domain to a fixed number of
    equal-width intervals. The span of original variable is the difference
    between the smallest and the largest feature value.

    .. attribute:: n

        Number of discretization intervals (default: 4).
    """
    def __init__(self, n=4):
        self.n = n

    def __call__(self, data, attribute):
        if type(data) == Orange.data.sql.table.SqlTable:
            filters = [f.to_sql() for f in data.row_filters]
            filters = [f for f in filters if f]
            att = attribute.to_sql()
            quantiles = [(i + 1) / self.n for i in range(self.n - 1)]
            cur = data._sql_query(['quantile(%s, ARRAY%s)' % (att, str(quantiles))], filters)
            points = cur.fetchone()[0]
        else:
            d = Orange.statistics.distribution.get_distribution(data, attribute)
            points = _discretization.split_eq_freq(d, n=self.n)
        return _discretized_var(data, attribute, points)


class EqualWidth(Discretization):
    """Infers the cut-off points so that the discretization intervals contain
    approximately equal number of training data instances.

    .. attribute:: n

        Number of discretization intervals (default: 4).
    """
    def __init__(self, n=4):
        self.n = n

    def __call__(self, data, attribute, fixed=None):
        if fixed:
            min, max = fixed[attribute.name]
            points = _split_eq_width_fixed(min, max, n=self.n)
        else:
            if type(data) == Orange.data.sql.table.SqlTable:
                filters = [f.to_sql() for f in data.row_filters]
                filters = [f for f in filters if f]
                att = attribute.to_sql()
                cur = data._sql_query(['min(%s)' % att, 'max(%s)' % att], filters)
                min, max = cur.fetchone()
                dif = (max - min) / self.n
                points = [min + (i + 1) * dif for i in range(self.n - 1)]
            else:
                # TODO: why is the whole distribution computed instead of just min/max
                d = Orange.statistics.distribution.get_distribution(data, attribute)
                points = _split_eq_width(d, n=self.n)
        return _discretized_var(data, attribute, points)

