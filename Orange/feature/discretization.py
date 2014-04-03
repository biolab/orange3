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
    """ Interval discretizer. The lower limits
    are inclusive, the upper exclusive. """
    def __init__(self, variable, points):
        super().__init__(variable)
        self.points = points

    def _transform(self, c):
        return np.where(np.isnan(c), np.NaN, np.digitize(c, self.points))


def _discretized_var(data, var, points):
    name = "D_" + data.domain[var].name
    var = data.domain[var]

    if len(points) >= 1:
        values = [ "<%f" % points[0] ] \
            + [ "[%f, %f)" % (p1, p2) for p1,p2 in zip(points, points[1:]) ] \
            + [ ">=%f" % points[-1] ]
    else:
        values = [ "single value" ]

    dvar = Orange.data.variable.DiscreteVariable(name=name, values=values)
    dvar.get_value_from = Discretizer(var, points)

    def discretized_attribute():
        if len(points) >= 1:
            sql = [ 'CASE' ]
            sql.extend([ 'WHEN "%s" < %f THEN \'%s\'' % (var.name, points[0], values[0]) ])
            sql.extend([ 'WHEN "%s" >= %f AND "%s" < %f THEN \'%s\'' % (var.name, p1, var.name, p2, values[i+1]) for i, (p1, p2) in enumerate(zip(points, points[1:])) ])
            sql.extend([ 'WHEN "%s" >= %f THEN \'%s\'' % (var.name, points[-1], values[-1]) ])
            sql.extend([ 'END' ])
        else:
            sql = [ "'%s'" % values[0] ]

        return " ".join(sql)

    dvar.to_sql = discretized_attribute

    return dvar


class Discretization:
    """Base class for discretization classes.
    """
    pass


class EqualFreq(Discretization):
    """ Discretizes the feature by spliting its domain to a fixed number of
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

            sql = "select EqualFreq((%s), (%s), (%s), (%s));"
            param = (data.table_name, attribute.name, filters if filters else None, self.n)
            cur = data._execute_sql_query(sql, param)

            points = [a for a, in cur.fetchall()]
        else:
            d = Orange.statistics.distribution.get_distribution(data, attribute)
            points = _discretization.split_eq_freq(d, n=self.n)
        return _discretized_var(data, attribute, points)


class EqualWidth(Discretization):
    """ Infers the cut-off points so that the discretization intervals contain
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

                sql = "select EqualWidth((%s), (%s), (%s), (%s));"
                param = (data.table_name, attribute.name, filters if filters else None, self.n)
                cur = data._execute_sql_query(sql, param)

                points = [a for a, in cur.fetchall()]
            else:
                d = Orange.statistics.distribution.get_distribution(data, attribute)
                points = _split_eq_width(d, n=self.n)
        return _discretized_var(data, attribute, points)

