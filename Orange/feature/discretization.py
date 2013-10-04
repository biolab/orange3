import numpy as np
import Orange.statistics.distribution

from Orange.feature.transformation import ColumnTransformation

from ..feature import _discretization

def _split_eq_width(dist, n):
    min = dist[0][0]
    max = dist[0][-1]
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
        sql = [ 'CASE' ]
        sql.extend([ 'WHEN "%s" < %f THEN 0' % (var.name, points[0]) ])
        sql.extend([ 'WHEN "%s" >= %f AND "%s" < %f THEN %d' % (var.name, p1, var.name, p2, i+1) for i, (p1, p2) in enumerate(zip(points, points[1:])) ])
        sql.extend([ 'WHEN "%s" >= %f THEN %d' % (var.name, points[-1], len(points)) ])
        sql.extend([ 'END' ])

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

    def __call__(self, data, attribute):
        d = Orange.statistics.distribution.get_distribution(data, attribute)
        points = _split_eq_width(d, n=self.n)
        return _discretized_var(data, attribute, points)

