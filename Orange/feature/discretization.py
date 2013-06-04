import numpy as np
import Orange.statistics.distribution

from Orange.feature.transformation import ColumnTransformation

def _split_eq_width(dist, n):
    min = dist[0][0]
    max = dist[0][-1]
    dif = (max-min)/n
    return [ min + (i+1)*dif for i in range(n-1) ]

def _split_eq_freq(dist, n):
    """ Direct translatation from Orange 2. """

    if n >= len(dist[0]): #n is greater than distributions
        return [(v1+v2)/2 for v1,v2 in zip(dist[0], dist[0][1:])]

    N = sum(dist[1])
    toGo = n
    inthis = 0
    prevel = None
    inone = N/toGo
    points = []

    for i,(v,k) in enumerate(zip(*dist)):
        if toGo <= 1:
            break
        inthis += k
        if inthis < inone or i == 0: 
            prevel = v
        else: #current count exceeded
            if i < len(dist[0]) - 1 and inthis - inone < k / 2:
                #exceeded for less than half the current count:
                #split after current
                vn = dist[0][i+1]
                points.append((vn + v)/2)
                N -= inthis
                inthis = 0
                prevel = vn
            else:
                #split before the current value
                points.append((prevel + v)/2)
                N -= inthis - k
                inthis = k
                prevel = v
            toGo -= 1
            if toGo:
                inone = N/toGo
    return points


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

    values = [ "<%f" % points[0] ] \
        + [ "[%f, %f)" % (p1, p2) for p1,p2 in zip(points, points[1:]) ] \
        + [ ">=%f" % points[1] ]

    dvar = Orange.data.variable.DiscreteVariable(name=name, values=values)
    dvar.get_value_from = Discretizer(var, points)
    return dvar


class EqualFreq:
    def __init__(self, n=4):
        self.n = n

    def __call__(self, data, attribute):
        d = Orange.statistics.distribution.get_distribution(data, attribute)
        points = _split_eq_freq(d, n=self.n)
        return _discretized_var(data, attribute, points)


class EqualWidth:
    def __init__(self, n=4):
        self.n = n

    def __call__(self, data, attribute):
        d = Orange.statistics.distribution.get_distribution(data, attribute)
        points = _split_eq_width(d, n=self.n)
        return _discretized_var(data, attribute, points)

