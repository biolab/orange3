import numpy as np
import Orange.statistics.distribution
import Orange.statistics.contingency
import numpy

from Orange.feature.transformation import ColumnTransformation
from Orange.data.sql.table import SqlTable

from ..feature import _discretization

def _split_eq_width(dist, n):
    min = dist[0][0]
    max = dist[0][-1]
    if min == max:
        return []
    dif = (max-min)/n
    return [ min + (i+1)*dif for i in range(n-1) ]

def _split_eq_width_fixed(min, max, n):
    if min == max:
        return []
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


#MDL-Entropy discretization

import numpy

def normalize(X, axis=None, out=None):
    """
    Normalize `X` array so it sums to 1.0 over the `axis`.

    Parameters
    ----------
    X : array
        Array to normalize.
    axis : optional int
        Axis over which the resulting array sums to 1.
    out : optional array
        Output array of the same shape as X.
    """
    X = numpy.asarray(X, dtype=float)
    scale = numpy.sum(X, axis=axis, keepdims=True)
    if out is None:
        return X / scale
    else:
        if out is not X:
            assert out.shape == X.shape
            out[:] = X
        out /= scale
        return out


def entropy_normalized(D, axis=None):
    """
    Compute the entropy of distribution array `D`.

    `D` must be a distribution (i.e. sum to 1.0 over `axis`)

    Parameters
    ----------
    D : array
        Distribution.
    axis : optional int
        Axis of `D` along which to compute the entropy.

    """
    # req: (numpy.sum(D, axis=axis) >= 0).all()
    # req: (numpy.sum(D, axis=axis) <= 1).all()
    # req: numpy.all(numpy.abs(numpy.sum(D, axis=axis) - 1) < 1e-9)

    D = numpy.asarray(D)
    Dc = numpy.clip(D, numpy.finfo(D.dtype).eps, 1.0)
    return - numpy.sum(D * numpy.log2(Dc), axis=axis)


def entropy(D, axis=None):
    """
    Compute the entropy of distribution `D`.

    Parameters
    ----------
    D : array
        Distribution.
    axis : optional int
        Axis of `D` along which to compute the entropy.

    """
    D = normalize(D, axis=axis)
    return entropy_normalized(D, axis=axis)


def entropy_cuts_sorted(CS):
    """
    Return the class information entropy induced by partitioning
    the `CS` distribution at all N-1 candidate cut points.

    Parameters
    ----------
    CS : (N, K) array of class distributions.
    """
    CS = numpy.asarray(CS)
    # |--|-------|--------|
    #  S1    ^       S2
    # S1 contains all points which are <= to cut point
    # Cumulative distributions for S1 and S2 (left right set)
    # i.e. a cut at index i separates the CS into S1Dist[i] and S2Dist[i]
    S1Dist = numpy.cumsum(CS, axis=0)[:-1]
    S2Dist = numpy.cumsum(CS[::-1], axis=0)[-2::-1]

    # Entropy of S1[i] and S2[i] sets
    ES1 = entropy(S1Dist, axis=1)
    ES2 = entropy(S2Dist, axis=1)

    # Number of cases in S1[i] and S2[i] sets
    S1_count = numpy.sum(S1Dist, axis=1)
    S2_count = numpy.sum(S2Dist, axis=1)

    # Number of all cases
    S_count = numpy.sum(CS)

    ES1w = ES1 * S1_count / S_count
    ES2w = ES2 * S2_count / S_count

    # E(A, T; S) Class information entropy of the partition S
    E = ES1w + ES2w

    return E, ES1, ES2


def entropy_disc(X, C):
    """
    Entropy discretization.

    :param X: (N, 1) array
    :param C: (N, K) array (class probabilities must sum(axis=1) to 1 )

    :rval:
    """
    sort_ind = numpy.argsort(X, axis=0)
    X = X[sort_ind]
    C = C[sort_ind]
    return entropy_discretize_sorted(X, C)


def entropy_discretize_sorted(C):
    """
    Entropy discretization on a sorted C.

    :param C: (N, K) array of class distributions.

    """
    E, ES1, ES2 = entropy_cuts_sorted(C)
    # TODO: Also get the left right distribution counts from
    # entropy_cuts_sorted,

    # Note the + 1
    cut_index = numpy.argmin(E) + 1

    # Distribution of classed in S1, S2 and S
    S1_c = numpy.sum(C[:cut_index], axis=0)
    S2_c = numpy.sum(C[cut_index:], axis=0)
    S_c = S1_c + S2_c

    ES = entropy(numpy.sum(C, axis=0))
    ES1, ES2 = ES1[cut_index - 1], ES2[cut_index - 1]

    # Information gain of the best split
    Gain = ES - E[cut_index - 1]
    # Number of classes in S, S1 and S2 (with non zero counts)
    k = numpy.sum(S_c > 0)
    k1 = numpy.sum(S1_c > 0)
    k2 = numpy.sum(S2_c > 0)

    assert k > 0
    delta = numpy.log2(3 ** k - 2) - (k * ES - k1 * ES1 - k2 * ES2)
    N = numpy.sum(S_c)

    if Gain > numpy.log2(N - 1) / N + delta / N:
        # Accept the cut point and recursively split the subsets.
        left, right = [], []
        if k1 > 1 and cut_index > 1:
            left = entropy_discretize_sorted(C[:cut_index, :])
        if k2 > 1 and cut_index < len(C) - 1:
            right = entropy_discretize_sorted(C[cut_index:, :])
        return left + [cut_index] + [i + cut_index for i in right]
    else:
        return []


class EntropyMDL(Discretization):
    def __call__(self, data, attribute):
        from Orange.statistics import contingency as c
        cont = c.get_contingency(data, attribute)
        #values, I = _join_contingency(cont)
        values, I = _discretization.join_contingency(cont)
        cut_ind = numpy.array(entropy_discretize_sorted(I))
        if len(cut_ind) > 0:
            points = values[cut_ind - 1]
            return _discretized_var(data, attribute, points)
        else:
            return None


def _join_contingency(contingency): #obsolete: use _discretization.join_contingency
    """
    Join contingency list into a single ordered distribution.
    """
    import time
    k = len(contingency)
    values = numpy.r_[tuple(contingency[i][0] for i in range(k))]
    I = numpy.zeros((len(values), k))
    start = 0
    for i in range(k):
        counts = contingency[i][1]
        span = len(counts)
        I[start: start + span, i] = contingency[i][1]
        start += span

    sort_ind = numpy.argsort(values)
    values, I = values[sort_ind], I[sort_ind, :]

    last = None
    iv = -1
    for i in range(len(values)):
        if last != values[i]:
            iv += 1
            last = values[i]
            values[iv] = last
            I[iv] = I[i]
        else:
            I[iv] += I[i]
    return values[:iv+1],I[:iv+1]

