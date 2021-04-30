import calendar
import re
import time
from typing import NamedTuple, List, Union, Callable
import datetime
from itertools import count

import numpy as np
import scipy.sparse as sp

from Orange.data import DiscreteVariable, Domain
from Orange.data.sql.table import SqlTable
from Orange.statistics import distribution, contingency, util as ut
from Orange.statistics.basic_stats import BasicStats
from Orange.util import Reprable, utc_from_timestamp
from .transformation import Transformation
from . import _discretize

__all__ = ["EqualFreq", "EqualWidth", "EntropyMDL", "DomainDiscretizer",
           "decimal_binnings", "time_binnings", "short_time_units",
           "BinDefinition"]


class Discretizer(Transformation):
    """Value transformer that returns an index of the bin for the given value.
    """
    def __init__(self, variable, points):
        super().__init__(variable)
        self.points = points

    @staticmethod
    def digitize(x, bins):
        if sp.issparse(x):
            if len(bins):
                x.data = np.digitize(x.data, bins)
            else:
                x = sp.csr_matrix(x.shape)
            return x
        else:
            return np.digitize(x, bins) if len(bins) else [0]*len(x)

    def transform(self, c):
        if sp.issparse(c):
            return self.digitize(c, self.points)
        elif c.size:
            return np.where(np.isnan(c), np.NaN, self.digitize(c, self.points))
        else:
            return np.array([], dtype=int)

    @staticmethod
    def _fmt_interval(low, high, formatter):
        assert low is not None or high is not None
        assert low is None or high is None or low < high
        if low is None or np.isinf(low):
            return f"< {formatter(high)}"
        if high is None or np.isinf(high):
            return f"≥ {formatter(low)}"
        return f"{formatter(low)} - {formatter(high)}"

    @classmethod
    def create_discretized_var(cls, var, points):
        def fmt(val):
            sval = var.str_val(val)
            # For decimal numbers, remove trailing 0's and . if no decimals left
            if re.match(r"^\d+\.\d+", sval):
                return sval.rstrip("0").rstrip(".")
            return sval

        lpoints = list(points)
        if lpoints:
            values = [
                cls._fmt_interval(low, high, fmt)
                for low, high in zip([-np.inf] + lpoints, lpoints + [np.inf])]
            to_sql = BinSql(var, lpoints)
        else:
            values = ["single_value"]
            to_sql = SingleValueSql(values[0])

        dvar = DiscreteVariable(name=var.name, values=values,
                                compute_value=cls(var, points),
                                sparse=var.sparse)
        dvar.source_variable = var
        dvar.to_sql = to_sql
        return dvar

    def __eq__(self, other):
        return super().__eq__(other) and self.points == other.points

    def __hash__(self):
        return hash((type(self), self.variable, tuple(self.points)))


class BinSql:
    def __init__(self, var, points):
        self.var = var
        self.points = points

    def __call__(self):
        return 'width_bucket(%s, ARRAY%s::double precision[])' % (
            self.var.to_sql(), str(self.points))


class SingleValueSql:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return "'%s'" % self.value


class Discretization(Reprable):
    """Abstract base class for discretization classes."""
    def __call__(self, data, variable):
        """
        Compute discretization of the given variable on the given data.
        Return a new variable with the appropriate domain
        (:obj:`Orange.data.DiscreteVariable.values`) and transformer
        (:obj:`Orange.data.Variable.compute_value`).
        """
        raise NotImplementedError(
            "Subclasses of 'Discretization' need to implement "
            "the call operator")


class EqualFreq(Discretization):
    """Discretization into bins with approximately equal number of data
    instances.

    .. attribute:: n

        Number of bins (default: 4). The actual number may be lower if the
        variable has less than n distinct values.
    """
    def __init__(self, n=4):
        self.n = n

    # noinspection PyProtectedMember
    def __call__(self, data, attribute):
        if type(data) == SqlTable:
            att = attribute.to_sql()
            quantiles = [(i + 1) / self.n for i in range(self.n - 1)]
            query = data._sql_query(
                ['quantile(%s, ARRAY%s)' % (att, str(quantiles))],
                use_time_sample=1000)
            with data._execute_sql_query(query) as cur:
                points = sorted(set(cur.fetchone()[0]))
        else:
            d = distribution.get_distribution(data, attribute)
            points = _discretize.split_eq_freq(d, self.n)
            # np.unique handles cases in which differences are below precision
            points = list(np.unique(points))
        return Discretizer.create_discretized_var(
            data.domain[attribute], points)

class EqualWidth(Discretization):
    """Discretization into a fixed number of bins with equal widths.

    .. attribute:: n

        Number of bins (default: 4).
    """
    def __init__(self, n=4):
        self.n = n

    # noinspection PyProtectedMember
    def __call__(self, data, attribute, fixed=None):
        if fixed:
            min, max = fixed[attribute.name]
            points = self._split_eq_width(min, max)
        else:
            if type(data) == SqlTable:
                stats = BasicStats(data, attribute)
                points = self._split_eq_width(stats.min, stats.max)
            else:
                values = data[:, attribute]
                values = values.X if values.X.size else values.Y
                if values.size:
                    min, max = ut.nanmin(values), ut.nanmax(values)
                    points = self._split_eq_width(min, max)
                else:
                    points = []
        return Discretizer.create_discretized_var(
            data.domain[attribute], points)

    def _split_eq_width(self, min, max):
        if np.isnan(min) or np.isnan(max) or min == max:
            return []
        dif = (max - min) / self.n
        return [min + (i + 1) * dif for i in range(self.n - 1)]


class BinDefinition(NamedTuple):
    thresholds: np.ndarray  # thresholds, including the top
    labels: List[str]  # friendly-formatted thresholds
    short_labels: List[str]  # shorter labels (e.g. simplified dates)
    width: Union[float, None]  # widths, if uniform; otherwise None
    width_label: str  # friendly-formatted width (e.g. '50' or '2 weeks')


# NamedTupleMeta doesn't allow to define __new__ so we need a subclass
# Name of the class has to be the same to match the namedtuple name
# pylint: disable=function-redefined
class BinDefinition(BinDefinition):
    def __new__(cls, thresholds, labels="%g",
                short_labels=None, width=None, width_label=""):

        def get_labels(fmt, default=None):
            if fmt is None:
                return default
            if isinstance(fmt, str):
                return [fmt % x for x in thresholds]
            elif isinstance(fmt, Callable):
                return [fmt(x) for x in thresholds]
            else:
                return fmt

        labels = get_labels(labels)
        short_labels = get_labels(short_labels, labels)
        if not width_label and width is not None:
            width_label = f"{width:g}"
        return super().__new__(
            cls, thresholds, labels, short_labels, width, width_label)

    @property
    def start(self) -> float:
        return self.thresholds[0]

    @property
    def nbins(self) -> int:
        return len(self.thresholds) - 1


def decimal_binnings(
        data, *, min_width=0, min_bins=2, max_bins=50,
        min_unique=5, add_unique=0,
        factors=(0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20),
        label_fmt="%g"):
    """
    Find a set of nice splits of data into bins

    The function first computes the scaling factor that is, the power of 10
    that brings the interval of values within [0, 1]. For instances, if the
    numbers come from interaval 10004001 and 10007005, the width of the
    interval is 3004, so the scaling factor is 1000.

    The function next considers bin widths that are products of scaling and
    different factors from 20 to 0.01 that make sense in decimal scale
    (see default value for argument `factors`). For each width, it rounds the
    minimal value down to this width and the maximal value up, and it computes
    the number of bins of that width that fit between these two values.
    If the number of bins is between `min_bins` and `max_bins`, and the width
    is at least `min_width`, this is a valid interval.

    If the data has no more than `min_unique` unique values, the function will
    add a set of bins that put each value into its own bin.

    If the data has no more than `add_unique` values, that last bins will put
    each value into its own bin.

    Args:
        data (np.ndarray):
            vector of data points; values may repeat, and nans and infs are
            filtered out.
        min_width (float): minimal bin width
        min_bins (int): minimal number of bins
        max_bins (int):
            maximal number of bins; the number of bins will never exceed the
            number of unique values
        min_unique (int):
            if the number of unique values are less or equal to `min_unique`,
            the function returns a single binning that matches that values in
            the data
        add_unique (int):
            similar to `min_unique` except that such bins are added to the list;
            set to 0 to disable
        factors (list of float):
            The factors with which the scaling is multiplied. Default is
            `(0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20)`,
            so if scaling is 1000, considered bin widths are 20000, 10000,
            5000, 2000, 1000, 500, 250, 200, 100, 50, 25, 20 and 10.
        label_fmt (str or Callable):
            A format string (default: "%g") used for threshold labels,
            or a function for formatting thresholds (e.g. var.str_val)

    Returns:
        bin_boundaries (list of np.ndarray): a list of bin boundaries,
            including the top boundary of the last interval, hence the list
            size equals the number bins + 1. These array match the `bin`
            argument of `numpy.histogram`.

            This is returned if `return_defs` is left `True`.

        bin_definition (list of BinDefinition):
            `BinDefinition` is a named tuple containing the beginning of the
            first bin (`start`), number of bins (`nbins`) and their widths
            (`width`). The last value can also be a `nd.array` with `nbins + 1`
            elements, which describes bins of unequal width and is used for
            binnings that match the unique values in the data (see `min_unique`
            and `add_unique`).

            This is returned if `return_defs` is `False`.
    """
    bins = []

    mn, mx, unique = _min_max_unique(data)
    if len(unique) <= max(min_unique, add_unique):
        bins.append(BinDefinition(_unique_thresholds(unique), label_fmt))
        if len(unique) <= min_unique:
            return bins

    diff = mx - mn
    f10 = 10 ** -np.floor(np.log10(diff))
    max_bins = min(max_bins, len(unique))
    for f in factors:
        width = f / f10
        if width < min_width:
            continue
        mn_ = np.floor(mn / width) * width
        mx_ = np.ceil(mx / width) * width
        nbins = np.round((mx_ - mn_) / width)
        if min_bins <= nbins <= max_bins \
                and (not bins or bins[-1].nbins != nbins):
            bins_ = mn_ + width * np.arange(nbins + 1)
            # to prevent values on the edge of the bin fall in the wrong bin
            # due to precision error on decimals that are not precise
            bins_ = np.around(bins_, decimals=np.finfo(bins_.dtype).precision)
            bin_def = BinDefinition(bins_, label_fmt, None, width)
            bins.append(bin_def)
    return bins


def time_binnings(data, *, min_bins=2, max_bins=50, min_unique=5, add_unique=0):
    """
    Find a set of nice splits of time variable data into bins

    The function considers bin widths of

    - 1, 5, 10, 15, 30 seconds.
    - 1, 5, 10, 15, 30 minutes,
    - 1, 2, 3, 6, 12 hours,
    - 1 day,
    - 1, 2 weeks,
    - 1, 2, 3, 6 months,
    - 1, 2, 5, 10, 25, 50, 100 years,

    and returns those that yield between `min_bins` and `max_bins` intervals.

    Args:
        data (np.ndarray):
            vector of data points; values may repeat, and nans and infs are
            filtered out.
        min_bins (int): minimal number of bins
        max_bins (int):
            maximal number of bins; the number of bins will never exceed the
            number of unique values

    Returns:
        bin_boundaries (list): a list of possible binning.
            Each element of `bin_boundaries` is a tuple consisting of a label
            describing the bin size (e.g. `2 weeks`) and a list of thresholds.
            Thresholds are given as pairs
            (number_of_seconds_since_epoch, label).
    """
    mn, mx, unique = _min_max_unique(data)
    mn = utc_from_timestamp(mn).timetuple()
    mx = utc_from_timestamp(mx).timetuple()
    bins = []
    if len(unique) <= max(min_unique, add_unique):
        bins.append(_unique_time_bins(unique))
    if len(unique) > min_unique:
        bins += _time_binnings(mn, mx, min_bins + 1, max_bins + 1)
    return bins


def _time_binnings(mn, mx, min_pts, max_pts):
    bins = []
    for place, step, fmt, unit in (
            [(5, x, "%H:%M:%S", "second") for x in (1, 5, 10, 15, 30)] +
            [(4, x, "%b %d %H:%M", "minute") for x in (1, 5, 10, 15, 30)] +
            [(3, x, "%y %b %d %H:%M", "hour") for x in (1, 2, 3, 6, 12)] +
            [(2, 1, "%y %b %d", "day")] +
            [(2, x, "%y %b %d", "week") for x in (7, 14)] +
            [(1, x, "%y %b", "month") for x in (1, 2, 3, 6)] +
            [(0, x, "%Y", "year") for x in (1, 2, 5, 10, 25, 50, 100)]):
        times = _time_range(mn, mx, place, step, min_pts, max_pts)
        if not times:
            continue
        times = [time.struct_time(t + (0, 0, 0)) for t in times]
        thresholds = [calendar.timegm(t) for t in times]
        labels = [time.strftime(fmt, t) for t in times]
        short_labels = _simplified_labels(labels)
        if place == 2 and step >= 7:
            unit_label = f"{step // 7} week{'s' * (step > 7)}"
        else:
            unit_label = f"{step} {unit}{'s' * (step > 1)}"
        new_bins = BinDefinition(
            thresholds, labels, short_labels, None, unit_label)
        if not bins or new_bins.nbins != bins[-1].nbins:
            bins.append(new_bins)
    return bins


# datetime + deltatime is not very useful here because deltatime is
# given a number of days, not years or months, so it doesn't allow
# for specifying a step of 1 month
def _time_range(start, end, place, step, min_pts, max_pts,
                _zeros=(0, 1, 1, 0, 0, 0)):
    if place == 2 and step % 7 == 0:
        startd = datetime.date(*start[:3])
        startd -= datetime.timedelta(days=-startd.weekday())
        start = [startd.year, startd.month, startd.day, 0, 0, 0]
    else:
        start = list(
            start[:place]
            + ((start[place] - _zeros[place]) // step * step + _zeros[place], )
            + _zeros[place + 1:])
    end = list(end[:place + 1] + _zeros[place + 1:])
    s = [tuple(start)]
    for _ in range(max_pts - 1):
        start[place] += step
        if place >= 3:  # hours, minutes, seconds
            for pos, maxval in enumerate((60, 60, 24), start=1):
                if start[-pos] >= maxval:
                    start[-pos - 1] += 1
                    start[-pos] %= maxval
        if place >= 2:
            md = _month_days(*start[:2])
            if start[2] > md:
                start[1] += 1
                start[2] %= md
        if start[1] > 12:
            start[0] += 1
            start[1] %= 12
        s.append(tuple(start))
        if start > end:
            return s if len(s) >= min_pts else None
    return None


def _month_days(year, month,
                _md=(None, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)):
    return _md[month] + (
        month == 2 and (year % 400 == 0 or year % 4 == 0 and year % 100 != 0))


def _simplified_labels(labels):
    labels = labels[:]
    to_remove = "42"
    while True:
        firsts = {f for f, *_ in (lab.split() for lab in labels)}
        if len(firsts) > 1:
            break
        to_remove = firsts.pop()
        flen = len(to_remove)
        if any(len(lab) == flen for lab in labels):
            break
        labels = [lab[flen+1:] for lab in labels]
    for i in range(len(labels) - 1, 0, -1):
        for k, c, d in zip(count(), labels[i].split(), labels[i - 1].split()):
            if c != d:
                labels[i] = " ".join(labels[i].split()[k:])
                break
    # If the last thing removed were month names and the labels continues with
    # hours, keep month name in the first label; "08 12:29" looks awkward.
    if not to_remove[0].isdigit() and ":" in labels[0]:
        labels[0] = f"{to_remove} {labels[0]}"
    return labels


def _unique_time_bins(unique):
    times = [utc_from_timestamp(x).timetuple() for x in unique]
    fmt = f'%y %b %d'
    fmt += " %H:%M" * (len({t[2:] for t in times}) > 1)
    fmt += ":%S" * bool(np.all(unique % 60 == 0))
    labels = [time.strftime(fmt, x) for x in times]
    short_labels = _simplified_labels(labels)
    return BinDefinition(_unique_thresholds(unique), labels, short_labels)


def _unique_thresholds(unique):
    if len(unique) >= 2:
        # make the last bin the same width as the one before
        last_boundary = 2 * unique[-1] - unique[-2]
    else:
        last_boundary = unique[0] + 1
    return np.hstack((unique, [last_boundary]))


def _min_max_unique(data):
    unique = np.unique(data)
    unique = unique[np.isfinite(unique)]
    if not unique.size:
        raise ValueError("no valid (non-nan) data")
    return unique[0], unique[-1], unique


short_time_units = dict(seconds="sec", minutes="min", hours="hrs",
                        weeks="wks", months="mon", years="yrs",
                        second="sec", minute="min", month="mon")


# noinspection PyPep8Naming
class EntropyMDL(Discretization):
    """
    Discretization into bins inferred by recursively splitting the values to
    minimize the class-entropy. The procedure stops when further splits would
    decrease the entropy for less than the corresponding increase of minimal
    description length (MDL). [FayyadIrani93].

    If there are no suitable cut-off points, the procedure returns a single bin,
    which means that the new feature is constant and can be removed.

    .. attribute:: force

        Induce at least one cut-off point, even when its information
        gain is lower than MDL (default: False).

    """
    def __init__(self, force=False):
        self.force = force

    def __call__(self, data, attribute):
        cont = contingency.get_contingency(data, attribute)
        values, I = cont.values, cont.counts.T
        cut_ind = np.array(self._entropy_discretize_sorted(I, self.force))
        if len(cut_ind) > 0:
            # "the midpoint between each successive pair of examples" (FI p.1)
            points = (values[cut_ind] + values[cut_ind - 1]) / 2.
        else:
            points = []
        return Discretizer.create_discretized_var(
            data.domain[attribute], points)

    @classmethod
    def _normalize(cls, X, axis=None, out=None):
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
        X = np.asarray(X, dtype=float)
        scale = np.sum(X, axis=axis, keepdims=True)
        if out is None:
            return X / scale
        else:
            if out is not X:
                assert out.shape == X.shape
                out[:] = X
            out /= scale
            return out

    @classmethod
    def _entropy_normalized(cls, D, axis=None):
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
        # req: (np.sum(D, axis=axis) >= 0).all()
        # req: (np.sum(D, axis=axis) <= 1).all()
        # req: np.all(np.abs(np.sum(D, axis=axis) - 1) < 1e-9)

        D = np.asarray(D)
        Dc = np.clip(D, np.finfo(D.dtype).eps, 1.0)
        return - np.sum(D * np.log2(Dc), axis=axis)

    @classmethod
    def _entropy(cls, D, axis=None):
        """
        Compute the entropy of distribution `D`.

        Parameters
        ----------
        D : array
            Distribution.
        axis : optional int
            Axis of `D` along which to compute the entropy.

        """
        D = cls._normalize(D, axis=axis)
        return cls._entropy_normalized(D, axis=axis)

    @classmethod
    def _entropy1(cls, D):
        """
        Compute the entropy of distributions in `D`
        (one per each row).
        """
        D = cls._normalize(D)
        return _discretize.entropy_normalized1(D)

    @classmethod
    def _entropy2(cls, D):
        """
        Compute the entropy of distributions in `D`
        (one per each row).
        """
        D = cls._normalize(D, axis=1)
        return _discretize.entropy_normalized2(D)

    @classmethod
    def _entropy_cuts_sorted(cls, CS):
        """
        Return the class information entropy induced by partitioning
        the `CS` distribution at all N-1 candidate cut points.

        Parameters
        ----------
        CS : (N, K) array of class distributions.
        """
        CS = np.asarray(CS)
        # |--|-------|--------|
        #  S1    ^       S2
        # S1 contains all points which are <= to cut point
        # Cumulative distributions for S1 and S2 (left right set)
        # i.e. a cut at index i separates the CS into S1Dist[i] and S2Dist[i]
        S1Dist = np.cumsum(CS, axis=0)[:-1]
        S2Dist = np.cumsum(CS[::-1], axis=0)[-2::-1]

        # Entropy of S1[i] and S2[i] sets
        ES1 = cls._entropy2(S1Dist)
        ES2 = cls._entropy2(S2Dist)

        # Number of cases in S1[i] and S2[i] sets
        S1_count = np.sum(S1Dist, axis=1)
        S2_count = np.sum(S2Dist, axis=1)

        # Number of all cases
        S_count = np.sum(CS)

        ES1w = ES1 * S1_count / S_count
        ES2w = ES2 * S2_count / S_count

        # E(A, T; S) Class information entropy of the partition S
        E = ES1w + ES2w

        return E, ES1, ES2

    @classmethod
    def _entropy_discretize_sorted(cls, C, force=False):
        """
        Entropy discretization on a sorted C.

        :param C: (N, K) array of class distributions.

        """
        E, ES1, ES2 = cls._entropy_cuts_sorted(C)
        # TODO: Also get the left right distribution counts from
        # entropy_cuts_sorted,

        # Note the + 1
        if len(E) == 0:
            return []
        cut_index = np.argmin(E) + 1

        # Distribution of classed in S1, S2 and S
        S1_c = np.sum(C[:cut_index], axis=0)
        S2_c = np.sum(C[cut_index:], axis=0)
        S_c = S1_c + S2_c

        ES = cls._entropy1(np.sum(C, axis=0))
        ES1, ES2 = ES1[cut_index - 1], ES2[cut_index - 1]

        # Information gain of the best split
        Gain = ES - E[cut_index - 1]
        # Number of different classes in S, S1 and S2
        k = float(np.sum(S_c > 0))
        k1 = float(np.sum(S1_c > 0))
        k2 = float(np.sum(S2_c > 0))

        assert k > 0
        delta = np.log2(3 ** k - 2) - (k * ES - k1 * ES1 - k2 * ES2)
        N = float(np.sum(S_c))

        if N > 1 and Gain > np.log2(N - 1) / N + delta / N:
            # Accept the cut point and recursively split the subsets.
            left, right = [], []
            if k1 > 1 and cut_index > 1:
                left = cls._entropy_discretize_sorted(C[:cut_index, :])
            if k2 > 1 and cut_index < len(C) - 1:
                right = cls._entropy_discretize_sorted(C[cut_index:, :])
            return left + [cut_index] + [i + cut_index for i in right]
        elif force:
            return [cut_index]
        else:
            return []


class DomainDiscretizer(Reprable):
    """Discretizes all continuous features in the data.

    .. attribute:: method

        Feature discretization method (instance of
        :obj:`Orange.preprocess.Discretization`). If `None` (default),
        :class:`Orange.preprocess.EqualFreq` with 4 intervals is
        used.

    .. attribute:: clean

        If `True`, features discretized into a single interval constant are
        removed. This is useful for discretization methods that infer the
        number of intervals from the data, such as
        :class:`Orange.preprocess.EntropyMDL` (default: `True`).

    .. attribute:: discretize_class

        Determines whether a target is also discretized if it is continuous.
        (default: `False`)
    """
    def __init__(self, discretize_class=False, method=None, clean=True,
                 fixed=None):
        self.discretize_class = discretize_class
        self.method = method
        self.clean = clean
        self.fixed = fixed

    def __call__(self, data, fixed=None):
        """
        Compute and return discretized domain.

        :param data: Data to discretize.
        """

        def transform_list(s, fixed=None):
            new_vars = []
            for var in s:
                if var.is_continuous:
                    if fixed and var.name in fixed.keys():
                        nv = method(data, var, fixed)
                    else:
                        nv = method(data, var)
                    if not self.clean or len(nv.values) > 1:
                        new_vars.append(nv)
                else:
                    new_vars.append(var)
            return new_vars
        if self.method is None:
            method = EqualFreq(n=4)
        else:
            method = self.method
        domain = data.domain
        new_attrs = transform_list(domain.attributes, fixed or self.fixed)
        if self.discretize_class:
            new_classes = transform_list(domain.class_vars)
        else:
            new_classes = domain.class_vars
        return Domain(new_attrs, new_classes)
