import numpy as np
from scipy import stats
import sklearn.metrics as skl_metrics
from sklearn.utils.extmath import row_norms, safe_sparse_dot

from Orange.distance import _distance
from Orange.statistics import util

from .base import (Distance, DistanceModel, FittedDistance, FittedDistanceModel,
                   SklDistance, _orange_to_numpy)


class EuclideanRowsModel(FittedDistanceModel):
    """
    Model for computation of Euclidean distances between rows.

    Means are used as offsets for normalization, and two deviations are
    used for scaling.
    """
    def __init__(self, attributes, impute, normalize,
                 continuous, discrete,
                 means, vars, dist_missing2_cont,
                 dist_missing_disc, dist_missing2_disc):
        super().__init__(attributes, 1, impute)
        self.normalize = normalize
        self.continuous = continuous
        self.discrete = discrete
        self.means = means
        self.vars = vars
        self.dist_missing2_cont = dist_missing2_cont
        self.dist_missing_disc = dist_missing_disc
        self.dist_missing2_disc = dist_missing2_disc

    def compute_distances(self, x1, x2=None):
        """
        The method
        - extracts normalized continuous attributes and then uses `row_norms`
          and `safe_sparse_do`t to compute the distance as x^2 - 2xy - y^2
          (the trick from sklearn);
        - calls a function in Cython that recomputes the distances between pairs
          of rows that yielded nan
        - calls a function in Cython that adds the contributions of discrete
          columns
        """
        if self.continuous.any():
            data1, data2 = self.continuous_columns(
                x1, x2, self.means, np.sqrt(2 * self.vars))

            # adapted from sklearn.metric.euclidean_distances
            xx = row_norms(data1, squared=True)[:, np.newaxis]
            if x2 is not None:
                yy = row_norms(data2, squared=True)[np.newaxis, :]
            else:
                yy = xx.T
            distances = safe_sparse_dot(data1, data2.T, dense_output=True)
            distances *= -2
            distances += xx
            distances += yy
            np.maximum(distances, 0, out=distances)
            if x2 is None:
                distances.flat[::distances.shape[0] + 1] = 0.0
            fixer = _distance.fix_euclidean_rows_normalized if self.normalize \
                else _distance.fix_euclidean_rows
            fixer(distances, data1, data2,
                  self.means, self.vars, self.dist_missing2_cont,
                  x2 is not None)
        else:
            distances = np.zeros((x1.shape[0],
                                  (x2 if x2 is not None else x1).shape[0]))

        if np.any(self.discrete):
            data1, data2 = self.discrete_columns(x1, x2)
            _distance.euclidean_rows_discrete(
                distances, data1, data2, self.dist_missing_disc,
                self.dist_missing2_disc, x2 is not None)

        if x2 is None:
            _distance.lower_to_symmetric(distances)
        return np.sqrt(distances)


class EuclideanColumnsModel(FittedDistanceModel):
    """
    Model for computation of Euclidean distances between columns.

    Means are used as offsets for normalization, and two deviations are
    used for scaling.
    """
    def __init__(self, attributes, impute, normalize, means, vars):
        super().__init__(attributes, 0, impute)
        self.normalize = normalize
        self.means = means
        self.vars = vars

    def compute_distances(self, x1, x2=None):
        """
        The method
        - extracts normalized continuous attributes and then uses `row_norms`
          and `safe_sparse_do`t to compute the distance as x^2 - 2xy - y^2
          (the trick from sklearn);
        - calls a function in Cython that adds the contributions of discrete
          columns
        """
        if self.normalize:
            x1 = x1 - self.means
            x1 /= np.sqrt(2 * self.vars)

        # adapted from sklearn.metric.euclidean_distances
        xx = row_norms(x1.T, squared=True)[:, np.newaxis]
        distances = safe_sparse_dot(x1.T, x1, dense_output=True)
        distances *= -2
        distances += xx
        distances += xx.T
        np.maximum(distances, 0, out=distances)
        distances.flat[::distances.shape[0] + 1] = 0.0

        fixer = _distance.fix_euclidean_cols_normalized if self.normalize \
            else _distance.fix_euclidean_cols
        fixer(distances, x1, self.means, self.vars)
        return np.sqrt(distances)


class Euclidean(FittedDistance):
    supports_sparse = True  # via fallback
    supports_discrete = True
    supports_normalization = True
    fallback = SklDistance('euclidean')
    rows_model_type = EuclideanRowsModel

    def __new__(cls, e1=None, e2=None, axis=1, impute=False, normalize=False):
        return super().__new__(cls, e1, e2, axis, impute, normalize=normalize)

    def get_continuous_stats(self, column):
        """
        Return mean, variance and distance betwwen pairs of missing values
        for the given columns. The method is called by inherited `fit_rows`
        to construct a row-distance model
        """
        mean = util.nanmean(column)
        var = util.nanvar(column)
        if self.normalize:
            if var == 0:
                return None
            dist_missing2_cont = 1
        else:
            dist_missing2_cont = 2 * var
            if np.isnan(dist_missing2_cont):
                dist_missing2_cont = 0
        return mean, var, dist_missing2_cont

    def fit_cols(self, attributes, x, n_vals):
        """
        Return `EuclideanColumnsModel` with stored means and variances
        for normalization and imputation.
        """
        self.check_no_discrete(n_vals)
        means = np.nanmean(x, axis=0)
        vars = np.nanvar(x, axis=0)
        if self.normalize and (np.isnan(vars).any() or not vars.all()):
            raise ValueError("some columns are constant or have no values")
        return EuclideanColumnsModel(
            attributes, self.impute, self.normalize, means, vars)


class ManhattanRowsModel(FittedDistanceModel):
    """
    Model for computation of Euclidean distances between rows.

    Means are used as offsets for normalization, and two deviations are
    used for scaling.
    """
    def __init__(self, attributes, impute, normalize,
                 continuous, discrete,
                 medians, mads, dist_missing2_cont,
                 dist_missing_disc, dist_missing2_disc):
        super().__init__(attributes, 1, impute)
        self.normalize = normalize
        self.continuous = continuous
        self.discrete = discrete
        self.medians = medians
        self.mads = mads
        self.dist_missing2_cont = dist_missing2_cont
        self.dist_missing_disc = dist_missing_disc
        self.dist_missing2_disc = dist_missing2_disc

    def compute_distances(self, x1, x2):
        """
        The method
        - extracts normalized continuous attributes and computes distances
          ignoring the possibility of nans
        - recomputes the distances between pairs of rows that yielded nans
        - adds the contributions of discrete columns using the same function as
          the Euclidean distance
        """
        if self.continuous.any():
            data1, data2 = self.continuous_columns(
                x1, x2, self.medians, 2 * self.mads)
            distances = _distance.manhattan_rows_cont(
                data1, data2, x2 is not None)
            if self.normalize:
                _distance.fix_manhattan_rows_normalized(
                    distances, data1, data2, x2 is not None)
            else:
                _distance.fix_manhattan_rows(
                    distances, data1, data2,
                    self.medians, self.mads, self.dist_missing2_cont,
                    x2 is not None)
        else:
            distances = np.zeros((x1.shape[0],
                                  (x2 if x2 is not None else x1).shape[0]))

        if np.any(self.discrete):
            data1, data2 = self.discrete_columns(x1, x2)
            # For discrete attributes, Euclidean is same as Manhattan
            _distance.euclidean_rows_discrete(
                distances, data1, data2, self.dist_missing_disc,
                self.dist_missing2_disc, x2 is not None)

        if x2 is None:
            _distance.lower_to_symmetric(distances)
        return distances


class ManhattanColumnsModel(FittedDistanceModel):
    """
    Model for computation of Manhattan distances between columns.

    Medians are used as offsets for normalization, and two MADS are
    used for scaling.
    """

    def __init__(self, attributes, impute, normalize, medians, mads):
        super().__init__(attributes, 0, impute)
        self.normalize = normalize
        self.medians = medians
        self.mads = mads

    def compute_distances(self, x1, x2=None):
        if self.normalize:
            x1 = x1 - self.medians
            x1 /= 2
            x1 /= self.mads
        return _distance.manhattan_cols(x1, self.medians, self.mads, self.normalize)


class Manhattan(FittedDistance):
    supports_sparse = True  # via fallback
    supports_discrete = True
    supports_normalization = True
    fallback = SklDistance('manhattan')
    rows_model_type = ManhattanRowsModel

    def __new__(cls, e1=None, e2=None, axis=1, impute=False, normalize=False):
        return super().__new__(cls, e1, e2, axis, impute, normalize=normalize)

    def get_continuous_stats(self, column):
        """
        Return median, MAD and distance betwwen pairs of missing values
        for the given columns. The method is called by inherited `fit_rows`
        to construct a row-distance model
        """
        median = np.nanmedian(column)
        mad = np.nanmedian(np.abs(column - median))
        if self.normalize:
            if mad == 0:
                return None
            dist_missing2_cont = 1
        else:
            dist_missing2_cont = 2 * mad
            if np.isnan(dist_missing2_cont):
                dist_missing2_cont = 0
        return median, mad, dist_missing2_cont

    def fit_cols(self, attributes, x, n_vals):
        """
        Return `ManhattanColumnsModel` with stored medians and MADs
        for normalization and imputation.
        """
        self.check_no_discrete(n_vals)
        medians = np.nanmedian(x, axis=0)
        mads = np.nanmedian(np.abs(x - medians), axis=0)
        if self.normalize and (np.isnan(mads).any() or not mads.all()):
            raise ValueError(
                "some columns have zero absolute distance from median, "
                "or no values")
        return ManhattanColumnsModel(
            attributes, self.impute, self.normalize, medians, mads)


class Cosine(FittedDistance):
    supports_sparse = True  # via fallback
    supports_discrete = True
    fallback = SklDistance('cosine')

    @staticmethod
    def discrete_to_indicators(x, discrete):
        """Change non-zero values of discrete attributes to 1."""
        if discrete.any():
            x = x.copy()
            for col, disc in enumerate(discrete):
                if disc:
                    x[:, col].clip(0, 1, out=x[:, col])
        return x

    def fit_rows(self, attributes, x, n_vals):
        """Return a model for cosine distances with stored means for imputation
        """
        discrete = n_vals > 0
        x = self.discrete_to_indicators(x, discrete)
        means = util.nanmean(x, axis=0)
        means = np.nan_to_num(means)
        return self.CosineModel(attributes, self.axis, self.impute,
                                discrete, means)

    fit_cols = fit_rows

    class CosineModel(FittedDistanceModel):
        """Model for computation of cosine distances across rows and columns.
        All non-zero discrete values are treated as 1."""
        def __init__(self, attributes, axis, impute, discrete, means):
            super().__init__(attributes, axis, impute)
            self.discrete = discrete
            self.means = means

        def compute_distances(self, x1, x2):
            """
            The method imputes the missing values as means and calls
            safe_sparse_dot. Imputation simplifies computation at a cost of
            (theoretically) slightly wrong distance between pairs of missing
             values.
            """

            def prepare_data(x):
                if self.discrete.any():
                    data = Cosine.discrete_to_indicators(x, self.discrete)
                else:
                    data = x.copy()
                for col, mean in enumerate(self.means):
                    column = data[:, col]
                    column[np.isnan(column)] = mean
                if self.axis == 0:
                    data = data.T
                data /= row_norms(data)[:, np.newaxis]
                return data

            data1 = prepare_data(x1)
            data2 = data1 if x2 is None else prepare_data(x2)
            dist = safe_sparse_dot(data1, data2.T)
            np.clip(dist, 0, 1, out=dist)
            if x2 is None:
                dist.flat[::dist.shape[0] + 1] = 1.0
            return 1 - dist


class JaccardModel(FittedDistanceModel):
    """
    Model for computation of cosine distances across rows and columns.
    All non-zero values are treated as 1.
    """
    def __init__(self, attributes, axis, impute, ps):
        super().__init__(attributes, axis, impute)
        self.ps = ps

    def compute_distances(self, x1, x2):
        """
        The method uses a function implemented in Cython. Data (`x1` and `x2`)
        is accompanied by two tables. One is a 2-d table in which elements of
        `x1` (`x2`) are replaced by 0's and 1's. The other is a vector
        indicating rows (or column) with nan values.

        The function in Cython uses a fast loop without any conditions to
        compute distances between rows without missing values, and a slower
        loop for those with missing values.
        """
        nonzeros1 = np.not_equal(x1, 0).view(np.int8)
        if self.axis == 1:
            nans1 = _distance.any_nan_row(x1)
            if x2 is None:
                nonzeros2, nans2 = nonzeros1, nans1
            else:
                nonzeros2 = np.not_equal(x2, 0).view(np.int8)
                nans2 = _distance.any_nan_row(x2)
            return _distance.jaccard_rows(
                nonzeros1, nonzeros2,
                x1, x1 if x2 is None else x2,
                nans1, nans2,
                self.ps,
                x2 is not None)
        else:
            nans1 = _distance.any_nan_row(x1.T)
            return _distance.jaccard_cols(
                nonzeros1, x1, nans1, self.ps)


class Jaccard(FittedDistance):
    supports_sparse = False
    supports_discrete = True
    fallback = SklDistance('jaccard')
    ModelType = JaccardModel

    def fit_rows(self, attributes, x, n_vals):
        """
        Return a model for computation of Jaccard values. The model stores
        frequencies of non-zero values per each column.
        """
        ps = np.fromiter(
            (_distance.p_nonzero(x[:, col]) for col in range(len(n_vals))),
            dtype=np.double, count=len(n_vals))
        return JaccardModel(attributes, self.axis, self.impute, ps)

    fit_cols = fit_rows


class CorrelationDistanceModel(DistanceModel):
    """Helper class for normal and absolute Pearson and Spearman correlation"""
    def __init__(self, absolute, axis=1, impute=False):
        super().__init__(axis, impute)
        self.absolute = absolute

    def compute_distances(self, x1, x2):
        rho = self.compute_correlation(x1, x2)
        if self.absolute:
            return (1. - np.abs(rho)) / 2.
        else:
            return (1. - rho) / 2.

    def compute_correlation(self, x1, x2):
        raise NotImplementedError()


class SpearmanModel(CorrelationDistanceModel):
    def compute_correlation(self, x1, x2):
        if x2 is None:
            n1 = x1.shape[1 - self.axis]
            if n1 == 2:
                # Special case to properly fill degenerate self correlations
                # (nan, inf on the diagonals)
                rho = stats.spearmanr(x1, x1, axis=self.axis)[0]
                assert rho.shape == (4, 4)
                rho = rho[:2, :2].copy()
            else:
                # scalar if n1 == 1
                rho = stats.spearmanr(x1, axis=self.axis)[0]
            return np.atleast_2d(rho)
        else:
            return _spearmanr2(x1, x2, axis=self.axis)


def _spearmanr2(a, b, axis=0):
    """
    Compute all pairwise spearman rank moment correlations between rows
    or columns of a and b

    Parameters
    ----------
    a : (N, M) numpy.ndarray
        The input cases a.
    b : (J, K) numpy.ndarray
        The input cases b. In case of axis == 0: J must equal N;
        otherwise if axis == 1 then K must equal M.
    axis : int
        If 0 the correlation are computed between a and b's columns.
        Otherwise if 1 the correlations are computed between rows.

    Returns
    -------
    cor : (N, J) or (M, K) nd.array
        If axis == 0 then (N, J) matrix of correlations between a x b columns
        else a (N, J) matrix of correlations between a x b rows.

    See Also
    --------
    scipy.stats.spearmanr
    """
    a, b = np.atleast_2d(a, b)
    assert a.shape[axis] == b.shape[axis]
    ar = np.apply_along_axis(stats.rankdata, axis, a)
    br = np.apply_along_axis(stats.rankdata, axis, b)

    return _corrcoef2(ar, br, axis=axis)


def _corrcoef2(a, b, axis=0):
    """
    Compute all pairwise Pearson product-moment correlation coefficients
    between rows or columns of a and b

    Parameters
    ----------
    a : (N, M) numpy.ndarray
        The input cases a.
    b : (J, K) numpy.ndarray
        The input cases b. In case of axis == 0: J must equal N;
        otherwise if axis == 1 then K must equal M.
    axis : int
        If 0 the correlation are computed between a and b's columns.
        Otherwise if 1 the correlations are computed between rows.

    Returns
    -------
    cor : (N, J) or (M, K) nd.array
        If axis == 0 then (N, J) matrix of correlations between a x b columns
        else a (N, J) matrix of correlations between a x b rows.

    See Also
    --------
    numpy.corrcoef
    """
    a, b = np.atleast_2d(a, b)
    if not (axis == 0 or axis == 1):
        raise ValueError("Invalid axis {} (only 0 or 1 accepted)".format(axis))

    mean_a = np.mean(a, axis=axis, keepdims=True)
    mean_b = np.mean(b, axis=axis, keepdims=True)
    assert a.shape[axis] == b.shape[axis]

    n = a.shape[1 - axis]
    m = b.shape[1 - axis]

    a = a - mean_a
    b = b - mean_b

    if axis == 0:
        C = a.T.dot(b)
        assert C.shape == (n, m)
    elif axis == 1:
        C = a.dot(b.T)
        assert C.shape == (n, m)

    ss_a = np.sum(a ** 2, axis=axis, keepdims=True)
    ss_b = np.sum(b ** 2, axis=axis, keepdims=True)

    if axis == 0:
        ss_a = ss_a.T
    else:
        ss_b = ss_b.T

    assert ss_a.shape == (n, 1)
    assert ss_b.shape == (1, m)
    C /= np.sqrt(ss_a)
    C /= np.sqrt(ss_b)
    return C


class CorrelationDistance(Distance):
    supports_missing = False


class SpearmanR(CorrelationDistance):
    def fit(self, _):
        return SpearmanModel(False, self.axis, self.impute)


class SpearmanRAbsolute(CorrelationDistance):
    def fit(self, _):
        return SpearmanModel(True, self.axis, self.impute)


class PearsonModel(CorrelationDistanceModel):
    def compute_correlation(self, x1, x2):
        if x2 is None:
            c = np.corrcoef(x1, rowvar=self.axis == 1)
            return np.atleast_2d(c)
        else:
            return _corrcoef2(x1, x2, axis=self.axis)


class PearsonR(CorrelationDistance):
    def fit(self, _):
        return PearsonModel(False, self.axis, self.impute)


class PearsonRAbsolute(CorrelationDistance):
    def fit(self, _):
        return PearsonModel(True, self.axis, self.impute)


class Mahalanobis(Distance):
    supports_sparse = False
    supports_missing = False

    def fit(self, data):
        """Return a model with stored inverse covariance matrix"""
        x = _orange_to_numpy(data)
        if self.axis == 0:
            x = x.T
        try:
            c = np.cov(x.T)
        except:
            raise MemoryError("Covariance matrix is too large.")
        try:
            vi = np.linalg.inv(c)
        except:
            raise ValueError("Computation of inverse covariance matrix failed.")
        return MahalanobisModel(self.axis, self.impute, vi)


class MahalanobisModel(DistanceModel):
    def __init__(self, axis, impute, vi):
        super().__init__(axis, impute)
        self.vi = vi

    def __call__(self, e1, e2=None, impute=None):
        # argument `impute` is here just for backward compatibility; don't use
        if impute is not None:
            self.impute = impute
        return super().__call__(e1, e2)

    def compute_distances(self, x1, x2):
        if self.axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        if x1.shape[1] != self.vi.shape[0] or \
                x2 is not None and x2.shape[1] != self.vi.shape[0]:
            raise ValueError('Incorrect number of features.')
        return skl_metrics.pairwise.pairwise_distances(
            x1, x2, metric='mahalanobis', VI=self.vi)


# TODO: Appears to have been used only in the Distances widget (where it had
# to be handled as a special case and is now replaced with the above class)
# and in tests. Remove?
class MahalanobisDistance:
    """
    Obsolete class needed for backward compatibility.

    Previous implementation of instances did not have a separate fitting phase,
    except for MahalanobisDistance, which was implemented in a single class
    but required first (explicitly) calling the method 'fit'. The backward
    compatibility hack in :obj:`Distance` cannot handle such use, hence it
    is provided in this class.
    """
    def __new__(cls, data=None, axis=1, _='Mahalanobis'):
        if data is None:
            return cls
        return Mahalanobis(axis=axis).fit(data)
