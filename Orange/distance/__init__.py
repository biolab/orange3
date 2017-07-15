import numpy as np
from scipy import stats
import sklearn.metrics as skl_metrics
from sklearn.utils.extmath import row_norms, safe_sparse_dot

from Orange.data import Table, Domain, Instance, RowInstance
from Orange.misc import DistMatrix
from Orange.distance import _distance
from Orange.statistics import util
from Orange.preprocess import SklImpute

__all__ = ['Euclidean', 'Manhattan', 'Cosine', 'Jaccard', 'SpearmanR',
           'SpearmanRAbsolute', 'PearsonR', 'PearsonRAbsolute', 'Mahalanobis',
           'MahalanobisDistance']


# TODO this *private* function is called from several widgets to prepare
# data for calling the below classes. After we (mostly) stopped relying
# on sklearn.metrics, this is (mostly) unnecessary
def _preprocess(table, impute=True):
    """Remove categorical attributes and impute missing values."""
    if not len(table):
        return table
    new_domain = Domain(
        [a for a in table.domain.attributes if a.is_continuous],
        table.domain.class_vars,
        table.domain.metas)
    new_data = table.transform(new_domain)
    if impute:
        new_data = SklImpute()(new_data)
    return new_data


# TODO I have put this function here as a substitute the above `_preprocess`.
# None of them really belongs here; (re?)move them, eventually.
def remove_discrete_features(data):
    """Remove discrete columns from the data."""
    new_domain = Domain(
        [a for a in data.domain.attributes if a.is_continuous],
        data.domain.class_vars,
        data.domain.metas)
    return data.transform(new_domain)


def impute(data):
    """Impute missing values."""
    return SklImpute()(data)


def _orange_to_numpy(x):
    """
    Return :class:`numpy.ndarray` (dense or sparse) with attribute data
    from the given instance of :class:`Orange.data.Table`,
    :class:`Orange.data.RowInstance` or :class:`Orange.data.Instance`.    .
    """
    if isinstance(x, Table):
        return x.X
    elif isinstance(x, Instance):
        return np.atleast_2d(x.x)
    elif isinstance(x, np.ndarray):
        return np.atleast_2d(x)
    else:
        return x    # e.g. None


class Distance:
    """
    Base class for construction of distances models (:obj:`DistanceModel`).
    
    Distances can be computed between all pairs of rows in one table, or
    between pairs where one row is from one table and one from another.
    
    If `axis` is set to `0`, the class computes distances between all pairs
    of columns in a table. Distances between columns from separate tables are
    probably meaningless, thus unsupported.
    
    The class can be used as follows:
    
    - Constructor is called only with keyword argument `axis` that
      specifies the axis over which the distances are computed, and with other
      subclass-specific keyword arguments.
    - Next, we call the method `fit(data)` to produce an instance of
      :obj:`DistanceModel`; the instance stores any parameters needed for
      computation of distances, such as statistics for normalization and
      handling of missing data.
    - We can then call the :obj:`DistanceModel` with data to compute the
      distance between its rows or columns, or with two data tables to
      compute distances between all pairs of rows.
    
    The second, shorter way to use this class is to call the constructor with
    one or two data tables and any additional keyword arguments. Constructor
    will execute the above steps and return :obj:`~Orange.misc.DistMatrix`.
    Such usage is here for backward compatibility, practicality and efficiency.
    
    Args:
        e1 (:obj:`~Orange.data.Table` or :obj:`~Orange.data.Instance` or :obj:`np.ndarray` or `None`):
            data on which to train the model and compute the distances
        e2 (:obj:`~Orange.data.Table` or :obj:`~Orange.data.Instance` or :obj:`np.ndarray` or `None`):
            if present, the class computes distances with pairs coming from
            the two tables
        axis (int):
            axis over which the distances are computed, 1 (default) for
            rows, 0 for columns
    
    Attributes:
        axis (int):
            axis over which the distances are computed, 1 (default) for
            rows, 0 for columns
        impute (bool):
            if `True` (default is `False`), nans in the computed distances
            are replaced with zeros, and infs with very large numbers.            

    The capabilities of the metrics are described with class attributes.

    If class attribute `supports_discrete` is `True`, the distance
    also uses discrete attributes to compute row distances. The use of discrete
    attributes depends upon the type of distance; e.g. Jaccard distance observes
    whether the value is zero or non-zero, while Euclidean and Manhattan
    distance observes whether a pair of values is same or different.
        
    Class attribute `supports_missing` indicates that the distance can cope
    with missing data. In such cases, letting the distance handle it should
    be preferred over pre-imputation of missing values.
    
    Class attribute `supports_normalization` indicates that the constructor
    accepts an argument `normalize`. If set to `True`, the metric will attempt
    to normalize the values in a sense that each attribute will have equal
    influence. For instance, the Euclidean distance subtract the mean and
    divides the result by the deviation, while Manhattan distance uses the
    median and MAD.
    
    If class attribute `supports_sparse` is `True`, the class will handle
    sparse data. Currently, all classes that do handle it rely on fallbacks to
    SKL metrics. These, however, do not support discrete data and missing
    values, and will fail silently.
    """
    supports_sparse = False
    supports_discrete = False
    supports_normalization = False
    supports_missing = True

    def __new__(cls, e1=None, e2=None, axis=1, impute=False, **kwargs):
        self = super().__new__(cls)
        self.axis = axis
        self.impute = impute
        # Ugly, but needed to allow allow setting subclass-specific parameters
        # (such as normalize) when `e1` is not `None` and the `__new__` in the
        # subclass is skipped
        self.__dict__.update(**kwargs)
        if e1 is None:
            return self

        # Fallbacks for sparse data and numpy tables. Remove when subclasses
        # no longer use fallbacks for sparse data, and handling numpy tables
        # becomes obsolete (or handled elsewhere)
        if (not hasattr(e1, "domain")
                or hasattr(e1, "is_sparse") and e1.is_sparse()):
            fallback = getattr(self, "fallback", None)
            if fallback is not None:
                return fallback(e1, e2, axis, impute)

        # Magic constructor
        model = self.fit(e1)
        return model(e1, e2)

    def fit(self, e1):
        """
        Return a :obj:`DistanceModel` fit to the data. Must be implemented in
        subclasses.
        
        Args:
            e1 (:obj:`~Orange.data.Table` or :obj:`~Orange.data.Instance` or
                :obj:`np.ndarray` or `None`:
                data on which to train the model and compute the distances

        Returns: `DistanceModel`
        """
        pass


class DistanceModel:
    """
    Base class for classes that compute distances between data rows or columns.
    Instances of these classes are not constructed directly but returned by
    the corresponding instances of :obj:`Distance`.
    
    Attributes:
        axis (int, readonly):
            axis over which the distances are computed, 1 (default) for
            rows, 0 for columns
        impute (bool):
            if `True` (default is `False`), nans in the computed distances
            are replaced with zeros, and infs with very large numbers 
            
    """
    def __init__(self, axis, impute=False):
        self._axis = axis
        self.impute = impute

    @property
    def axis(self):
        return self._axis

    def __call__(self, e1, e2=None):
        """
        If e2 is omitted, calculate distances between all rows (axis=1) or
        columns (axis=2) of e1. If e2 is present, calculate distances between
        all pairs if rows from e1 and e2.
        
        This method converts the data into numpy arrays, calls the method
        `compute_data` and packs the result into `DistMatrix`. Subclasses are
        expected to define the `compute_data` and not the `__call__` method.

        Args:
            e1 (Orange.data.Table or Orange.data.Instance or numpy.ndarray):
                input data
            e2 (Orange.data.Table or Orange.data.Instance or numpy.ndarray):
                secondary data
                
        Returns:
            A distance matrix (Orange.misc.distmatrix.DistMatrix)
        """
        if self.axis == 0 and e2 is not None:
            # Backward compatibility fix
            if e2 is e1:
                e2 = None
            else:
                raise ValueError("Two tables cannot be compared by columns")

        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        dist = self.compute_distances(x1, x2)
        if self.impute and np.isnan(dist).any():
            dist = np.nan_to_num(dist)
        if isinstance(e1, Table) or isinstance(e1, RowInstance):
            dist = DistMatrix(dist, e1, e2, self.axis)
        else:
            dist = DistMatrix(dist)
        return dist

    def compute_distances(self, x1, x2):
        """
        Compute the distance between rows or colums of `x1`, or between rows
        of `x1` and `x2`. This method must be implement by subclasses. Do not
        call directly."""
        pass


class FittedDistanceModel(DistanceModel):
    """
    Convenient common parent class for distance models with separate methods
    for fitting and for computation of distances across rows and columns.
    
    Results of fitting are packed into a dictionary for easier passing to
    Cython function that do the heavy lifting in these classes.
    
    Attributes:
        attributes (list of `Variable`): attributes on which the model was fit
        fit_params (dict): data used by the model
        
    Class attributes:
        distance_by_cols: a function that accepts a numpy array and parameters
            and returns distances by columns. Usually a Cython function.
        distance_by_rows: a function that accepts one or two numpy arrays,
            an indicator whether the distances are to be computed within
            a single array or between two arrays, and parameters; and
            returns distances by columns. Usually a Cython function.
    """
    def __init__(self, attributes, axis=1, impute=False, fit_params=None):
        super().__init__(axis, impute)
        self.attributes = attributes
        self.fit_params = fit_params

    def __call__(self, e1, e2=None):
        """
        Check the validity of the domains before passing the data to the 
        inherited method.
        """
        if e1.domain.attributes != self.attributes or \
                e2 is not None and e2.domain.attributes != self.attributes:
            raise ValueError("mismatching domains")
        return super().__call__(e1, e2)

    def compute_distances(self, x1, x2=None):
        """
        Compute distances by calling either `distance_by_cols` or
        `distance_by_wors`
        """
        if self.axis == 0:
            return self.distance_by_cols(x1, self.fit_params)
        else:
            return self.distance_by_rows(
                x1,
                x2 if x2 is not None else x1,
                x2 is not None,
                self.fit_params)


class FittedDistance(Distance):
    """
    Convenient common parent class for distancess with separate methods for
    fitting and for computation of distances across rows and columns.
    Results of fitting are packed into a dictionary for easier passing to
    Cython function that do the heavy lifting in these classes.
    
    The class implements a method `fit` that calls either `fit_columns`
    or `fit_rows` with the data and the number of values for discrete
    attributes.

    Class attribute `ModelType` contains the type of the model returned by
    `fit`.
    """
    ModelType = None  #: Option[FittedDistanceModel]

    def fit(self, data):
        attributes = data.domain.attributes
        x = _orange_to_numpy(data)
        n_vals = np.fromiter(
            (len(attr.values) if attr.is_discrete else 0
             for attr in attributes),
            dtype=np.int32, count=len(attributes))
        fit_params = [self.fit_cols, self.fit_rows][self.axis](x, n_vals)
        # pylint: disable=not-callable
        return self.ModelType(attributes, axis=self.axis, fit_params=fit_params)

    def fit_cols(self, x, n_vals):
        if any(n_vals):
            raise ValueError("columns with discrete values are incommensurable")

    def fit_rows(self, x, n_vals):
        pass


# Fallbacks for distances in sparse data
# To be removed as the corresponding functionality is implemented above

class SklDistance:
    """
    Wrapper for functions sklearn's metrics. Used only as temporary fallbacks
    when `Euclidean`, `Manhattan`, `Cosine` or `Jaccard` are given sparse data
    or raw numpy arrays. These classes can't handle discrete or missing data
    and normalization. Do not use for wrapping new classes.
    """
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, e1, e2=None, axis=1, impute=False):
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        dist = skl_metrics.pairwise.pairwise_distances(
            x1, x2, metric=self.metric)
        if impute and np.isnan(dist).any():
            dist = np.nan_to_num(dist)
        if isinstance(e1, Table) or isinstance(e1, RowInstance):
            dist_matrix = DistMatrix(dist, e1, e2, axis)
        else:
            dist_matrix = DistMatrix(dist)
        return dist_matrix


class EuclideanModel(FittedDistanceModel):
    def distance_by_rows(self, x1, x2, two_tables, fit_params):
        vars = fit_params["vars"]
        means = fit_params["means"]
        dist_missing = fit_params["dist_missing"]
        dist_missing2 = fit_params["dist_missing2"]
        normalize = fit_params["normalize"]

        cols = vars >= 0
        if np.any(cols):
            if normalize:
                data1 = x1[:, cols]
                data1 -= means[cols]
                data1 /= np.sqrt(2 * vars[cols])
                if two_tables:
                    data2 = x2[:, cols]
                    data2 -= means
                    data2 /= np.sqrt(2 * vars[cols])
                else:
                    data2 = data1
            elif np.all(cols):
                data1, data2 = x1, x2
            else:
                data1 = x1[:, cols]
                data2 = x2[:, cols] if two_tables else data1

            # adapted from sklearn.metric.euclidean_distances
            xx = row_norms(data1, squared=True)[:, np.newaxis]
            if two_tables:
                yy = row_norms(data2, squared=True)[np.newaxis, :]
            else:
                yy = xx.T
            distances = safe_sparse_dot(data1, data2.T, dense_output=True)
            distances *= -2
            distances += xx
            distances += yy
            np.maximum(distances, 0, out=distances)
            if not two_tables:
                distances.flat[::distances.shape[0] + 1] = 0.0

            if normalize:
                _distance.fix_euclidean_rows_normalized(distances, data1, data2, means[cols], vars[cols], dist_missing2[cols], two_tables)
            else:
                _distance.fix_euclidean_rows(distances, data1, data2, means[cols], vars[cols], dist_missing2[cols], two_tables)

        else:
            distances = np.zeros((x1.shape[0], x2.shape[0]))

        cols = vars == -1
        if np.any(cols):
            if np.all(cols):
                data1, data2 = x1, x2
            else:
                data1 = x1[:, cols]
                data2 = x2[:, cols] if two_tables else data1
            _distance.euclidean_rows_discrete(distances, data1, data2, dist_missing[cols], dist_missing2[cols], two_tables)

        return np.sqrt(distances)

    def distance_by_cols(self, x1, fit_params):
        vars = fit_params["vars"]
        means = fit_params["means"]
        normalize = fit_params["normalize"]

        if normalize:
            x1 = x1 - means
            x1 /= np.sqrt(2 * vars)

        # adapted from sklearn.metric.euclidean_distances
        xx = row_norms(x1.T, squared=True)[:, np.newaxis]
        distances = safe_sparse_dot(x1.T, x1, dense_output=True)
        distances *= -2
        distances += xx
        distances += xx.T
        np.maximum(distances, 0, out=distances)
        distances.flat[::distances.shape[0] + 1] = 0.0

        if normalize:
            _distance.fix_euclidean_cols_normalized(distances, x1, means, vars)
        else:
            _distance.fix_euclidean_cols(distances, x1, means, vars)
        return np.sqrt(distances)


class Euclidean(FittedDistance):
    supports_sparse = True  # via fallback
    supports_discrete = True
    supports_normalization = True
    fallback = SklDistance('euclidean')
    ModelType = EuclideanModel

    def __new__(cls, e1=None, e2=None, axis=1, impute=False, normalize=False):
        return super().__new__(cls, e1, e2, axis, impute, normalize=normalize)

    def fit_rows(self, x, n_vals):
        """
        Compute statistics needed for normalization and for handling
        missing data for row distances. Returns a dictionary with the
        following keys:
        
        - means: a means of numeric columns; undefined for discrete
        - vars: variances of numeric columns, -1 for discrete, -2 to ignore
        - dist_missing: a 2d-array; dist_missing[col, value] is the distance
            added for the given `value` in discrete column `col` if the value
            for the other row is missing; undefined for numeric columns
        - dist_missing2: the value used for distance if both values are missing;
            used for discrete and numeric columns
        - normalize: set to `self.normalize`, so it is passed to the Cython
            function
            
        A column is marked to be ignored if all its values are nan or if
        `self.normalize` is `True` and the variance of the column is 0.
        """
        super().fit_rows(x, n_vals)
        n_cols = len(n_vals)
        n_bins = max(n_vals)
        means = np.zeros(n_cols, dtype=float)
        vars = np.empty(n_cols, dtype=float)
        dist_missing = np.zeros((n_cols, n_bins), dtype=float)
        dist_missing2 = np.zeros(n_cols, dtype=float)

        for col in range(n_cols):
            column = x[:, col]
            if n_vals[col]:
                vars[col] = -1
                dist_missing[col] = util.bincount(column, minlength=n_bins)[0]
                dist_missing[col] /= max(1, sum(dist_missing[col]))
                dist_missing2[col] = 1 - np.sum(dist_missing[col] ** 2)
                dist_missing[col] = 1 - dist_missing[col]
            elif np.isnan(column).all():  # avoid warnings in nanmean and nanvar
                vars[col] = -2
            else:
                means[col] = util.nanmean(column)
                vars[col] = util.nanvar(column)
                if self.normalize:
                    dist_missing2[col] = 1
                    if vars[col] == 0:
                        vars[col] = -2
                else:
                    dist_missing2[col] = 2 * vars[col]
                    if np.isnan(dist_missing2[col]):
                        dist_missing2[col] = 0

        return dict(means=means, vars=vars,
                    dist_missing=dist_missing, dist_missing2=dist_missing2,
                    normalize=int(self.normalize))

    def fit_cols(self, x, n_vals):
        """
        Compute statistics needed for normalization and for handling
        missing data for columns. Returns a dictionary with the
        following keys:

        - means: column means
        - vars: column variances
        - normalize: set to self.normalize, so it is passed to the Cython
            function
        """
        super().fit_cols(x, n_vals)
        means = np.nanmean(x, axis=0)
        vars = np.nanvar(x, axis=0)
        if self.normalize and (np.isnan(vars).any() or not vars.all()):
            raise ValueError("some columns are constant or have no values")
        return dict(means=means, vars=vars, normalize=int(self.normalize))


class ManhattanModel(FittedDistanceModel):
    distance_by_cols = _distance.manhattan_cols
    distance_by_rows = _distance.manhattan_rows


class Manhattan(FittedDistance):
    supports_sparse = True  # via fallback
    supports_discrete = True
    supports_normalization = True
    fallback = SklDistance('manhattan')
    ModelType = ManhattanModel

    def __new__(cls, e1=None, e2=None, axis=1, impute=False, normalize=False):
        return super().__new__(cls, e1, e2, axis, impute, normalize=normalize)

    def fit_rows(self, x, n_vals):
        """
        Compute statistics needed for normalization and for handling
        missing data for row distances. Returns a dictionary with the
        following keys:

        - medians: medians of numeric columns; undefined for discrete
        - mads: medians of absolute distances from the median for numeric
            columns, -1 for discrete, -2 to ignore
        - dist_missing: a 2d-array; dist_missing[col, value] is the distance
            added for the given `value` in discrete column `col` if the value
            for the other row is missing; undefined for numeric columns
        - dist_missing2: the value used for distance if both values are missing;
            used for discrete and numeric columns
        - normalize: set to `self.normalize`, so it is passed to the Cython
            function

        A column is marked to be ignored if all its values are nan or if
        `self.normalize` is `True` and the median of the column is 0.
        """
        super().fit_rows(x, n_vals)
        n_cols = len(n_vals)
        n_bins = max(n_vals)

        medians = np.zeros(n_cols)
        mads = np.zeros(n_cols)
        dist_missing = np.zeros((n_cols, max(n_vals)))
        dist_missing2 = np.zeros(n_cols)
        for col in range(n_cols):
            column = x[:, col]
            if n_vals[col]:
                mads[col] = -1
                dist_missing[col] = util.bincount(column, minlength=n_bins)[0]
                dist_missing[col] /= max(1, sum(dist_missing[col]))
                dist_missing2[col] = 1 - np.sum(dist_missing[col] ** 2)
                dist_missing[col] = 1 - dist_missing[col]
            elif np.isnan(column).all():  # avoid warnings in nanmedian
                mads[col] = -2
            else:
                medians[col] = np.nanmedian(column)
                mads[col] = np.nanmedian(np.abs(column - medians[col]))
                if self.normalize:
                    dist_missing2[col] = 1
                    if mads[col] == 0:
                        mads[col] = -2
                else:
                    dist_missing2[col] = 2 * mads[col]
        return dict(medians=medians, mads=mads,
                    dist_missing=dist_missing, dist_missing2=dist_missing2,
                    normalize=int(self.normalize))

    def fit_cols(self, x, n_vals):
        """
        Compute statistics needed for normalization and for handling
        missing data for columns. Returns a dictionary with the
        following keys:

        - medians: column medians
        - mads: medians of absolute distances from medians
        - normalize: set to self.normalize, so it is passed to the Cython
            function
        """
        super().fit_cols(x, n_vals)
        medians = np.nanmedian(x, axis=0)
        mads = np.nanmedian(np.abs(x - medians), axis=0)
        if self.normalize and (np.isnan(mads).any() or not mads.all()):
            raise ValueError(
                "some columns have zero absolute distance from median, "
                "or no values")
        return dict(medians=medians, mads=mads, normalize=int(self.normalize))


class CosineModel(FittedDistanceModel):
    distance_by_rows = _distance.cosine_rows
    distance_by_cols = _distance.cosine_cols


class Cosine(FittedDistance):
    supports_sparse = True  # via fallback
    supports_discrete = True
    fallback = SklDistance('cosine')
    ModelType = CosineModel

    def fit_rows(self, x, n_vals):
        """
        Compute statistics needed for normalization and for handling
        missing data for row and column based distances. Although the
        computation is asymmetric, the same statistics are needed in both cases.
        
        Returns a dictionary with the following keys:

        - means: means of numeric columns; relative frequencies of non-zero
            values for discrete
        - vars: variances of numeric columns, -1 for discrete, -2 to ignore
        - dist_missing2: the value used for distance if both values are missing;
            used for discrete and numeric columns

        A column is marked to be ignored if all its values are nan.
        """
        super().fit_rows(x, n_vals)
        n, n_cols = x.shape
        means = np.zeros(n_cols, dtype=float)
        vars = np.empty(n_cols, dtype=float)
        dist_missing2 = np.zeros(n_cols, dtype=float)

        for col in range(n_cols):
            column = x[:, col]
            if n_vals[col]:
                vars[col] = -1
                nonnans = n - np.sum(np.isnan(column))
                means[col] = 1 - np.sum(column == 0) / nonnans
                dist_missing2[col] = means[col]
            elif np.isnan(column).all():  # avoid warnings in nanmean and nanvar
                vars[col] = -2
            else:
                means[col] = util.nanmean(column)
                vars[col] = util.nanvar(column)
                dist_missing2[col] = means[col] ** 2
                if np.isnan(dist_missing2[col]):
                    dist_missing2[col] = 0

        return dict(means=means, vars=vars, dist_missing2=dist_missing2)

    fit_cols = fit_rows


class JaccardModel(FittedDistanceModel):
    distance_by_cols = _distance.jaccard_cols
    distance_by_rows = _distance.jaccard_rows


class Jaccard(FittedDistance):
    supports_sparse = False
    supports_discrete = True
    fallback = SklDistance('jaccard')
    ModelType = JaccardModel

    def fit_rows(self, x, n_vals):
        """
        Compute statistics needed for normalization and for handling
        missing data for row and column based distances. Although the
        computation is asymmetric, the same statistics are needed in both cases.

        Returns a dictionary with the following key:

        - ps: relative freuenceies of non-zero values
        """

        return {
            "ps": np.fromiter(
                (_distance.p_nonzero(x[:, col]) for col in range(len(n_vals))),
                dtype=np.double, count=len(n_vals))}

    fit_cols = fit_rows


class CorrelationDistanceModel(DistanceModel):
    """Helper class for normal and absolute Pearson and Spearman correlation"""
    def __init__(self, absolute, axis=1, impute=False):
        super().__init__(axis, impute)
        self.absolute = absolute

    def compute_distances(self, x1, x2):
        if x2 is None:
            x2 = x1
        rho = self.compute_correlation(x1, x2)
        if self.absolute:
            return (1. - np.abs(rho)) / 2.
        else:
            return (1. - rho) / 2.

    def compute_correlation(self, x1, x2):
        pass


class SpearmanModel(CorrelationDistanceModel):
    def compute_correlation(self, x1, x2):
        rho = stats.spearmanr(x1, x2, axis=self.axis)[0]
        if isinstance(rho, np.float):
            return np.array([[rho]])
        slc = x1.shape[1 - self.axis]
        return rho[:slc, slc:]


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
        if self.axis == 0:
            x1 = x1.T
            x2 = x2.T
        return np.array([[stats.pearsonr(i, j)[0] for j in x2] for i in x1])


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
