# This module defines abstract base classes; derived classes are abstract, too
# pylint: disable=abstract-method

import numpy as np
import sklearn.metrics as skl_metrics

from Orange.data import Table, Domain, Instance, RowInstance
from Orange.misc import DistMatrix
from Orange.preprocess import SklImpute
from Orange.statistics import util


# TODO: When we upgrade to numpy 1.13, change use argument copy=False in
# nan_to_num instead of assignment

# TODO this *private* function is called from several widgets to prepare
# data for calling the below classes. After we (mostly) stopped relying
# on sklearn.metrics, this is (mostly) unnecessary
# Afterwards, also remove the following line:
# pylint: disable=redefined-outer-name
def _preprocess(table, impute=True):
    """Remove categorical attributes and impute missing values."""
    if not len(table):  # this can be an array, pylint: disable=len-as-condition
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
def remove_discrete_features(data, to_metas=False):
    """Remove discrete columns from the data."""
    new_domain = Domain(
        [a for a in data.domain.attributes if a.is_continuous],
        data.domain.class_vars,
        data.domain.metas
        + (() if not to_metas
           else tuple(a for a in data.domain.attributes if not a.is_continuous))
    )
    return data.transform(new_domain)


def remove_nonbinary_features(data, to_metas=False):
    """Remove non-binary columns from the data."""
    new_domain = Domain(
        [a for a in data.domain.attributes
         if a.is_discrete and len(a.values) == 2],
        data.domain.class_vars,
        data.domain.metas +
        (() if not to_metas
         else tuple(a for a in data.domain.attributes
               if not (a.is_discrete and len(a.values) == 2))
         if to_metas else ()))
    return data.transform(new_domain)

def impute(data):
    """Impute missing values."""
    return SklImpute()(data)


def _orange_to_numpy(x):
    """
    Return :class:`numpy.ndarray` (dense or sparse) with attribute data
    from the given instance of :class:`Orange.data.Table`,
    :class:`Orange.data.RowInstance` or :class:`Orange.data.Instance`.
    """
    if isinstance(x, Table):
        return x.X
    elif isinstance(x, Instance):
        return np.atleast_2d(x.x)
    elif isinstance(x, np.ndarray):
        return np.atleast_2d(x)
    else:
        return x  # e.g. None


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
        e1 (:obj:`~Orange.data.Table` or :obj:`~Orange.data.Instance` or \
                :obj:`np.ndarray` or `None`):
            data on which to train the model and compute the distances
        e2 (:obj:`~Orange.data.Table` or :obj:`~Orange.data.Instance` or \
                :obj:`np.ndarray` or `None`):
            if present, the class computes distances with pairs coming from
            the two tables
        axis (int):
            axis over which the distances are computed, 1 (default) for
            rows, 0 for columns
        impute (bool):
            if `True` (default is `False`), nans in the computed distances
            are replaced with zeros, and infs with very large numbers.
        callback (callable or None):
            callback function

    Attributes:
        axis (int):
            axis over which the distances are computed, 1 (default) for
            rows, 0 for columns
        impute (bool):
            if `True` (default is `False`), nans in the computed distances
            are replaced with zeros, and infs with very large numbers.
        normalize (bool):
            if `True`, columns are normalized before computation. This attribute
            applies only if the distance supports normalization.

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

    # Predefined here to silence pylint, which doesn't look into __new__
    normalize = False
    axis = 1
    impute = False

    def __new__(cls, e1=None, e2=None, axis=1, impute=False,
                callback=None, **kwargs):
        self = super().__new__(cls)
        self.axis = axis
        self.impute = impute
        self.callback = callback
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
                # pylint: disable=not-callable
                return fallback(e1, e2, axis, impute)

        # Magic constructor
        model = self.fit(e1)
        return model(e1, e2)

    def fit(self, data):
        """
        Abstract method returning :obj:`DistanceModel` fit to the data

        Args:
            e1 (Orange.data.Table, Orange.data.Instance, np.ndarray):
                data for fitting the distance model

        Returns:
            model (DistanceModel)
        """
        raise NotImplementedError

    @staticmethod
    def check_no_discrete(n_vals):
        """
        Raise an exception if there are any discrete attributes.

        Args:
            n_vals (list of int): number of attributes values, 0 for continuous
        """
        if any(n_vals):
            raise ValueError("columns with discrete values are incommensurable")


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
        callback (callable or None):
            callback function

    """
    def __init__(self, axis, impute=False, callback=None):
        self._axis = axis
        self.impute = impute
        self.callback = callback

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
        with np.errstate(invalid="ignore"):  # nans are handled below
            dist = self.compute_distances(x1, x2)
            if self.impute and np.isnan(dist).any():
                dist = np.nan_to_num(dist)
            if isinstance(e1, (Table, RowInstance)):
                dist = DistMatrix(dist, e1, e2, self.axis)
            else:
                dist = DistMatrix(dist)
            return dist

    def compute_distances(self, x1, x2):
        """
        Abstract method for computation of distances between rows or columns of
        `x1`, or between rows of `x1` and `x2`. Do not call directly."""
        raise NotImplementedError


class FittedDistanceModel(DistanceModel):
    """
    Base class for models that store attribute-related data for normalization
    and imputation, and that treat discrete and continuous columns separately.

    Attributes:
        attributes (list of `Variable`): attributes on which the model was fit
        discrete (np.ndarray): bool array indicating discrete attributes
        continuous (np.ndarray): bool array indicating continuous attributes
        normalize (bool):
            if `True` (default is `False`) continuous columns are normalized
        callback (callable or None): callback function
    """
    def __init__(self, attributes, axis=1, impute=False, callback=None):
        super().__init__(axis, impute, callback)
        self.attributes = attributes
        self.discrete = None
        self.continuous = None
        self.normalize = False

    def __call__(self, e1, e2=None):
        if self.attributes is not None and (
                e1.domain.attributes != self.attributes
                or e2 is not None and e2.domain.attributes != self.attributes):
            raise ValueError("mismatching domains")
        return super().__call__(e1, e2)

    def continuous_columns(self, x1, x2, offset, scale):
        """
        Extract and scale continuous columns from data tables.
        If the second table is None, it defaults to the first table.

        Values are scaled if `self.normalize` is `True`.

        Args:
            x1 (np.ndarray): first table
            x2 (np.ndarray or None): second table
            offset (float): a constant (e.g. mean, median) subtracted from data
            scale: (float): divider (e.g. deviation)

        Returns:
            data1 (np.ndarray): scaled continuous columns from `x1`
            data2 (np.ndarray): scaled continuous columns from `x2` or `x1`
        """
        if self.continuous.all() and not self.normalize:
            data1, data2 = x1, x2
        else:
            data1 = x1[:, self.continuous]
            if x2 is not None:
                data2 = x2[:, self.continuous]
            if self.normalize:
                data1 = x1[:, self.continuous]
                data1 -= offset
                data1 /= scale
                if x2 is not None:
                    data2 = x2[:, self.continuous]
                    data2 -= offset
                    data2 /= scale
        if x2 is None:
            data2 = data1
        return data1, data2

    def discrete_columns(self, x1, x2):
        """
        Return discrete columns from the given tables.
        If the second table is None, it defaults to the first table.
        """
        if self.discrete.all():
            data1, data2 = x1, x1 if x2 is None else x2
        else:
            data1 = x1[:, self.discrete]
            data2 = data1 if x2 is None else x2[:, self.discrete]
        return data1, data2


class FittedDistance(Distance):
    """
    Base class for fitting models that store attribute-related data for
    normalization and imputation, and that treat discrete and continuous
    columns separately.

    The class implements a method `fit` that calls either `fit_columns`
    or `fit_rows` with the data and the number of values for discrete
    attributes. The provided method `fit_rows` calls methods
    `get_discrete_stats` and `get_continuous_stats` that can be implemented
    in derived classes.

    Class attribute `rows_model_type` contains the type of the model returned by
    `fit_rows`.
    """
    rows_model_type = None  #: Option[FittedDistanceModel]

    def fit(self, data):
        """
        Prepare the data on attributes, call `fit_cols` or `fit_rows` and
        return the resulting model.
        """
        x = _orange_to_numpy(data)
        if hasattr(data, "domain"):
            attributes = data.domain.attributes
            n_vals = np.fromiter(
                (len(attr.values) if attr.is_discrete else 0
                 for attr in attributes),
                dtype=np.int32, count=len(attributes))
        else:
            assert isinstance(x, np.ndarray)
            attributes = None
            n_vals = np.zeros(x.shape[1], dtype=np.int32)
        return [self.fit_cols, self.fit_rows][self.axis](attributes, x, n_vals)

    def fit_cols(self, attributes, x, n_vals):
        """
        Return DistanceModel for computation of distances between columns.
        Derived classes must define this method.

        Args:
            attributes (list of Orange.data.Variable): list of attributes
            x (np.ndarray): data
            n_vals (np.ndarray): number of attribute values, 0 for continuous
        """
        raise NotImplementedError

    def fit_rows(self, attributes, x, n_vals):
        """
        Return a DistanceModel for computation of distances between rows.

        The model type is `self.row_distance_model`. It stores the data for
        imputation of discrete and continuous values, and for normalization
        of continuous values. Typical examples are the Euclidean and Manhattan
        distance, for which the following data is stored:

        For continuous columns:

        - offsets[col] is the number subtracted from values in column `col`
        - scales[col] is the divisor for values in columns `col`
        - dist_missing2_cont[col]: the value used for distance between two
            missing values in column `col`

        For discrete values:

        - dist_missing_disc[col, value] is the distance added for the given
            `value` in the column `col` if the value for the other row is
            missing
        - dist_missing2_disc[col]: the distance between two missing values in
            column `col`
        """
        n_cols = len(n_vals)

        discrete = n_vals > 0
        n_bins = max(n_vals, default=0)
        n_discrete = sum(discrete)
        dist_missing_disc = np.zeros((n_discrete, n_bins), dtype=float)
        dist_missing2_disc = np.zeros(n_discrete, dtype=float)

        continuous = ~discrete
        n_continuous = sum(continuous)
        offsets = np.zeros(n_continuous, dtype=float)
        scales = np.empty(n_continuous, dtype=float)
        dist_missing2_cont = np.zeros(n_continuous, dtype=float)

        curr_disc = curr_cont = 0
        for col in range(n_cols):
            column = x[:, col]
            if np.isnan(column).all():
                continuous[col] = discrete[col] = False
            elif discrete[col]:
                discrete_stats = self.get_discrete_stats(column, n_bins)
                if discrete_stats is not None:
                    dist_missing_disc[curr_disc], \
                    dist_missing2_disc[curr_disc] = discrete_stats
                    curr_disc += 1
            else:
                continuous_stats = self.get_continuous_stats(column)
                if continuous_stats is not None:
                    offsets[curr_cont], scales[curr_cont],\
                    dist_missing2_cont[curr_cont] = continuous_stats
                    curr_cont += 1
                else:
                    continuous[col] = False
        # pylint: disable=not-callable
        return self.rows_model_type(
            attributes, impute, getattr(self, "normalize", False),
            continuous, discrete,
            offsets[:curr_cont], scales[:curr_cont],
            dist_missing2_cont[:curr_cont],
            dist_missing_disc, dist_missing2_disc,
            self.callback)

    @staticmethod
    def get_discrete_stats(column, n_bins):
        """
        Return tables used computing distance between missing discrete values.

        Args:
            column (np.ndarray): column data
            n_bins (int): maximal number of bins in the dataset

        Returns:
            dist_missing_disc (np.ndarray): `dist_missing_disc[value]` is
                1 - probability of `value`, which is used as the distance added
                for the given `value` in the column `col` if the value for the
                other row is missing
            dist_missing2_disc (float): the distance between two missing
                values in this columns
        """
        dist = util.bincount(column, minlength=n_bins)[0]
        dist /= max(1, sum(dist))
        return 1 - dist, 1 - np.sum(dist ** 2)

    def get_continuous_stats(self, column):
        """
        Compute statistics for imputation and normalization of continuous data.
        Derived classes must define this method.

        Args:
            column (np.ndarray): column data

        Returns:
            offset (float): the number subtracted from values in column
            scales (float): the divisor for values in column
            dist_missing2_cont (float): the value used for distance between two
                missing values in column
        """
        raise NotImplementedError


# Fallbacks for distances in sparse data
# To be removed as the corresponding functionality is implemented properly

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
        if isinstance(e1, (Table, RowInstance)):
            dist_matrix = DistMatrix(dist, e1, e2, axis)
        else:
            dist_matrix = DistMatrix(dist)
        return dist_matrix
