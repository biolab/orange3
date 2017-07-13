import numpy as np
from scipy import stats
import sklearn.metrics as skl_metrics


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


def remove_discrete_features(data):
    new_domain = Domain(
        [a for a in data.domain.attributes if a.is_continuous],
        data.domain.class_vars,
        data.domain.metas)
    return data.transform(new_domain)


def impute(data):
    return SklImpute()(data)


def _orange_to_numpy(x):
    """Convert :class:`Orange.data.Table` and :class:`Orange.data.RowInstance`
    to :class:`numpy.ndarray`.
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
    supports_sparse = False
    supports_discrete = False
    supports_normalization = False
    supports_missing = True

    def __new__(cls, e1=None, e2=None, axis=1, **kwargs):
        self = super().__new__(cls)
        self.axis = axis
        # Ugly, but needed for backwards compatibility hack below, to allow
        # setting parameters like 'normalize'
        self.__dict__.update(**kwargs)
        if e1 is None:
            return self

        # Handling sparse data and maintaining backwards compatibility with
        # old-style calls
        if (not hasattr(e1, "domain")
                or hasattr(e1, "is_sparse") and e1.is_sparse()):
            fallback = getattr(self, "fallback", None)
            if fallback is not None:
                return fallback(e1, e2, axis,
                                impute=kwargs.get("impute", False))
        model = self.fit(e1)
        return model(e1, e2)

    def fit(self, e1):
        pass


class DistanceModel:
    def __init__(self, axis, impute=False):
        self.axis = axis
        self.impute = impute

    def __call__(self, e1, e2=None):
        """
        If e2 is omitted, calculate distances between all rows (axis=1) or
        columns (axis=2) of e1. If e2 is present, calculate distances between
        all pairs if rows from e1 and e2.

        Args:
            e1 (Orange.data.Table or Orange.data.RowInstance or numpy.ndarray):
                input data
            e2 (Orange.data.Table or Orange.data.RowInstance or numpy.ndarray):
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
        if isinstance(e1, Table) or isinstance(e1, RowInstance):
            dist = DistMatrix(dist, e1, e2, self.axis)
        else:
            dist = DistMatrix(dist)
        return dist

    def compute_distances(self, x1, x2):
        pass


class FittedDistanceModel(DistanceModel):
    def __init__(self, attributes, axis, impute=False, fit_params=None):
        super().__init__(axis, impute)
        self.attributes = attributes
        self.fit_params = fit_params

    def __call__(self, e1, e2=None):
        if e1.domain.attributes != self.attributes or \
                e2 is not None and e2.domain.attributes != self.attributes:
            raise ValueError("mismatching domains")
        return super().__call__(e1, e2)

    def compute_distances(self, x1, x2=None):
        if self.axis == 0:
            return self.distance_by_cols(x1, self.fit_params)
        else:
            return self.distance_by_rows(
                x1,
                x2 if x2 is not None else x1,
                x2 is not None,
                self.fit_params)


class FittedDistance(Distance):
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
    def __init__(self, metric, name, supports_sparse):
        """
        Args:
            metric: The metric to be used for distance calculation
            name (str): Name of the distance
            supports_sparse (boolean): Whether this metric works on sparse data
                or not.
        """
        self.metric = metric
        self.name = name
        self.supports_sparse = supports_sparse

    def __call__(self, e1, e2=None, axis=1, impute=False):
        """
        :param e1: input data instances, we calculate distances between all
            pairs
        :type e1: :class:`Orange.data.Table` or
            :class:`Orange.data.RowInstance` or :class:`numpy.ndarray`
        :param e2: optional second argument for data instances if provided,
            distances between each pair, where first item is from e1 and
            second is from e2, are calculated
        :type e2: :class:`Orange.data.Table` or
            :class:`Orange.data.RowInstance` or :class:`numpy.ndarray`
        :param axis: if axis=1 we calculate distances between rows, if axis=0
            we calculate distances between columns
        :type axis: int
        :param impute: if impute=True all NaN values in matrix are replaced
            with 0
        :type impute: bool
        :return: the matrix with distances between given examples
        :rtype: :class:`Orange.misc.distmatrix.DistMatrix`
        """
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        dist = skl_metrics.pairwise.pairwise_distances(
            x1, x2, metric=self.metric)
        if isinstance(e1, Table) or isinstance(e1, RowInstance):
            dist_matrix = DistMatrix(dist, e1, e2, axis)
        else:
            dist_matrix = DistMatrix(dist)
        return dist_matrix


class EuclideanModel(FittedDistanceModel):
    distance_by_cols = _distance.euclidean_cols
    distance_by_rows = _distance.euclidean_rows


class Euclidean(FittedDistance):
    supports_sparse = True  # via fallback
    supports_discrete = True
    supports_normalization = True
    fallback = SklDistance('euclidean', 'Euclidean', True)
    ModelType = EuclideanModel

    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("normalize", False)
        return super().__new__(cls, *args, **kwargs)

    def fit_rows(self, x, n_vals):
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
    fallback = SklDistance('manhattan', 'Manhattan', True)
    ModelType = ManhattanModel

    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("normalize", False)
        return super().__new__(cls, *args, **kwargs)

    def fit_rows(self, x, n_vals):
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
    fallback = SklDistance('cosine', 'Cosine', True)
    ModelType = CosineModel

    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("normalize", False)
        return super().__new__(cls, *args, **kwargs)

    def fit_rows(self, x, n_vals):
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
    fallback = SklDistance('jaccard', 'Jaccard', True)
    ModelType = JaccardModel

    def fit_rows(self, x, n_vals):
        return {
            "ps": np.fromiter(
                (_distance.p_nonzero(x[:, col]) for col in range(len(n_vals))),
                dtype=np.double, count=len(n_vals))}

    fit_cols = fit_rows


class CorrelationDistanceModel(DistanceModel):
    def __init__(self, absolute, axis=1, impute=False):
        super().__init__(axis, impute)
        self.absolute = absolute

    def compute_distances(self, x1, x2):
        if x2 is None:
            x2 = x1
        rho = self.compute_correlation(x1, x2)
        if np.isnan(rho).any() and impute:
            rho = np.nan_to_num(rho)
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
        return SpearmanModel(False, self.axis, getattr(self, "impute", False))


class SpearmanRAbsolute(CorrelationDistance):
    def fit(self, _):
        return SpearmanModel(True, self.axis, getattr(self, "impute", False))


class PearsonModel(CorrelationDistanceModel):
    def compute_correlation(self, x1, x2):
        if self.axis == 0:
            x1 = x1.T
            x2 = x2.T
        return np.array([[stats.pearsonr(i, j)[0] for j in x2] for i in x1])


class PearsonR(CorrelationDistance):
    def fit(self, _):
        return PearsonModel(False, self.axis, getattr(self, "impute", False))


class PearsonRAbsolute(CorrelationDistance):
    def fit(self, _):
        return PearsonModel(True, self.axis, getattr(self, "impute", False))


class Mahalanobis(Distance):
    supports_sparse = False
    supports_missing = False

    def fit(self, data):
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
        return MahalanobisModel(self.axis, getattr(self, "impute", False), vi)


class MahalanobisModel(DistanceModel):
    def __init__(self, axis, impute, vi):
        super().__init__(axis)
        self.impute = impute
        self.vi = vi

    def __call__(self, e1, e2=None, impute=None):
        # backward compatibility
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

        dist = skl_metrics.pairwise.pairwise_distances(
            x1, x2, metric='mahalanobis', VI=self.vi)
        if np.isnan(dist).any() and self.impute:
            dist = np.nan_to_num(dist)
        return dist


# Backward compatibility

class MahalanobisDistance:
    def __new__(cls, data=None, axis=1, _='Mahalanobis'):
        if data is None:
            return cls
        return Mahalanobis(axis=axis).fit(data)
