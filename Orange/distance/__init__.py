import numpy as np
from scipy import stats
import sklearn.metrics as skl_metrics

from Orange import data
from Orange.misc import DistMatrix
from Orange.preprocess import SklImpute

__all__ = ['Euclidean', 'Manhattan', 'Cosine', 'Jaccard', 'SpearmanR', 'SpearmanRAbsolute',
           'PearsonR', 'PearsonRAbsolute', 'Mahalanobis', 'MahalanobisDistance']

def _preprocess(table):
    """Remove categorical attributes and impute missing values."""
    if not len(table):
        return table
    new_domain = data.Domain([a for a in table.domain.attributes if a.is_continuous],
                             table.domain.class_vars,
                             table.domain.metas)
    new_data = data.Table(new_domain, table)
    new_data = SklImpute(new_data)
    return new_data


def _orange_to_numpy(x):
    """Convert :class:`Orange.data.Table` and :class:`Orange.data.RowInstance` to :class:`numpy.ndarray`."""
    if isinstance(x, data.Table):
        return x.X
    elif isinstance(x, data.Instance):
        return np.atleast_2d(x.x)
    elif isinstance(x, np.ndarray):
        return np.atleast_2d(x)
    else:
        return x    # e.g. None


class Distance:
    def __call__(self, e1, e2=None, axis=1, impute=False):
        """
        :param e1: input data instances, we calculate distances between all pairs
        :type e1: :class:`Orange.data.Table` or :class:`Orange.data.RowInstance` or :class:`numpy.ndarray`
        :param e2: optional second argument for data instances
           if provided, distances between each pair, where first item is from e1 and second is from e2, are calculated
        :type e2: :class:`Orange.data.Table` or :class:`Orange.data.RowInstance` or :class:`numpy.ndarray`
        :param axis: if axis=1 we calculate distances between rows,
           if axis=0 we calculate distances between columns
        :type axis: int
        :param impute: if impute=True all NaN values in matrix are replaced with 0
        :type impute: bool
        :return: the matrix with distances between given examples
        :rtype: :class:`Orange.misc.distmatrix.DistMatrix`
        """
        raise NotImplementedError('Distance is an abstract class and should not be used directly.')


class SklDistance(Distance):
    """Generic scikit-learn distance."""
    def __init__(self, metric, name, supports_sparse):
        """
        Args:
            metric: The metric to be used for distance calculation
            name (str): Name of the distance
            supports_sparse (boolean): Whether this metric works on sparse data or not.
        """
        self.metric = metric
        self.name = name
        self.supports_sparse = supports_sparse

    def __call__(self, e1, e2=None, axis=1, impute=False):
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        dist = skl_metrics.pairwise.pairwise_distances(x1, x2, metric=self.metric)
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2, axis)
        else:
            dist = DistMatrix(dist)
        return dist

Euclidean = SklDistance('euclidean', 'Euclidean', True)
Manhattan = SklDistance('manhattan', 'Manhattan', True)
Cosine = SklDistance('cosine', 'Cosine', True)
Jaccard = SklDistance('jaccard', 'Jaccard', False)


class SpearmanDistance(Distance):
    """ Generic Spearman's rank correlation coefficient. """
    def __init__(self, absolute, name):
        """
        Constructor for Spearman's and Absolute Spearman's distances.

        Args:
            absolute (boolean): Whether to use absolute values or not.
            name (str): Name of the distance

        Returns:
            If absolute=True return Spearman's Absolute rank class else return Spearman's rank class.
        """
        self.absolute = absolute
        self.name = name
        self.supports_sparse = False

    def __call__(self, e1, e2=None, axis=1, impute=False):
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if x2 is None:
            x2 = x1
        slc = len(x1) if axis == 1 else x1.shape[1]
        rho, _ = stats.spearmanr(x1, x2, axis=axis)
        if np.isnan(rho).any() and impute:
            rho = np.nan_to_num(rho)
        if self.absolute:
            dist = (1. - np.abs(rho)) / 2.
        else:
            dist = (1. - rho) / 2.
        if isinstance(dist, np.float):
            dist = np.array([[dist]])
        elif isinstance(dist, np.ndarray):
            dist = dist[:slc, slc:]
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2, axis)
        else:
            dist = DistMatrix(dist)
        return dist

SpearmanR = SpearmanDistance(absolute=False, name='Spearman')
SpearmanRAbsolute = SpearmanDistance(absolute=True, name='Spearman absolute')


class PearsonDistance(Distance):
    """ Generic Pearson's rank correlation coefficient. """
    def __init__(self, absolute, name):
        """
        Constructor for Pearson's and Absolute Pearson's distances.

        Args:
            absolute (boolean): Whether to use absolute values or not.
            name (str): Name of the distance

        Returns:
            If absolute=True return Pearson's Absolute rank class else return Pearson's rank class.
        """
        self.absolute = absolute
        self.name = name
        self.supports_sparse = False

    def __call__(self, e1, e2=None, axis=1, impute=False):
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if x2 is None:
            x2 = x1
        if axis == 0:
            x1 = x1.T
            x2 = x2.T
        rho = np.array([[stats.pearsonr(i, j)[0] for j in x2] for i in x1])
        if np.isnan(rho).any() and impute:
            rho = np.nan_to_num(rho)
        if self.absolute:
            dist = (1. - np.abs(rho)) / 2.
        else:
            dist = (1. - rho) / 2.
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2, axis)
        else:
            dist = DistMatrix(dist)
        return dist

PearsonR = PearsonDistance(absolute=False, name='Pearson')
PearsonRAbsolute = PearsonDistance(absolute=True, name='Pearson absolute')


class MahalanobisDistance(Distance):
    """Mahalanobis distance."""
    def __init__(self, data=None, axis=1, name='Mahalanobis'):
        self.name = name
        self.supports_sparse = False
        self.axis = None
        self.VI = None
        if data is not None:
            self.fit(data, axis)

    def fit(self, data, axis=1):
        """
        Compute the covariance matrix needed for calculating distances.

        Args:
            data: The dataset used for calculating covariances.
            axis: If axis=1 we calculate distances between rows, if axis=0 we calculate distances between columns.
        """
        x = _orange_to_numpy(data)
        if axis == 0:
            x = x.T
        n, m = x.shape
        if n <= m:
            raise ValueError('Too few observations for the number of dimensions.')
        self.axis = axis
        self.VI = np.linalg.inv(np.cov(x.T))

    def __call__(self, e1, e2=None, axis=None, impute=False):
        assert self.VI is not None, "Mahalanobis distance must be initialized with the fit() method."

        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)

        if axis is not None:
            assert axis == self.axis, "Axis must match its value at initialization."
        if self.axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        if x1.shape[1] != self.VI.shape[0] or x2 is not None and x2.shape[1] != self.VI.shape[0]:
            raise ValueError('Incorrect number of features.')

        dist = skl_metrics.pairwise.pairwise_distances(x1, x2, metric='mahalanobis', VI=self.VI)
        if np.isnan(dist).any() and impute:
            dist = np.nan_to_num(dist)
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2, self.axis)
        else:
            dist = DistMatrix(dist)
        return dist

Mahalanobis = MahalanobisDistance()
