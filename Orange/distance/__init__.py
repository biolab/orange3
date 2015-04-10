import numpy as np
from scipy import stats, sparse
import sklearn.metrics as skl_metrics
import sklearn.preprocessing as skl_preprocessing

from Orange import data
from Orange.misc import DistMatrix


def _preprocess(table):
    """Remove categorical attributes and impute missing values."""
    new_domain = data.Domain([i for i in table.domain.attributes
                              if isinstance(i, data.ContinuousVariable)], table.domain.class_var)
    new_data = data.Table(new_domain, table)
    new_data.X = skl_preprocessing.Imputer().fit_transform(new_data.X)
    new_data.X = new_data.X if sparse.issparse(new_data.X) else np.squeeze(new_data.X)
    return new_data


def _orange_to_numpy(x):
    """Convert :class:`Orange.data.Table` and :class:`Orange.data.RowInstance` to :class:`numpy.ndarray`."""
    if isinstance(x, data.Table):
        return x.X
    elif isinstance(x, data.RowInstance):
        return x.x
    else:
        return x


class Distance():
    def __call__(self, e1, e2=None, axis=1):
        """
        :param e1: input data instances, we calculate distances between all pairs
        :type e1: :class:`Orange.data.Table` or :class:`Orange.data.RowInstance` or :class:`numpy.ndarray`
        :param e2: optional second argument for data instances
           if provided, distances between each pair, where first item is from e1 and second is from e2, are calculated
        :type e2: :class:`Orange.data.Table` or :class:`Orange.data.RowInstance` or :class:`numpy.ndarray`
        :param axis: if axis=1 we calculate distances between rows,
           if axis=0 we calculate distances between columns
        :type axis: int
        :return: the matrix with distances between given examples
        :rtype: :class:`Orange.misc.distmatrix.DistMatrix`
        """
        raise NotImplementedError('Distance is an abstract class and should not be used directly.')


class SklDistance(Distance):
    """Generic scikit-learn distance."""
    def __init__(self, metric):
        """
        :param metric: The metric to be used for distance calculation
        :type metric: str
        """
        self.metric = metric

    def __call__(self, e1, e2=None, axis=1):
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        if not sparse.issparse(x1):
            x1 = np.atleast_2d(x1)
        if e2 is not None and not sparse.issparse(x2):
            x2 = np.atleast_2d(x2)
        dist = skl_metrics.pairwise.pairwise_distances(x1, x2, metric=self.metric)
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2, axis)
        else:
            dist = DistMatrix(dist)
        return dist

Euclidean = SklDistance('euclidean')
Manhattan = SklDistance('manhattan')
Cosine = SklDistance('cosine')
Jaccard = SklDistance('jaccard')


class SpearmanDistance(Distance):
    """ Generic Spearman's rank correlation coefficient. """
    def __init__(self, absolute):
        """
        Constructor for Spearman's and Absolute Spearman's distances.

        :param absolute: Whether to use absolute values or not.
        :return: If absolute=True return Spearman's Absolute rank class else return Spearman's rank class.
        """
        self.absolute = absolute

    def __call__(self, e1, e2=None, axis=1):
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if x2 is None:
            x2 = x1
        if x1.ndim == 1 or x2.ndim == 1:
            axis = 0
            slc = len(x1) if x1.ndim > 1 else 1
        else:
            slc = len(x1) if axis == 1 else x1.shape[1]
        # stats.spearmanr does not work when e1=Table and e2=RowInstance
        # so we replace e1 and e2 and then transpose the result
        transpose = False
        if x1.ndim == 2 and x2.ndim == 1:
            x1, x2 = x2, x1
            slc = len(e1) if x1.ndim > 1 else 1
            transpose = True
        rho, _ = stats.spearmanr(x1, x2, axis=axis)
        if self.absolute:
            dist = (1. - np.abs(rho)) / 2.
        else:
            dist = (1. - rho) / 2.
        if isinstance(dist, np.float):
            dist = np.array([[dist]])
        elif isinstance(dist, np.ndarray):
            dist = dist[:slc, slc:]
        if transpose:
           dist = dist.T
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2, axis)
        else:
            dist = DistMatrix(dist)
        return dist

SpearmanR = SpearmanDistance(absolute=False)
SpearmanRAbsolute = SpearmanDistance(absolute=True)


class PearsonDistance(Distance):
    """ Generic Pearson's rank correlation coefficient. """
    def __init__(self, absolute):
        """
        Constructor for Pearson's and Absolute Pearson's distances.

        :param absolute: Whether to use absolute values or not.
        :return: If absolute=True return Pearson's Absolute rank class else return Pearson's rank class.
        """
        self.absolute = absolute

    def __call__(self, e1, e2=None, axis=1):
        x1 = _orange_to_numpy(e1)
        x2 = _orange_to_numpy(e2)
        if x2 is None:
            x2 = x1
        if axis == 0:
            x1 = x1.T
            x2 = x2.T
        if x1.ndim == 1:
            x1 = list([x1])
        if x2.ndim == 1:
            x2 = list([x2])
        rho = np.array([[stats.pearsonr(i, j)[0] for j in x2] for i in x1])
        if self.absolute:
            dist = (1. - np.abs(rho)) / 2.
        else:
            dist = (1. - rho) / 2.
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2, axis)
        else:
            dist = DistMatrix(dist)
        return dist


PearsonR = PearsonDistance(absolute=False)
PearsonRAbsolute = PearsonDistance(absolute=True)
