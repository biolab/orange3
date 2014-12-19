import numpy as np
from scipy import stats, sparse
from sklearn import metrics, preprocessing

from Orange import data
from Orange.misc import DistMatrix


def _impute(data):
    """Imputation transformer for completing missing values."""
    imp_data = data.Table(data)
    imp_data.X = preprocessing.Imputer().fit_transform(imp_data.X)
    imp_data.X = imp_data.X if sparse.issparse(imp_data.X) else np.squeeze(imp_data.X)
    return imp_data


def _orange_to_numpy(x):
    if isinstance(x, data.Table):
        return x.X
    elif isinstance(x, data.RowInstance):
        return x.x
    else:
        return x


class SklDistance():
    """
        Generic scikit-learn distance.

        NOTE: The VI argument is only used for Mahalanobis distance!
    """
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, e1, e2=None, axis=1, **kwargs):
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
        dist = metrics.pairwise.pairwise_distances(x1, x2, metric=self.metric, **kwargs)
        if isinstance(e1, data.Table) or isinstance(e1, data.RowInstance):
            dist = DistMatrix(dist, e1, e2)
        else:
            dist = DistMatrix(dist)
        return dist


class SklMahalanobis(SklDistance):
    def __init__(self):
        self.metric = 'mahalanobis'

    def __call__(self, e1, e2=None, axis=1, VI=None):
        return super().__call__(e1, e2, axis, VI=VI)


Euclidean = SklDistance('euclidean')
Manhattan = SklDistance('manhattan')
Cosine = SklDistance('cosine')
Jaccard = SklDistance('jaccard')
Mahalanobis = SklMahalanobis()


class SpearmanDistance():
    """Spearman's rank correlation coefficient."""

    def __init__(self, absolute):
        self.absolute = absolute

    def __call__(self, e1, e2=None, axis=1):
        """
        :param e1: data instances.
        :param e2: data instances.

        Returns Spearman's dissimilarity between e1 and e2,
        i.e.

        .. math:: (1-r)/2

        where r is Spearman's rank coefficient.
        """
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
            dist = DistMatrix(dist, e1, e2)
        else:
            dist = DistMatrix(dist)
        return dist

SpearmanR = SpearmanDistance(absolute=False)
SpearmanRAbsolute = SpearmanDistance(absolute=True)


class PearsonDistance():
    """Pearson's rank correlation coefficient."""

    def __init__(self, absolute):
        self.absolute = absolute

    def __call__(self, e1, e2=None, axis=1):
        """
        :param e1: data instances.
        :param e2: data instances.

        Returns Pearson's dissimilarity between e1 and e2,
        i.e.

        .. math:: (1-r)/2

        where r is Pearson's rank coefficient.
        """
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
            dist = DistMatrix(dist, e1, e2)
        else:
            dist = DistMatrix(dist)
        return dist


PearsonR = PearsonDistance(absolute=False)
PearsonRAbsolute = PearsonDistance(absolute=True)
