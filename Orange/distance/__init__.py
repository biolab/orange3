import numpy as np
from scipy import stats
from sklearn import metrics

from Orange import data
from Orange.misc import DistMatrix


class Euclidean():
    """Euclidean distance."""
    def __call__(self, e1, e2=None, axis=1):
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else None
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        dist = metrics.pairwise.pairwise_distances(x1, x2, metric='euclidean')
        if dist.size == 1:
            dist = dist[0, 0]
        else:
            dist = DistMatrix(dist, e1, e2)
        return dist


class Manhattan():
    """Manhattan distance."""
    def __call__(self, e1, e2=None, axis=1):
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else None
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        dist = metrics.pairwise.pairwise_distances(x1, x2, metric='manhattan')
        if dist.size == 1:
            dist = dist[0, 0]
        else:
            dist = DistMatrix(dist, e1, e2)
        return dist


class Cosine():
    """Cosine distance."""
    def __call__(self, e1, e2=None, axis=1):
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else None
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        dist = metrics.pairwise.pairwise_distances(x1, x2, metric='cosine')
        if dist.size == 1:
            dist = dist[0, 0]
        else:
            dist = DistMatrix(dist, e1, e2)
        return dist


class Jaccard():
    """Jaccard distance"""
    def __call__(self, e1, e2=None, axis=1):
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else None
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        if isinstance(e1, data.RowInstance):
            x1 = x1.reshape(1, len(x1))
        if isinstance(e2, data.RowInstance):
            x2 = x2.reshape(1, len(x2))
        dist = metrics.pairwise.pairwise_distances(x1, x2, metric='jaccard')
        if dist.size == 1:
            dist = dist[0, 0]
        else:
            dist = DistMatrix(dist, e1, e2)
        return dist


class Mahalanobis():
    """ Mahalanobis distance."""
    def __call__(self, e1, e2=None, VI=None, axis=1):
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else None
        if axis == 0:
            x1 = x1.T
            if x2 is not None:
                x2 = x2.T
        if isinstance(e1, data.RowInstance):
            x1 = x1.reshape(1, len(x1))
        if isinstance(e2, data.RowInstance):
            x2 = x2.reshape(1, len(x2))
        dist = metrics.pairwise.pairwise_distances(x1, x2, metric='mahalanobis', VI=VI)
        if dist.size == 1:
            dist = dist[0, 0]
        else:
            dist = DistMatrix(dist, e1, e2)
        return dist


class SpearmanR():
    """Spearman's rank correlation coefficient."""
    def __call__(self, e1, e2=None, axis=1):
        """
        :param e1: data instances.
        :param e2: data instances.

        Returns Spearman's dissimilarity between e1 and e2,
        i.e.

        .. math:: (1-r)/2

        where r is Spearman's rank coefficient.
        """
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else x1
        if x1.ndim == 1 or x2.ndim == 1:
            axis = 0
            slc = len(x1) if x1.ndim > 1 else 1
        else:
            slc = len(x1) if axis == 1 else len(e1.domain.attributes)
        # stats.spearmanr does not work when e1=Table and e2=RowInstance
        # so we replace e1 and e2 and then transpose the result
        transpose = False
        if isinstance(e1, data.Table) and isinstance(e2, data.RowInstance):
            x1, x2 = x2, x1
            slc = len(e1) if x1.ndim > 1 else 1
            transpose = True
        rho, _ = stats.spearmanr(x1, x2, axis=axis)
        dist = (1. - rho) / 2.
        if isinstance(dist, np.ndarray):
            dist = dist[:slc, slc:]
            if transpose:
                dist = dist.T
            dist = DistMatrix(dist, e1, e2)
        return dist


class SpearmanRAbsolute():
    """Spearman's absolute rank correlation coefficient."""
    def __call__(self, e1, e2=None, axis=1):
        """
        Return absolute Spearman's dissimilarity between e1 and e2,
        i.e.

        .. math:: (1 - abs(r))/2

        where r is Spearman's correlation coefficient.
        """
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else x1
        if x1.ndim == 1 or x2.ndim == 1:
            axis = 0
            slc = len(x1) if x1.ndim > 1 else 1
        else:
            slc = len(e1) if axis == 1 else len(e1.domain.attributes)
        # stats.spearmanr does not work when e1=Table and e2=RowInstance
        # so we replace e1 and e2 and then transpose the result
        transpose = False
        if isinstance(e1, data.Table) and isinstance(e2, data.RowInstance):
            x1, x2 = x2, x1
            slc = len(e1) if x1.ndim > 1 else 1
            transpose = True
        rho, _ = stats.spearmanr(x1, x2, axis=axis)
        dist = (1. - np.abs(rho)) / 2.
        if isinstance(dist, np.ndarray):
            dist = dist[:slc, slc:]
            if transpose:
                dist = dist.T
            dist = DistMatrix(dist, e1, e2)
        return dist


class PearsonR():
    """Pearson's rank correlation coefficient."""
    def __call__(self, e1, e2=None, axis=1):
        """
        :param e1: data instances.
        :param e2: data instances.

        Returns Pearson's dissimilarity between e1 and e2,
        i.e.

        .. math:: (1-r)/2

        where r is Pearson's rank coefficient.
        """
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else x1
        if axis == 0:
            x1 = x1.T
            x2 = x2.T
        if x1.ndim == 1:
            x1 = list([x1])
        if x2.ndim == 1:
            x2 = list([x2])
        rho = np.array([[stats.pearsonr(i, j)[0] for j in x2] for i in x1])
        dist = (1. - rho) / 2.
        if dist.size == 1:
            dist = dist[0][0]
        else:
            dist = DistMatrix(dist, e1, e2)
        return dist


class PearsonRAbsolute():
    """Pearson's absolute rank correlation coefficient."""
    def __call__(self, e1, e2=None, axis=1):
        """
        :param e1: data instances.
        :param e2: data instances.

        Returns absolute Pearson's dissimilarity between e1 and e2,
        i.e.

        .. math:: (1-abs(r))/2

        where r is Pearson's rank coefficient.
        """
        x1 = e1.x if isinstance(e1, data.RowInstance) else e1.X
        x2 = e2.x if isinstance(e2, data.RowInstance) else e2.X if e2 is not None else x1
        if axis == 0:
            x1 = x1.T
            x2 = x2.T
        if x1.ndim == 1:
            x1 = list([x1])
        if x2.ndim == 1:
            x2 = list([x2])
        rho = np.array([[stats.pearsonr(i, j)[0] for j in x2] for i in x1])
        dist = (1. - np.abs(rho)) / 2.
        if dist.size == 1:
            dist = dist[0][0]
        else:
            dist = DistMatrix(dist, e1, e2)
        return dist