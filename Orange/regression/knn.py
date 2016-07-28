import sklearn.neighbors as skl_neighbors
from Orange.base import KNNBase
from Orange.regression import SklLearner

__all__ = ["KNNRegressionLearner"]


class KNNRegressionLearner(KNNBase, SklLearner):
    __wraps__ = skl_neighbors.KNeighborsRegressor
    name = 'knn regression'
