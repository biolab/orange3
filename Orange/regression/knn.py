import sklearn.neighbors as skl_neighbors

from Orange.classification import KNNLearner
from Orange.regression import SklLearner

__all__ = ["KNNRegressionLearner"]


class KNNRegressionLearner(SklLearner):
    __wraps__ = skl_neighbors.KNeighborsRegressor
    name = 'knn regression'

    options = KNNLearner.options
