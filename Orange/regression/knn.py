import sklearn.neighbors as skl_neighbors
from Orange.regression import SklLearner

__all__ = ["KNNRegressionLearner"]

# How knn working
# http://www.saedsayad.com/k_nearest_neighbors.htm

class KNNRegressionLearner(SklLearner):
    __wraps__ = skl_neighbors.KNeighborsRegressor
    name = 'knn regression'

    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform",
                 algorithm='auto',
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
