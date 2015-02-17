import sklearn.neighbors as skl_neighbors
from Orange.classification import SklLearner

__all__ = ["KNNLearner"]


class KNNLearner(SklLearner):
    __wraps__ = skl_neighbors.KNeighborsClassifier
    name = 'knn'

    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform",
                 algorithm='auto',
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
