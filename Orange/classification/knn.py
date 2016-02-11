import sklearn.neighbors as skl_neighbors
from Orange.classification import SklLearner, SklModel

__all__ = ["KNNLearner"]

class KNNClassifier(SklModel):
    pass

class KNNLearner(SklLearner):
    __wraps__ = skl_neighbors.KNeighborsClassifier
    __returns__ = KNNClassifier
    name = 'knn'

    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform",
                 algorithm='auto',
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
