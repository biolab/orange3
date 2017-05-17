import sklearn.neighbors as skl_neighbors
from Orange.base import KNNBase
from Orange.classification import SklLearner

__all__ = ["KNNLearner"]


class KNNLearner(KNNBase, SklLearner):
    __wraps__ = skl_neighbors.KNeighborsClassifier
