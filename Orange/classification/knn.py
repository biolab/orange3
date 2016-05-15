import sklearn.neighbors as skl_neighbors
from Orange.classification import SklLearner
from Orange import options

__all__ = ["KNNLearner"]


class KNNLearner(SklLearner):
    __wraps__ = skl_neighbors.KNeighborsClassifier
    name = 'knn'

    WEIGHTS = ("uniform", "distance")
    METRICS = ("euclidean", "manhattan", "chebyshev", "mahalanobis")

    n_neighbors = options.IntegerOption(default=5, verbose_name='Number of neighbors',
                                        range=(1, 1000), step=1)
    weights = options.ChoiceOption(choices=WEIGHTS)
    metric = options.ChoiceOption(choices=METRICS)
