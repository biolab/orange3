import sklearn.neighbors as skl_neighbors
from numpy import cov
from Orange.classification import SklLearner, SklModel

__all__ = ["KNNLearner"]


class KNNLearner(SklLearner):
    __wraps__ = skl_neighbors.KNeighborsClassifier

    def __init__(self, n_neighbors=5, metric="euclidean", metric_params=None,
                 weights="uniform", algorithm='auto',
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y, W):
        if self.params['metric'] == "mahalanobis":
            self.params['metric_params'] = {'V': cov(X.T)}
        skclf = self.__wraps__(**self.params)
        skclf.fit(X, Y.ravel())
        return SklModel(skclf)
