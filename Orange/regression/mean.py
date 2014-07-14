import numpy

from Orange.classification import Model, Fitter
from Orange.statistics import distribution

__all__ = ["MeanLearner", "MeanModel"]


class MeanLearner(Fitter):
    def __init__(self):
        pass

    def fit_storage(self, data):
        dist = distribution.get_distribution(data, data.domain.class_var)
        return MeanModel(dist)


class MeanModel(Model):
    def __init__(self, dist):
        self.dist = dist

    def predict(self, X):
        mean = self.dist.mean()
        return numpy.zeros(X.shape[0]) + mean
