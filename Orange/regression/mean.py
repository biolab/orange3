import numpy

from Orange import classification, data
from Orange.statistics import distribution


class MeanFitter(classification.Fitter):
    def fit_storage(self, dat):
        if not isinstance(dat.domain.class_var, data.ContinuousVariable):
            raise ValueError("regression.MeanFitter expects a domain with a "
                             "(single) continuous variable")
        dist = distribution.get_distribution(dat, dat.domain.class_var)
        return MeanModel(dist)

class MeanModel(classification.Model):
    def __init__(self, dist):
        self.dist = dist
        if dist.any():
            self.mean = self.dist.mean()
        else:
            self.mean = 0.0

    def predict(self, X):
        return numpy.zeros(X.shape[0]) + self.mean
