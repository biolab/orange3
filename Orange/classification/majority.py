from numpy import tile, array

from Orange import classification
from Orange.statistics import distribution


class MajorityLearner(classification.Fitter):
    def fit_storage(self, data):
        dist = distribution.get_distribution(data, data.domain.class_var)
        N = dist.sum()
        if N > 0:
            dist /= N
        else:
            dist.fill(1 / len(dist))
        return ConstantClassifier(dist=dist)

class ConstantClassifier(classification.Model):
    def __init__(self, dist):
        self.dist = array(dist)

    def predict(self, X):
        return tile(self.dist, (len(X), 1))
