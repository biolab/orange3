from numpy import tile, array

from Orange import classification, data
from Orange.statistics import distribution


class MajorityFitter(classification.Fitter):
    def fit_storage(self, dat):
        if not isinstance(dat.domain.class_var, data.DiscreteVariable):
            raise ValueError("classification.MajorityFitter expects a domain with a "
                             "(single) discrete variable")
        dist = distribution.get_distribution(dat, dat.domain.class_var)
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

    def __str__(self):
        return 'ConstantClassifier {}'.format(self.dist)
