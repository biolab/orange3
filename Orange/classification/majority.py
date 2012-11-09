import numpy as np
import bottleneck as bn
from Orange import classification

class MajorityLearner(classification.Fitter):
    def fit(self, X, Y, W):
        assert np.issubdtype(Y.dtype, int)
        n_values = len(self.domain.class_var.values)
        # dist, _ = bn.bincount(Y, n_values-1, W)
        dist = np.bincount(Y.reshape(-1), minlength=n_values)
        N = np.sum(dist)
        if N > 0:
            dist = dist / N
        else:
            dist.fill(1/len(dist))
        return ConstantClassifier(np.argmax(dist), dist)


class ConstantClassifier(classification.Model):
    def __init__(self, value=None, probs=None):
        self.value = value
        self.probs = probs

    def predict(self, X):
        return np.tile(self.value, len(X)), np.tile(self.probs, (len(X), 1))
