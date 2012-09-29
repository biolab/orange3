import numpy as np
import bottleneck as bn
from Orange import classification

class MajorityLearner(classification.Fitter):
    def __call__(self, data):
        if len(data.domain.class_vars) > 1:
            raise NotImplementedError(
                "Majority learner does not support multiple classes")

        y = data.Y
        w = data.W if data.has_weights else None
        assert np.issubdtype(y.dtype, int)

        n_values = data.domain.class_var.values()
        dist, _ = bn.bincount(y, n_values-1, w)
        N = np.sum(dist)
        if N > 0:
            dist /= N
        else:
            dist.fill(1/len(dist))
        return ConstantClassifier(data.domain, probs=dist)


class ConstantClassifier(classification.Model):
    def __init__(self, domain, value=None, probs=None):
        super().__init__(self, domain)
        self.value = value
        self.probs = probs

    def predict(self, X):
        r = []
        if self.value is not None:
            r.append(np.tile(self.value, len(X)))
        if self.probs is not None:
            r.append(np.tile(self.probs, (len(X), 1)))
        return r
