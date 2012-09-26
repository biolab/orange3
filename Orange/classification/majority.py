import numpy as np
from Orange import classification

class MajorityLearner(classification.Learner):
    def __call__(self, data):
        if len(data.domain.class_vars) > 1:
            raise NotImplementedError(
                "Majority learner does not support multiple classes")

        y = data.Y
        w = data.W if data.W.shape[-1] != 0 else None
        assert np.issubdtype(y.dtype, int)

        n_values = data.domain.class_var.values()
        dist = np.bincount(y, w, minlength=n_values).astype(float)[:n_values]
        N = sum(dist)
        if N > 0:
            dist /= sum(dist)
        else:
            dist.fill(1/len(dist))
        return ConstantClassifier(data.domain, dist=dist)


class ConstantClassifier(classification.Classifier):
    def __init__(self, domain, value=None, dist=None):
        super().__init__(self, domain)
        if value is None:
            self.value = np.argmax(dist)
        else:
            self.value = value
        if self.dist is None:
            self.dist = np.zeros(len(domain.class_var.values), float)
            self.dist[self.value] = 1
        else:
            self.dist = dist

    def predict_class(self, _):
        return self.value

    def predict_dist(self, _):
        return self.dist

    def predict_inst(self, x):
        return self.value, self.dist

    def predict_table_class(self, data):
        return np.tile(self.value, len(data))

    def predict_table_prob(self, data):
        return np.tile(self.dist, (len(data), 1))
