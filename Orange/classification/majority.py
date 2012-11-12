import numpy as np
import bottleneck as bn
from Orange import classification

class MajorityLearner(classification.Fitter):
    I_can_has_multiclass = True

    def fit(self, X, Y, W):
        assert np.issubdtype(Y.dtype, int)

        weights = (W is not None and W.size == Y.size)
        n_class_vars = len(self.domain.class_vars)
        n_values = max(max(cv.values)+1 for cv in self.domain.class_vars)
        value = np.empty(n_class_vars, dtype=int)
        probs = np.empty((n_class_vars, n_values), dtype=float)
        for i, y in enumerate(Y.T):
            # dist, _ = bn.bincount(Y, n_values-1, W)
            if weights: dist = np.bincount(y, weights=W.T[i], minlength=n_values)
            else: dist = np.bincount(y, minlength=n_values)
            N = np.sum(dist)
            if N > 0: dist = dist / N
            else: dist.fill(1/len(dist))
            value[i] = np.argmax(dist)
            probs[i] = dist
        return ConstantClassifier(value, probs)


class ConstantClassifier(classification.Model):
    def __init__(self, value=None, probs=None):
        self.value = value
        self.probs = probs
        self.multitarget = len(value) > 1
        self.ret = classification.Model.ValueProbs

    def predict(self, X):
        if self.multitarget:
            pred_value = np.tile(self.value, (len(X),1))
            pred_probs = np.tile(self.probs, (len(X),1,1))
        else:
            pred_value = np.tile(self.value[0], len(X))
            pred_probs = np.tile(self.probs[0], (len(X),1))

        if self.ret == classification.Model.Value: return pred_value
        elif self.ret == classification.Model.Probs: return pred_probs
        else: return pred_value, pred_probs
