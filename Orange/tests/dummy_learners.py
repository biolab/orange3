import numpy as np

from Orange.classification import SklLearner, SklModel
from Orange.classification.base_classification import SklModelClassification


class DummyLearner(SklLearner):
    def fit(self, X, Y, W):
        rows = Y.shape[0]
        value = Y[np.random.randint(0, rows)]
        class_vals = np.unique(Y)
        prob = (class_vals == value) * 0.8 + 0.1
        return DummyPredictor(value, prob)


class DummySklModel:
    def __init__(self, value, prob):
        self.value = value
        self.prob = prob

    def predict(self, X):
        return np.tile(self.value, len(X))

    def predict_proba(self, X):
        return np.tile(self.prob, (len(X), 1))


class DummyPredictor(SklModelClassification):
    def __init__(self, value, prob):
        SklModel.__init__(self, DummySklModel(value, prob))


class DummyMulticlassLearner(SklLearner):
    supports_multiclass = True

    def incompatibility_reason(self, domain):
        reason = 'Not all class variables are discrete'
        return None if all(c.is_discrete for c in domain.class_vars) else reason

    def fit(self, X, Y, W):
        rows, class_vars = Y.shape
        rid = np.random.randint(0, rows)
        value = [Y[rid, cid] for cid in range(Y.shape[1])]
        used_vals = [np.unique(y) for y in Y.T]
        max_vals = max(len(np.unique(y)) for y in Y.T)
        prob = np.zeros((class_vars, max_vals))
        for c in range(class_vars):
            class_prob = (used_vals[c] == value[c]) * 0.8 + 0.1
            prob[c, :] = np.hstack((class_prob,
                                    np.zeros(max_vals - len(class_prob))))
        return DummyMulticlassPredictor(value, prob)


class DummySklMulticlassModel:
    def __init__(self, value, prob):
        self.value = value
        self.prob = prob

    def predict(self, X):
        return np.tile(self.value, (len(X), 1))

    def predict_proba(self, X):
        return np.tile(self.prob, (len(X), 1, 1))


class DummyMulticlassPredictor(SklModelClassification):
    def __init__(self, value, prob):
        SklModel.__init__(self, DummySklMulticlassModel(value, prob))
