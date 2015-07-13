from Orange.base import SklLearner, SklModel, Model
import numpy as np


class DummyLearner(SklLearner):
    def fit(self, X, Y, W):
        rows = Y.shape[0]
        value = Y[np.random.randint(0, rows)]
        class_vals = np.unique(Y)
        prob = (class_vals == value) * 0.8 + 0.1
        return DummyPredictor(value, prob)


class DummyPredictor(SklModel):
    def __init__(self, value, prob):
        self.value = value
        self.prob = prob
        self.ret = Model.ValueProbs

    def predict(self, X):
        rows = X.shape[0]
        value = np.tile(self.value, rows)
        probs = np.tile(self.prob, (rows, 1))
        if self.ret == Model.Value:
            return value
        elif self.ret == Model.Value:
            return probs
        else:
            return value, probs

    def __call__(self, data, ret=Model.Value):
        prediction = super().__call__(data, ret=ret)

        if ret == Model.Value:
            return prediction

        if ret == Model.Probs:
            probs = prediction
        else:  # ret == Model.ValueProbs
            value, probs = prediction

        # Expand probability predictions for class values which are not present
        if ret != self.Value:
            n_class = len(self.domain.class_vars)
            max_values = max(len(cv.values) for cv in self.domain.class_vars)
            if max_values != probs.shape[-1]:
                if not self.supports_multiclass:
                    probs = probs[:, np.newaxis, :]
                probs_ext = np.zeros((len(probs), n_class, max_values))
                for c in range(n_class):
                    i = 0
                    class_values = len(self.domain.class_vars[c].values)
                    for cv in range(class_values):
                        if (i < len(self.used_vals[c]) and
                                    cv == self.used_vals[c][i]):
                            probs_ext[:, c, cv] = probs[:, c, i]
                            i += 1
                if self.supports_multiclass:
                    probs = probs_ext
                else:
                    probs = probs_ext[:, 0, :]

        if ret == Model.Probs:
            return probs
        else:  # ret == Model.ValueProbs
            return value, probs


class DummyMulticlassLearner(SklLearner):
    supports_multiclass = True

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


class DummyMulticlassPredictor(SklModel):
    def __init__(self, value, prob):
        self.value = value
        self.prob = prob
        self.ret = Model.ValueProbs

    def predict(self, X):
        rows = X.shape[0]
        value = np.tile(self.value, (rows, 1))
        probs = np.tile(self.prob, (rows, 1, 1))
        if self.ret == Model.Value:
            return value
        elif self.ret == Model.Value:
            return probs
        else:
            return value, probs

    def __call__(self, data, ret=Model.Value):
        prediction = super().__call__(data, ret=ret)

        if ret == Model.Value:
            return prediction

        if ret == Model.Probs:
            probs = prediction
        else:  # ret == Model.ValueProbs
            value, probs = prediction

        # Expand probability predictions for class values which are not present
        if ret != self.Value:
            n_class = len(self.domain.class_vars)
            max_values = max(len(cv.values) for cv in self.domain.class_vars)
            if max_values != probs.shape[-1]:
                if not self.supports_multiclass:
                    probs = probs[:, np.newaxis, :]
                probs_ext = np.zeros((len(probs), n_class, max_values))
                for c in range(n_class):
                    i = 0
                    class_values = len(self.domain.class_vars[c].values)
                    for cv in range(class_values):
                        if (i < len(self.used_vals[c]) and
                                    cv == self.used_vals[c][i]):
                            probs_ext[:, c, cv] = probs[:, c, i]
                            i += 1
                if self.supports_multiclass:
                    probs = probs_ext
                else:
                    probs = probs_ext[:, 0, :]

        if ret == Model.Probs:
            return probs
        else:  # ret == Model.ValueProbs
            return value, probs
