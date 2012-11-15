from Orange import classification
import numpy as np


class DummyLearner(classification.Fitter):
    def fit(self, X, Y, W):
        value = Y[np.random.randint(0,len(Y)),0]
        prob = sum(np.hstack(Y)==value)/len(Y)
        return DummyPredictor(value, prob)

class DummyPredictor(classification.Model):
    def __init__(self, value, prob):
        self.value = value
        self.prob = prob

    def predict(self, X):
        return np.tile(self.value, len(X)), self.prob


class DummyMulticlassLearner(classification.Fitter):
    I_can_has_multiclass = True

    def fit(self, X, Y, W):
        return DummyMulticlassPredictor(Y[np.random.randint(0,len(Y))])

class DummyMulticlassPredictor(classification.Model):
    def __init__(self, values):
        self.values = values

    def predict(self, X):
        return np.tile(self.values, (len(X),1))

