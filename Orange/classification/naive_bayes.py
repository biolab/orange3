from Orange import classification
from sklearn.naive_bayes import GaussianNB
import numpy as np

class BayesLearner(classification.Fitter):
    def fit(self, X, Y, W):
        clf = GaussianNB()
        return BayesClassifier(clf.fit(X, Y))


class BayesClassifier(classification.Model):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
