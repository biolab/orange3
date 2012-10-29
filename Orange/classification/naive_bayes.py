from Orange import classification
from Orange.data import Table
from sklearn.naive_bayes import GaussianNB
import numpy as np

class BayesLearner(classification.Fitter):
    def __call__(self, data):
        assert isinstance(data, Table)
        clf = GaussianNB()
        clf.fit(data.X, data.Y[:,0])
        return BayesClassifier(data.domain, clf)


class BayesClassifier(classification.Model):
    def __init__(self, domain, clf):
        super().__init__(domain)
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
