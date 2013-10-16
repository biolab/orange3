from Orange import classification
from sklearn.naive_bayes import GaussianNB
from Orange.statistics import contingency
import numpy as np
from math import log, e


class BayesLearner(classification.Fitter):
    def fit(self, X, Y, W):
        clf = GaussianNB()
        return BayesClassifier(clf.fit(X, Y.reshape(-1)))


class BayesClassifier(classification.Model):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob


class BayesStorageLearner(classification.Fitter):
    def fit_storage(self, table):
        cont = contingency.get_contingencies(table)
        class_freq = np.diag(
            contingency.get_contingency(table, table.domain.class_var))
        return BayesStorageClassifier(cont, class_freq, table.domain)


class BayesStorageClassifier(classification.Model):
    def __init__(self, cont, class_freq, domain):
        self.cont = cont
        self.class_freq = class_freq
        self.domain = domain

    def predict(self, X):
        ncv = len(self.domain.class_var.values)
        values = np.zeros(len(X))
        for i, x in enumerate(X):
            max_log_prob = None
            value = None
            for c in range(ncv):
                py = (1 + self.class_freq[c]) / (ncv + sum(self.class_freq))
                log_prob = log(py)
                for a in range(len(self.domain.attributes)):
                    a_values, a_counts = self.cont[a][c]
                    pos = np.where(a_values == x[a])[0]
                    relevant = 1
                    if len(pos) == 1:
                        relevant += a_counts[pos[0]]
                    total = len(a_values) + self.class_freq[c]
                    log_prob += log(relevant / total)
                if max_log_prob is None or log_prob > max_log_prob:
                    max_log_prob = log_prob
                    value = c
            values[i] = value
        return values
