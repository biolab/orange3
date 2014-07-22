import numpy as np
from sklearn.naive_bayes import GaussianNB

from Orange import classification
from Orange.data import Storage
from Orange.statistics import contingency


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
        if not isinstance(table, Storage):
            raise TypeError("Data is not a subclass of Orange.data.Storage.")
        if not all(var.var_type == var.VarTypes.Discrete
                   for var in table.domain.variables):
            raise NotImplementedError("Only discrete variables are supported.")

        cont = contingency.get_contingencies(table)
        class_freq = np.diag(
            contingency.get_contingency(table, table.domain.class_var))
        return BayesStorageClassifier(cont, class_freq, table.domain)


class BayesStorageClassifier(classification.Model):
    def __init__(self, cont, class_freq, domain):
        super().__init__(domain)
        self.cont = cont
        self.class_freq = class_freq

    def predict_storage(self, data):
        if isinstance(data, Storage):
            return np.array([self.predict_storage(ins) for ins in data])
        ncv = len(self.domain.class_var.values)
        max_log_prob = None
        value = None
        for c in range(ncv):
            py = (1 + self.class_freq[c]) / (ncv + sum(self.class_freq))
            log_prob = np.log(py)
            for i, a in enumerate(self.domain.attributes):
                relevant = 1 + self.cont[i][c][a.to_val(data[a])]
                total = len(a.values) + self.class_freq[c]
                log_prob += np.log(relevant / total)
            if max_log_prob is None or log_prob > max_log_prob:
                max_log_prob = log_prob
                value = c
        return value
