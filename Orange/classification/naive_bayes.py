from Orange import classification
from Orange.data import Table
from sklearn.naive_bayes import GaussianNB
import numpy as np

class BayesLearner(classification.Fitter):
    def __call__(self, data):
        assert isinstance(data, Table)
        clf = GaussianNB()
        clf.fit(data.X, data.Y[:,0])
        class_vals = np.unique(data.Y[:,0])
        return BayesClassifier(data.domain, clf, class_vals)


class BayesClassifier(classification.Model):
    def __init__(self, domain, clf, class_vals):
        super().__init__(domain)
        self.clf = clf
        self.used_vals = class_vals

    def predict(self, X):
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)

        # expand probability predictions with zero columns for class values which are not present
        if len(self.domain.class_var.values) != len(prob):
            prob_ext = np.ndarray((len(value), len(self.domain.class_var.values)))
            i = 0
            for ci, cv in enumerate(self.domain.class_var.values):
                if i < len(self.used_vals) and cv == self.used_vals[i]:
                    prob_ext[:,ci] = prob[:,i]
                    i += 1
                else:
                    prob_ext[:,ci] = np.zeros(len(value))
            prob = prob_ext

        return value, prob
