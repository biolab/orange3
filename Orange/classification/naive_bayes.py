import numpy as np

from Orange.classification import Fitter, Model
from Orange.data import Instance, Storage, Table, DiscreteVariable
from Orange.statistics import contingency


class BayesLearner(Fitter):
    def fit_storage(self, table):
        if not isinstance(table, Storage):
            raise TypeError("Data is not a subclass of Orange.data.Storage.")
        if not all(isinstance(var, DiscreteVariable)
                   for var in table.domain.variables):
            raise NotImplementedError("Only discrete variables are supported.")

        cont = contingency.get_contingencies(table)
        class_freq = np.diag(
            contingency.get_contingency(table, table.domain.class_var))
        return BayesClassifier(cont, class_freq, table.domain)


class BayesClassifier(Model):
    def __init__(self, cont, class_freq, domain):
        super().__init__(domain)
        self.cont = cont
        self.class_freq = class_freq

    def predict_storage(self, data):
        if isinstance(data, Instance):
            data = [data]
        ncv = len(self.domain.class_var.values)
        probs = np.zeros((len(data), ncv))
        for i, ins in enumerate(data):
            for c in range(ncv):
                py = (1 + self.class_freq[c]) / (ncv + sum(self.class_freq))
                log_prob = np.log(py)
                for ai, a in enumerate(self.domain.attributes):
                    relevant = 1 + self.cont[ai][c][a.to_val(ins[a])]
                    total = len(a.values) + self.class_freq[c]
                    log_prob += np.log(relevant / total)
                probs[i, c] = log_prob
        np.exp(probs, out=probs)
        probs /= probs.sum(axis=1)[:, None]
        values = probs.argmax(axis=1)
        return values, probs
