import numpy as np
import scipy.sparse as sp

from Orange.classification import Learner, Model
from Orange.data import Instance, Storage, Table
from Orange.statistics import contingency
from Orange.preprocess import Discretize, RemoveNaNColumns

__all__ = ["NaiveBayesLearner"]


class NaiveBayesLearner(Learner):
    """
    Naive Bayes classifier. Works only with discrete attributes. By default,
    continuous attributes are discretized.

    Parameters
    ----------
    preprocessors : list, optional (default="[Orange.preprocess.Discretize]")
        An ordered list of preprocessors applied to data before training
        or testing.
    """
    preprocessors = [RemoveNaNColumns(), Discretize()]
    name = 'naive bayes'

    def fit_storage(self, table):
        if not isinstance(table, Storage):
            raise TypeError("Data is not a subclass of Orange.data.Storage.")
        if not all(var.is_discrete
                   for var in table.domain.variables):
            raise NotImplementedError("Only discrete variables are supported.")

        cont = contingency.get_contingencies(table)
        class_freq = np.array(np.diag(
            contingency.get_contingency(table, table.domain.class_var)))
        class_prob = (class_freq + 1) / (np.sum(class_freq) + len(class_freq))
        log_cont_prob = [np.log(
            (np.array(c) + 1) / (np.sum(np.array(c), axis=0)[None, :] +
                                 c.shape[0]) / class_prob[:, None])
                         for c in cont]
        return NaiveBayesModel(log_cont_prob, class_prob, table.domain)


class NaiveBayesModel(Model):
    def __init__(self, log_cont_prob, class_prob, domain):
        super().__init__(domain)
        self.log_cont_prob = log_cont_prob
        self.class_prob = class_prob

    def predict_storage(self, data):
        if type(data) is Table:  # pylint: disable=unidiomatic-typecheck
            return self.predict(data.X)
        if isinstance(data, Instance):
            return Table(np.atleast_2d(data.x))

        if not len(data) or not len(data[0]):
            probs = np.tile(self.class_prob, (len(data), 1))
        else:
            isnan = np.isnan
            zeros = np.zeros_like(self.class_prob)
            probs = np.atleast_2d(np.exp(
                np.log(self.class_prob) +
                np.array([
                    zeros if isnan(ins.x).all() else
                    sum(attr_prob[:, int(attr_val)]
                        for attr_val, attr_prob in zip(ins, self.log_cont_prob)
                        if not isnan(attr_val))
                    for ins in data])))
        probs /= probs.sum(axis=1)[:, None]
        values = probs.argmax(axis=1)
        return values, probs

    def predict(self, X):
        if not self.log_cont_prob:
            probs = self._priors(X)
        elif sp.issparse(X):
            probs = self._sparse_probs(X)
        else:
            probs = self._dense_probs(X)
        probs = np.exp(probs)
        probs /= probs.sum(axis=1)[:, None]
        values = probs.argmax(axis=1)
        return values, probs

    def _priors(self, data):
        return np.tile(np.log(self.class_prob), (data.shape[0], 1))

    def _dense_probs(self, data):
        probs = self._priors(data)
        zeros = np.zeros((1, probs.shape[1]))
        for col, attr_prob in zip(data.T, self.log_cont_prob):
            col = col.copy()
            col[np.isnan(col)] = attr_prob.shape[1] - 1
            col = col.astype(int)
            probs0 = np.vstack((attr_prob.T, zeros))
            probs += probs0[col]
        return probs

    def _sparse_probs(self, data):
        probs = self._priors(data)

        n_vals = max(p.shape[1] for p in self.log_cont_prob) + 1
        log_prob = np.zeros((len(self.log_cont_prob),
                             n_vals,
                             self.log_cont_prob[0].shape[0]))
        for i, p in enumerate(self.log_cont_prob):
            p0 = p.T[0].copy()
            probs[:] += p0
            log_prob[i, :p.shape[1]] = p.T - p0

        dat = data.data.copy()
        dat[np.isnan(dat)] = n_vals - 1
        dat = dat.astype(int)

        if sp.isspmatrix_csr(data):
            for row, start, end in zip(probs, data.indptr, data.indptr[1:]):
                row += log_prob[data.indices[start:end],
                                dat[start:end]].sum(axis=0)
        else:
            csc = data.tocsc()
            for start, end, attr_prob in zip(csc.indptr, csc.indptr[1:],
                                             log_prob):
                probs[csc.indices[start:end]] += attr_prob[dat[start:end]]

        return probs


NaiveBayesLearner.__returns__ = NaiveBayesModel
