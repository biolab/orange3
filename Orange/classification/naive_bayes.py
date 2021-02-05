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
            raise NotImplementedError("Only categorical variables are "
                                      "supported.")

        cont = contingency.get_contingencies(table)
        class_freq = np.array(np.diag(
            contingency.get_contingency(table, table.domain.class_var)))
        nclss = (class_freq != 0).sum()
        if not nclss:
            raise ValueError("Data has no defined target values.")

        # Laplacian smoothing considers only classes that appear in the data,
        # in part to avoid cases where the probabilities are affected by empty
        # (or completely spurious) classes that appear because of Orange's reuse
        # of variables. See GH-2943.
        # The corresponding elements of class_probs are set to zero only after
        # mock non-zero values are used in computation of log_cont_prob to
        # prevent division by zero.
        class_prob = (class_freq + 1) / (np.sum(class_freq) + nclss)
        log_cont_prob = [np.log(
            (np.array(c) + 1) / (np.sum(np.array(c), axis=0)[None, :] + nclss)
            / class_prob[:, None])
                         for c in cont]
        class_prob[class_freq == 0] = 0
        return NaiveBayesModel(log_cont_prob, class_prob, table.domain)


class NaiveBayesModel(Model):
    def __init__(self, log_cont_prob, class_prob, domain):
        super().__init__(domain)
        self.log_cont_prob = log_cont_prob
        self.class_prob = class_prob

    def predict_storage(self, data):
        if isinstance(data, Instance):
            data = Table.from_numpy(None, np.atleast_2d(data.x))
        if type(data) is Table:  # pylint: disable=unidiomatic-typecheck
            return self.predict(data.X)

        if not len(data) or not len(data[0]):
            probs = np.tile(self.class_prob, (len(data), 1))
        else:
            isnan = np.isnan
            zeros = np.zeros_like(self.class_prob)
            probs = self.class_prob * np.exp(np.array([
                zeros if isnan(ins.x).all() else
                sum(attr_prob[:, int(attr_val)]
                    for attr_val, attr_prob in zip(ins, self.log_cont_prob)
                    if not isnan(attr_val))
                for ins in data]))
        probs /= probs.sum(axis=1)[:, None]
        values = probs.argmax(axis=1)
        return values, probs

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.class_prob.shape[0]))
        if self.log_cont_prob is not None:
            if sp.issparse(X):
                self._sparse_probs(X, probs)
            else:
                self._dense_probs(X, probs)
        np.exp(probs, probs)
        probs *= self.class_prob
        probs /= probs.sum(axis=1)[:, None]
        values = probs.argmax(axis=1)
        return values, probs

    def _dense_probs(self, data, probs):
        zeros = np.zeros((1, probs.shape[1]))
        for col, attr_prob in zip(data.T, self.log_cont_prob):
            col = col.copy()
            col[np.isnan(col)] = attr_prob.shape[1] - 1
            col = col.astype(int)
            probs0 = np.vstack((attr_prob.T, zeros))
            probs += probs0[col]
        return probs

    def _sparse_probs(self, data, probs):
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
