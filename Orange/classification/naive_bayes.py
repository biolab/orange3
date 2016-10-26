import numpy as np

from Orange.classification import Learner, Model
from Orange.data import Instance, Storage
from Orange.statistics import contingency
from Orange.preprocess import Discretize

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
    name = 'naive bayes'

    preprocessors = [Discretize()]

    def fit_storage(self, table):
        if not isinstance(table, Storage):
            raise TypeError("Data is not a subclass of Orange.data.Storage.")
        if not all(var.is_discrete
                   for var in table.domain.variables):
            raise NotImplementedError("Only discrete variables are supported.")

        cont = contingency.get_contingencies(table)
        class_freq = np.array(np.diag(
            contingency.get_contingency(table, table.domain.class_var)))
        return NaiveBayesModel(cont, class_freq, table.domain)


class NaiveBayesModel(Model):
    def __init__(self, cont, class_freq, domain):
        super().__init__(domain)
        self.cont = cont
        self.class_freq = class_freq

    def predict_storage(self, data):
        if isinstance(data, Instance):
            data = [data]
        n_cls = len(self.class_freq)
        class_prob = (self.class_freq + 1) / (np.sum(self.class_freq) + n_cls)
        if len(data.domain.attributes) == 0:
            probs = np.tile(class_prob, (len(data), 1))
        else:
            log_cont_prob = [np.log(np.divide(np.array(c) + 1,
                                              self.class_freq.reshape((n_cls, 1)) +
                                              c.shape[1])) for c in self.cont]
            probs = np.exp(np.array([np.sum(attr_prob[:, int(attr_val)]
                                            for attr_val, attr_prob
                                            in zip(ins, log_cont_prob)
                                            if not np.isnan(attr_val))
                                     for ins in data]) + np.log(class_prob))
        probs /= probs.sum(axis=1)[:, None]
        values = probs.argmax(axis=1)
        return values, probs

NaiveBayesLearner.__returns__ = NaiveBayesModel
