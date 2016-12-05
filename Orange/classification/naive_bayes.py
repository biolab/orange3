import numpy as np

from Orange.classification import Learner, Model
from Orange.data import Instance, Storage
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
        if isinstance(data, Instance):
            data = [data]
        if len(data.domain.attributes) == 0:
            probs = np.tile(self.class_prob, (len(data), 1))
        else:
            probs = np.exp(np.array([np.sum(attr_prob[:, int(attr_val)]
                                            for attr_val, attr_prob
                                            in zip(ins, self.log_cont_prob)
                                            if not np.isnan(attr_val))
                                     for ins in data]) + np.log(
                self.class_prob))
        probs /= probs.sum(axis=1)[:, None]
        values = probs.argmax(axis=1)
        return values, probs


NaiveBayesLearner.__returns__ = NaiveBayesModel
