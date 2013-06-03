import numpy as np
from Orange.statistics import contingency


class Score:
    def __new__(cls, *args):
        self = super().__new__(cls)
        if args:
            return self(*args)
        else:
            return self

    def __call__(self, feature, data):
        cont = contingency.Discrete(data, feature)
        return self.from_contingency(cont)


def _entropy(D):
    """Entropy of class-distribution matrix"""
    P = D / np.sum(D, axis=0)
    PC = np.clip(P, 1e-15, 1)
    return np.sum(np.sum(- P * np.log2(PC), axis=0) * np.sum(D, axis=0) / np.sum(D))


def _gini(D):
    """Gini index of class-distribution matrix"""
    P = D / np.sum(D, axis=0)
    return sum((np.ones(1 if len(D.shape) == 1 else D.shape[1]) - np.sum(np.square(P), axis=0)) \
               * 0.5 * np.sum(D, axis=0) / np.sum(D))


class InfoGain(Score):
    """
    Information gain of a feature in class-labeled data set.

    :param feature: feature id
    :param data: data set
    :type data: Orange.data.Table
    :return: float
    """
    def from_contingency(self, cont):
        h_class = _entropy(np.sum(cont, axis=1))
        h_residual = _entropy(cont)
        return h_class - h_residual


class GainRatio(Score):
    """
    Gain ratio score of a feature in class-labeled data set.

    :param feature: feature id
    :param data: data set
    :type data: Orange.data.Table
    :return: float
    """
    def from_contingency(self, cont):
        h_class = _entropy(np.sum(cont, axis=1))
        h_residual = _entropy(cont)
        h_attribute = _entropy(np.sum(cont, axis=0))
        return (h_class - h_residual) / h_attribute


class Gini(Score):
    """
    Gini score of a feature in class-labeled data set.

    :param feature: feature id
    :param data: data set
    :type data: Orange.data.Table
    :return: float
    """
    def from_contingency(self, cont):
        return _gini(np.sum(cont, axis=1)) - _gini(cont)
