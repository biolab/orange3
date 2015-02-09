import numpy as np
from sklearn.feature_selection import chi2 as skl_chi2
from sklearn.feature_selection import f_classif as skl_f_classif
from sklearn.feature_selection import f_regression as skl_f_regression

from Orange.statistics import contingency, distribution
from Orange.data.variable import DiscreteVariable, ContinuousVariable


class SklScorer:

    def __call__(self, feature, data):
        if not data.domain.class_var:
            raise ValueError("Data with class labels required.")
        X = data.X[:, [data.domain.index(feature)]]
        y = data.Y.flatten()
        return self.score(X, y)


class Chi2(SklScorer):

    def __call__(self, feature, data):
        if not isinstance(data.domain[feature], DiscreteVariable):
            raise ValueError("Discrete feature required.")
        if not isinstance(data.domain.class_var, DiscreteVariable):
            raise ValueError("Data with discrete class labels required.")
        return super().__call__(feature, data)

    def score(self, X, y):
        f, p = skl_chi2(X, y)
        return f[0]


class ANOVA(SklScorer):

    def __call__(self, feature, data):
        if not isinstance(data.domain[feature], ContinuousVariable):
            raise ValueError("Continuous feature required.")
        if not isinstance(data.domain.class_var, DiscreteVariable):
            raise ValueError("Data with discrete class labels required.")
        return super().__call__(feature, data)

    def score(self, X, y):
        f, p = skl_f_classif(X, y)
        return f[0]


class UnivariateLinearRegression(SklScorer):

    def __call__(self, feature, data):
        if not isinstance(data.domain[feature], ContinuousVariable):
            raise ValueError("Continuous feature required.")
        if not isinstance(data.domain.class_var, ContinuousVariable):
            raise ValueError("Regression problem required.")
        return super().__call__(feature, data)

    def score(self, X, y):
        f, p = skl_f_regression(X, y)
        return f[0]


class ClassificationScorer:

    def __call__(self, feature, data):
        if not data.domain.class_var:
            raise ValueError("Data with class labels required.")
        elif not isinstance(data.domain.class_var, DiscreteVariable):
            raise ValueError("Data with discrete class labels required.")
        cont = contingency.Discrete(data, feature)
        instances_with_class = np.sum(distribution.Discrete(data, data.domain.class_var))
        return self.from_contingency(cont, 1. - np.sum(cont.unknowns)/instances_with_class)


def _entropy(D):
    """Entropy of class-distribution matrix"""
    P = D / np.sum(D, axis=0)
    PC = np.clip(P, 1e-15, 1)
    return np.sum(np.sum(- P * np.log2(PC), axis=0) * np.sum(D, axis=0) / np.sum(D))


def _gini(D):
    """Gini index of class-distribution matrix"""
    P = D / np.sum(D, axis=0)
    return sum((np.ones(1 if len(D.shape) == 1 else D.shape[1]) - np.sum(np.square(P), axis=0))
               * 0.5 * np.sum(D, axis=0) / np.sum(D))


class InfoGain(ClassificationScorer):
    """
    Information gain of a feature in class-labeled data set.

    :param feature: feature id
    :param data: data set
    :type data: Orange.data.Table
    :return: float
    """
    def from_contingency(self, cont, nan_adjustment):
        h_class = _entropy(np.sum(cont, axis=1))
        h_residual = _entropy(cont)
        return (h_class - h_residual) * nan_adjustment


class GainRatio(ClassificationScorer):
    """
    Gain ratio score of a feature in class-labeled data set.

    :param feature: feature id
    :param data: data set
    :type data: Orange.data.Table
    :return: float
    """
    def from_contingency(self, cont, nan_adjustment):
        h_class = _entropy(np.sum(cont, axis=1))
        h_residual = _entropy(cont)
        h_attribute = _entropy(np.sum(cont, axis=0))
        return nan_adjustment * (h_class - h_residual) / h_attribute


class Gini(ClassificationScorer):
    """
    Gini score of a feature in class-labeled data set.

    :param feature: feature id
    :param data: data set
    :type data: Orange.data.Table
    :return: float
    """
    def from_contingency(self, cont, nan_adjustment):
        return (_gini(np.sum(cont, axis=1)) - _gini(cont)) * nan_adjustment
