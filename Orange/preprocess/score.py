import numpy as np
from sklearn import feature_selection as skl_fss

from Orange.statistics import contingency, distribution
from Orange.data.variable import DiscreteVariable, ContinuousVariable
from Orange.classification.base import WrapperMeta

__all__ = ["Chi2", "ANOVA", "UnivariateLinearRegression",
           "InfoGain", "GainRatio", "Gini"]


class SklScorer(metaclass=WrapperMeta):
    feature_type = None
    class_type = None

    def __new__(cls, *args):
        self = super().__new__(cls)
        if args:
            return self(*args)
        else:
            return self

    def __call__(self, feature, data):
        if not data.domain.class_var:
            raise ValueError("Data with class labels required.")
        if not isinstance(data.domain[feature], self.feature_type):
            raise ValueError("Scoring method %s requires a feature of type %s." %
                             (type(self).__name__, self.feature_type.__name__))
        if not isinstance(data.domain.class_var, self.class_type):
            raise ValueError("Scoring method %s requires a class variable of type %s." %
                             (type(self).__name__, self.class_type.__name__))

        X = data.X[:, [data.domain.index(feature)]]
        y = data.Y.flatten()
        return self.score(X, y)


class Chi2(SklScorer):
    """
    A wrapper for `${sklname}`. The following is the documentation
    from `scikit-learn <http://scikit-learn.org>`_.

    ${skldoc}
    """
    __wraps__ = skl_fss.chi2
    feature_type = DiscreteVariable
    class_type = DiscreteVariable

    def score(self, X, y):
        f, p = skl_fss.chi2(X, y)
        return f[0]


class ANOVA(SklScorer):
    """
    A wrapper for `${sklname}`. The following is the documentation
    from `scikit-learn <http://scikit-learn.org>`_.

    ${skldoc}
    """
    __wraps__ = skl_fss.f_classif
    feature_type = ContinuousVariable
    class_type = DiscreteVariable

    def score(self, X, y):
        f, p = skl_fss.f_classif(X, y)
        return f[0]


class UnivariateLinearRegression(SklScorer):
    """
    A wrapper for `${sklname}`. The following is the documentation
    from `scikit-learn <http://scikit-learn.org>`_.

    ${skldoc}
    """
    __wraps__ = skl_fss.f_regression
    feature_type = ContinuousVariable
    class_type = ContinuousVariable

    def score(self, X, y):
        f, p = skl_fss.f_regression(X, y)
        return f[0]


class ClassificationScorer:
    """
    Base class for feature scores in a class-labeled data set.

    Parameters
    ----------
    feature : int, string, Orange.data.Variable
        Feature id
    data : Orange.data.Table
        Data set

    Attributes
    ----------
    feature_type : Orange.data.Variable
        Required type of features.

    class_type : Orange.data.Variable
        Required type of class variable.
    """
    feature_type = DiscreteVariable
    class_type = DiscreteVariable

    def __new__(cls, *args):
        self = super().__new__(cls)
        if args:
            return self(*args)
        else:
            return self

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
    Information gain is the expected decrease of entropy. See `Wikipedia entry on information gain
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.
    """
    def from_contingency(self, cont, nan_adjustment):
        h_class = _entropy(np.sum(cont, axis=1))
        h_residual = _entropy(cont)
        return nan_adjustment * (h_class - h_residual)


class GainRatio(ClassificationScorer):
    """
    Information gain ratio is the ratio between information gain and
    the entropy of the feature's
    value distribution. The score was introduced in [Quinlan1986]_
    to alleviate overestimation for multi-valued features. See `Wikipedia entry on gain ratio
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.

    .. [Quinlan1986] J R Quinlan: Induction of Decision Trees, Machine Learning, 1986.
    """
    def from_contingency(self, cont, nan_adjustment):
        h_class = _entropy(np.sum(cont, axis=1))
        h_residual = _entropy(cont)
        h_attribute = _entropy(np.sum(cont, axis=0))
        return nan_adjustment * (h_class - h_residual) / h_attribute


class Gini(ClassificationScorer):
    """
    Gini index is the probability that two randomly chosen instances will have different
    classes. See `Wikipedia entry on gini index <http://en.wikipedia.org/wiki/Gini_coefficient>`_.
    """
    def from_contingency(self, cont, nan_adjustment):
        return (_gini(np.sum(cont, axis=1)) - _gini(cont)) * nan_adjustment
