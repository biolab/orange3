import numpy as np
from sklearn import feature_selection as skl_fss
from Orange.misc.wrapper_meta import WrapperMeta

from Orange.statistics import contingency, distribution
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from Orange.preprocess.preprocess import Discretize, Continuize

__all__ = ["Chi2", "ANOVA", "UnivariateLinearRegression",
           "InfoGain", "GainRatio", "Gini"]


class Scorer:
    feature_type = None
    class_type = None
    preprocessors = ()

    def __new__(cls, *args):
        self = super().__new__(cls)
        self.preprocessors = list(self.preprocessors)
        if args:
            return self(*args)
        else:
            return self

    def __call__(self, data, feature=None):
        if not data.domain.class_var:
            raise ValueError("Data with class labels required.")
        if not isinstance(data.domain.class_var, self.class_type):
            raise ValueError("Scoring method %s requires a class variable of type %s." %
                             (type(self).__name__, self.class_type.__name__))

        if feature is not None:
            f = data.domain[feature]
            data = data.from_table(Domain([f], data.domain.class_vars), data)

        for pp in self.preprocessors:
            data = pp(data)

        if any(not isinstance(a, self.feature_type)
               for a in data.domain.attributes):
            raise ValueError('Only %ss are supported' % self.feature_type)

        return self.score_data(data, feature)

    def score_data(self, data, feature):
        raise NotImplementedError


class SklScorer(Scorer, metaclass=WrapperMeta):
    def score_data(self, data, feature):
        score = self.score(data.X, data.Y)
        if feature is not None:
            return score[0]
        return score


class Chi2(SklScorer):
    """
    A wrapper for `${sklname}`. The following is the documentation
    from `scikit-learn <http://scikit-learn.org>`_.

    ${skldoc}
    """
    __wraps__ = skl_fss.chi2
    feature_type = DiscreteVariable
    class_type = DiscreteVariable
    preprocessors = [
        Discretize()
    ]

    def score(self, X, y):
        f, p = skl_fss.chi2(X, y)
        return f


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
        return f


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
        return f


class ClassificationScorer(Scorer):
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
    preprocessors = [
        Discretize()
    ]

    def score_data(self, data, feature):
        instances_with_class = \
            np.sum(distribution.Discrete(data, data.domain.class_var))

        def score_from_contingency(f):
            cont = contingency.Discrete(data, f)
            return self.from_contingency(
                cont, 1. - np.sum(cont.unknowns)/instances_with_class)

        scores = [score_from_contingency(f) for f in data.domain.attributes]

        if feature is not None:
            return scores[0]
        return scores


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
