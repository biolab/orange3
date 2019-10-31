import re
from collections import defaultdict
from itertools import chain

import numpy as np
from sklearn import feature_selection as skl_fss

from Orange.data import Domain, Variable, DiscreteVariable, ContinuousVariable
from Orange.data.filter import HasClass
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess.preprocess import Discretize, SklImpute, RemoveNaNColumns
from Orange.statistics import contingency, distribution
from Orange.util import Reprable

__all__ = ["Chi2",
           "ANOVA",
           "UnivariateLinearRegression",
           "InfoGain",
           "GainRatio",
           "Gini",
           "ReliefF",
           "RReliefF",
           "FCBF"]


class Scorer(Reprable):
    feature_type = None
    class_type = None
    supports_sparse_data = None
    preprocessors = [HasClass()]

    @property
    def friendly_name(self):
        """Return type name with camel-case separated into words.
        Derived classes can provide a better property or a class attribute.
        """
        return re.sub("([a-z])([A-Z])",
                      lambda mo: mo.group(1) + " " + mo.group(2).lower(),
                      type(self).__name__)

    @staticmethod
    def _friendly_vartype_name(vartype):
        if vartype == DiscreteVariable:
            return "categorical"
        if vartype == ContinuousVariable:
            return "numeric"
        # Fallbacks
        name = vartype.__name__
        if name.endswith("Variable"):
            return name.lower()[:-8]
        return name

    def __call__(self, data, feature=None):
        if not data.domain.class_var:
            raise ValueError(
                "{} requires data with a target variable."
                .format(self.friendly_name))
        if not isinstance(data.domain.class_var, self.class_type):
            raise ValueError(
                "{} requires a {} target variable."
                .format(self.friendly_name,
                        self._friendly_vartype_name(self.class_type)))

        if feature is not None:
            f = data.domain[feature]
            data = data.transform(Domain([f], data.domain.class_vars))

        orig_domain = data.domain
        for pp in self.preprocessors:
            data = pp(data)

        for var in data.domain.attributes:
            if not isinstance(var, self.feature_type):
                raise ValueError(
                    "{} cannot score {} variables."
                    .format(self.friendly_name,
                            self._friendly_vartype_name(type(var))))

        if feature is not None:
            return self.score_data(data, feature)

        scores = np.full(len(orig_domain.attributes), np.nan)
        names = [a.name for a in data.domain.attributes]
        mask = np.array([a.name in names for a in orig_domain.attributes])
        if len(mask):
            scores[mask] = self.score_data(data, feature)
        return scores

    def score_data(self, data, feature):
        raise NotImplementedError


class SklScorer(Scorer, metaclass=WrapperMeta):
    supports_sparse_data = True

    preprocessors = Scorer.preprocessors + [
        SklImpute()
    ]

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
    preprocessors = SklScorer.preprocessors + [
        Discretize(remove_const=False)
    ]

    def score(self, X, y):
        f, _ = skl_fss.chi2(X, y)
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
        f, _ = skl_fss.f_classif(X, y)
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


class LearnerScorer(Scorer):

    def score(self, data):
        raise NotImplementedError

    def score_data(self, data, feature=None):

        def join_derived_features(scores):
            """ Obtain scores for original attributes.

            If a learner preprocessed the data before building a model, current scores do not
            directly correspond to the domain. For example, continuization creates multiple
            features from a discrete feature. """
            scores_grouped = defaultdict(list)
            for attr, score in zip(model_attributes, scores):
                # Go up the chain of preprocessors to obtain the original variable, but no further
                # than the data.domain, because the data is perhaphs already preprocessed.
                while not (attr in data.domain) and getattr(attr, 'compute_value', False):
                    attr = getattr(attr.compute_value, 'variable', attr)
                scores_grouped[attr].append(score)
            return [max(scores_grouped[attr])
                    if attr in scores_grouped else 0
                    for attr in data.domain.attributes]

        scores, model_attributes = self.score(data)
        scores = np.atleast_2d(scores)
        if data.domain.attributes != model_attributes:
            scores = np.array([join_derived_features(row) for row in scores])

        return scores[:, data.domain.attributes.index(feature)] \
            if feature else scores


class ClassificationScorer(Scorer):
    """
    Base class for feature scores in a class-labeled dataset.

    Parameters
    ----------
    feature : int, string, Orange.data.Variable
        Feature id
    data : Orange.data.Table
        Dataset

    Attributes
    ----------
    feature_type : Orange.data.Variable
        Required type of features.

    class_type : Orange.data.Variable
        Required type of class variable.
    """
    feature_type = DiscreteVariable
    class_type = DiscreteVariable
    supports_sparse_data = True
    preprocessors = Scorer.preprocessors + [
        Discretize(remove_const=False)
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


def _entropy(dist):
    """Entropy of class-distribution matrix"""
    p = dist / np.sum(dist, axis=0)
    pc = np.clip(p, 1e-15, 1)
    return np.sum(np.sum(- p * np.log2(pc), axis=0) *
                  np.sum(dist, axis=0) / np.sum(dist))


def _gini(dist):
    """Gini index of class-distribution matrix"""
    p = np.asarray(dist / np.sum(dist, axis=0))
    return np.sum((1 - np.sum(p ** 2, axis=0)) *
                  np.sum(dist, axis=0) / np.sum(dist))


def _symmetrical_uncertainty(data, attr1, attr2):
    """Symmetrical uncertainty, Press et al., 1988."""
    cont = np.asarray(contingency.Discrete(data, attr1, attr2), dtype=float)
    ig = InfoGain().from_contingency(cont, 1)
    return 2 * ig / (_entropy(cont) + _entropy(cont.T))


class FCBF(ClassificationScorer):
    """
    Fast Correlation-Based Filter. Described in:

    Yu, L., Liu, H.,
    Feature selection for high-dimensional data: A fast correlation-based filter solution.
    2003. http://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf
    """
    def score_data(self, data, feature=None):
        attributes = data.domain.attributes
        s = []
        for i, attr in enumerate(attributes):
            s.append((_symmetrical_uncertainty(data, attr, data.domain.class_var), i))
        s.sort()
        worst = []

        p = 1
        while True:
            try:
                _, Fp = s[-p]
            except IndexError:
                break
            q = p + 1
            while True:
                try:
                    suqc, Fq = s[-q]
                except IndexError:
                    break
                if _symmetrical_uncertainty(data, attributes[Fp], attributes[Fq]) >= suqc:
                    del s[-q]
                    worst.append((1e-4*suqc, Fq))
                else:
                    q += 1
            p += 1
        best = s
        scores = [i[0] for i in sorted(chain(best, worst), key=lambda i: i[1])]
        return np.array(scores) if not feature else scores[0]


class InfoGain(ClassificationScorer):
    """
    Information gain is the expected decrease of entropy. See `Wikipedia entry on information gain
    <http://en.wikipedia.org/wiki/Information_gain_ratio>`_.
    """
    def from_contingency(self, cont, nan_adjustment):
        h_class = _entropy(np.sum(cont, axis=1))
        h_residual = _entropy(np.compress(np.sum(cont, axis=0), cont, axis=1))
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
        h_residual = _entropy(np.compress(np.sum(cont, axis=0), cont, axis=1))
        h_attribute = _entropy(np.sum(cont, axis=0))
        if h_attribute == 0:
            h_attribute = 1
        return nan_adjustment * (h_class - h_residual) / h_attribute


class Gini(ClassificationScorer):
    """
    Gini impurity is the probability that two randomly chosen instances will have different
    classes. See `Wikipedia entry on Gini impurity
    <https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity>`_.
    """
    def from_contingency(self, cont, nan_adjustment):
        return (_gini(np.sum(cont, axis=1)) - _gini(cont)) * nan_adjustment


class ReliefF(Scorer):
    """
    ReliefF algorithm. Contrary to most other scorers, Relief family of
    algorithms is not as myoptic but tends to give unreliable results with
    datasets with lots (hundreds) of features.

    Robnik-Šikonja, M., Kononenko, I.
    Theoretical and empirical analysis of ReliefF and RReliefF.
    2003. http://lkm.fri.uni-lj.si/rmarko/papers/robnik03-mlj.pdf
    """
    feature_type = Variable
    class_type = DiscreteVariable
    supports_sparse_data = False
    friendly_name = "ReliefF"
    preprocessors = Scorer.preprocessors + [RemoveNaNColumns()]

    def __init__(self, n_iterations=50, k_nearest=10, random_state=None):
        self.n_iterations = n_iterations
        self.k_nearest = k_nearest
        self.random_state = random_state

    def score_data(self, data, feature):
        if len(data.domain.class_vars) != 1:
            raise ValueError('ReliefF requires one single class')
        if not data.domain.class_var.is_discrete:
            raise ValueError('ReliefF supports classification; use RReliefF '
                             'for regression')
        if len(data.domain.class_var.values) == 1:  # Single-class value non-problem
            return 0 if feature else np.zeros(data.X.shape[1])
        if isinstance(self.random_state, np.random.RandomState):
            rstate = self.random_state
        else:
            rstate = np.random.RandomState(self.random_state)

        from Orange.preprocess._relieff import relieff
        weights = np.asarray(relieff(data.X, data.Y,
                                     self.n_iterations, self.k_nearest,
                                     np.array([a.is_discrete for a in data.domain.attributes]),
                                     rstate))
        if feature:
            return weights[0]
        return weights


class RReliefF(Scorer):
    feature_type = Variable
    class_type = ContinuousVariable
    supports_sparse_data = False
    friendly_name = "RReliefF"
    preprocessors = Scorer.preprocessors + [RemoveNaNColumns()]

    def __init__(self, n_iterations=50, k_nearest=50, random_state=None):
        self.n_iterations = n_iterations
        self.k_nearest = k_nearest
        self.random_state = random_state

    def score_data(self, data, feature):
        if len(data.domain.class_vars) != 1:
            raise ValueError('RReliefF requires one single class')
        if not data.domain.class_var.is_continuous:
            raise ValueError('RReliefF supports regression; use ReliefF '
                             'for classification')
        if isinstance(self.random_state, np.random.RandomState):
            rstate = self.random_state
        else:
            rstate = np.random.RandomState(self.random_state)
        from Orange.preprocess._relieff import rrelieff
        weights = np.asarray(rrelieff(data.X, data.Y,
                                      self.n_iterations, self.k_nearest,
                                      np.array([a.is_discrete for a in data.domain.attributes]),
                                      rstate))
        if feature:
            return weights[0]
        return weights


if __name__ == '__main__':
    from Orange.data import Table
    X = np.random.random((500, 20))
    X[np.random.random(X.shape) > .95] = np.nan
    y_cls = np.zeros(X.shape[0])
    y_cls[(X[:, 0] > .5) ^ (X[:, 1] > .6)] = 1
    y_reg = np.nansum(X[:, 0:3], 1)
    for relief, y in ((ReliefF(), y_cls),
                      (RReliefF(), y_reg)):
        data = Table.from_numpy(None, X, y)
        weights = relief.score_data(data, False)
        print(relief.__class__.__name__)
        print('Best =', weights.argsort()[::-1])
        print('Weights =', weights[weights.argsort()[::-1]])
    X *= 10
    data = Table.from_numpy(None, X, y_cls)
    weights = FCBF().score_data(data, False)
    print('FCBF')
    print('Best =', weights.argsort()[::-1])
    print('Weights =', weights[weights.argsort()[::-1]])
