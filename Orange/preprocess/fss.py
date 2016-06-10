import random
import Orange
import numpy as np
from scipy.sparse import issparse

from itertools import takewhile
from operator import itemgetter

from Orange.preprocess.preprocess import Preprocess
from Orange.preprocess.score import ANOVA, GainRatio, UnivariateLinearRegression
from Orange.data import Domain

__all__ = ["SelectBestFeatures", "RemoveNaNColumns", "SelectRandomFeatures"]


class SelectBestFeatures:
    """
    A feature selector that builds a new data set consisting of either the top
    `k` features or all those that exceed a given `threshold`. Features are
    scored using the provided feature scoring `method`. By default it is
    assumed that feature importance diminishes with decreasing scores.

    If both `k` and `threshold` are set, only features satisfying both
    conditions will be selected.

    If `method` is not set, it is automatically selected when presented with
    the data set. Data sets with both continuous and discrete features are
    scored using a method suitable for the majority of features.

    Parameters
    ----------
    method : Orange.preprocess.score.ClassificationScorer, Orange.preprocess.score.SklScorer
        Univariate feature scoring method.

    k : int
        The number of top features to select.

    threshold : float
        A threshold that a feature should meet according to the provided method.

    decreasing : boolean
        The order of feature importance when sorted from the most to the least
        important feature.
    """

    def __init__(self, method=None, k=None, threshold=None, decreasing=True):
        self.method = method
        self.k = k
        self.threshold = threshold
        self.decreasing = decreasing

    def __call__(self, data):
        method = self.method
        # select default method according to the provided data
        if method is None:
            autoMethod = True
            discr_ratio = (sum(a.is_discrete
                               for a in data.domain.attributes)
                           / len(data.domain.attributes))
            if data.domain.has_discrete_class:
                if discr_ratio >= 0.5:
                    method = GainRatio()
                else:
                    method = ANOVA()
            else:
                method = UnivariateLinearRegression()

        if not isinstance(data.domain.class_var, method.class_type):
            raise ValueError(("Scoring method {} requires a class variable " +
                              "of type {}.").format(
                (method if type(method) == type else type(method)).__name__,
                method.class_type.__name__)
            )
        features = data.domain.attributes
        try:
            scores = method(data)
        except ValueError:
            scores = self.score_only_nice_features(data, method)
        best = sorted(zip(scores, features), key=itemgetter(0),
                      reverse=self.decreasing)
        if self.k:
            best = best[:self.k]
        if self.threshold:
            pred = ((lambda x: x[0] >= self.threshold) if self.decreasing else
                    (lambda x: x[0] <= self.threshold))
            best = takewhile(pred, best)

        domain = Orange.data.Domain([f for s, f in best],
                                    data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)

    def score_only_nice_features(self, data, method):
        mask = np.array([isinstance(a, method.feature_type)
                         for a in data.domain.attributes])
        features = [f for f in data.domain.attributes
                    if isinstance(f, method.feature_type)]
        scores = [method(data, f) for f in features]
        bad = float('-inf') if self.decreasing else float('inf')
        all_scores = np.array([bad] * len(data.domain.attributes))
        all_scores[mask] = scores
        return all_scores


class SelectRandomFeatures:
    """
    A feature selector that selects random `k` features from an input
    data set and returns a data set with selected features. Parameter
    `k` is either an integer (number of feature) or float (from 0.0 to
    1.0, proportion of retained features).

    Parameters
    ----------

    k : int or float (default = 0.1)
        The number or proportion of features to retain.
    """

    def __init__(self, k=0.1):
        self.k = k

    def __call__(self, data):
        if type(self.k) == float:
            self.k = int(len(data.domain.attributes) * self.k)
        domain = Orange.data.Domain(
            random.sample(data.domain.attributes,
                          min(self.k, len(data.domain.attributes))),
            data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class RemoveNaNColumns(Preprocess):
    """
    Remove features from the data domain if they contain
    `threshold` or more unknown values.

    `threshold` can be an integer or a float in the range (0, 1) representing
    the fraction of the data size. When not provided, columns with only missing
    values are removed (default).
    """
    def __init__(self, threshold=None):
        self.threshold = threshold

    def __call__(self, data, threshold=None):
        # missing entries in sparse data are treated as zeros so we skip removing NaNs
        if issparse(data.X):
            return data

        if threshold is None:
            threshold = data.X.shape[0] if self.threshold is None else \
                        self.threshold
        if isinstance(threshold, float):
            threshold = threshold * data.X.shape[0]
        nans = np.sum(np.isnan(data.X), axis=0)
        att = [a for a, n in zip(data.domain.attributes, nans) if n < threshold]
        domain = Orange.data.Domain(att, data.domain.class_vars,
                                    data.domain.metas)
        return Orange.data.Table(domain, data)
