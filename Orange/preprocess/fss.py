import random
import Orange
import numpy as np

from itertools import takewhile
from operator import itemgetter

from Orange.preprocess.preprocess import Preprocess

__all__ = ["SelectBestFeatures", "RemoveNaNColumns", "SelectRandomFeatures"]


class SelectBestFeatures:
    """
    A feature selector that builds a new data set consisting of either the top
    `k` features or all those that exceed a given `threshold`. Features are
    scored using the provided feature scoring `method`. By default it is
    assumed that feature importance diminishes with decreasing scores.

    If both `k` and `threshold` are set, only features satisfying both
    conditions will be selected.

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
        if not isinstance(data.domain.class_var, self.method.class_type):
            raise ValueError(("Scoring method {} requires a class variable " +
                              "of type {}.").format(
                type(self.method), self.method.class_type))
        features = data.domain.attributes
        try:
            scores = self.method(data)
        except ValueError:
            scores = self.score_only_nice_features(data)
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

    def score_only_nice_features(self, data):
        mask = [1 if isinstance(a, self.method.feature_type) else 0
                for a in data.domain.attributes]
        features = [f for f in data.domain.attributes
                    if isinstance(f, self.method.feature_type)]
        scores = [self.method(data, f) for f in features]
        all_scores = np.array([float('-inf')] * len(data.domain.attributes))
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
            random.sample(data.domain.attributes, self.k),
            data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class RemoveNaNColumns(Preprocess):
    """
    Removes data columns that contain only unknown values. Returns the
    resulting data set. Does not check optional class attribute(s).

    data : data table
        an input data table
    """
    def __call__(self, data):
        nan_col = np.all(np.isnan(data.X), axis=0)
        att = [a for a, nan in zip(data.domain.attributes, nan_col) if not nan]
        domain = Orange.data.Domain(att, data.domain.class_vars,
                                    data.domain.metas)
        return Orange.data.Table(domain, data)
