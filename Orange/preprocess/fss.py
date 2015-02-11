from itertools import takewhile
from operator import itemgetter

import Orange

__all__ = ["SelectBestFeatures"]


class SelectBestFeatures:
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
        features = [f for f in data.domain.attributes
                    if isinstance(f, self.method.feature_type)]
        other = [f for f in data.domain.attributes
                 if not isinstance(f, self.method.feature_type)]
        scores = [self.method(f, data) for f in features]
        best = sorted(zip(scores, features), key=itemgetter(0),
                      reverse=self.decreasing)
        if self.k:
            best = best[:self.k]
        if self.threshold:
            pred = ((lambda x: x[0] >= self.threshold) if self.decreasing else
                    (lambda x: x[0] <= self.threshold))
            best = takewhile(pred, best)

        domain = Orange.data.Domain([f for s, f in best] + other,
                                    data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
