from bisect import bisect_left, bisect_right

import Orange

__all__ = ["SelectKBest", "SelectThreshold"]


class FeatureSelector:
    def __init__(self, method=None, decreasing=True):
        self.method = method
        self.decreasing = decreasing

    def select_best(self, scores):
        raise NotImplementedError("Subclasses should override this method.")

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
        top_ind = self.select_best(scores)
        domain = Orange.data.Domain([features[i] for i in top_ind] + other,
                                    data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class SelectKBest(FeatureSelector):
    def __init__(self, k=10, method=None, decreasing=True):
        super().__init__(method, decreasing)
        self.k = k

    def select_best(self, scores):
        top = sorted(zip(scores, range(len(scores))), reverse=self.decreasing)
        return [i for s, i in top[:self.k]]


class SelectThreshold(FeatureSelector):
    def __init__(self, threshold=None, method=None, decreasing=True):
        super().__init__(method, decreasing)
        self.threshold = threshold

    def select_best(self, scores):
        top = sorted(zip(scores, range(len(scores))))
        if self.decreasing:
            top = top[bisect_left(top, (self.threshold, -1)):][::-1]
        else:
            top = top[:bisect_right(top, (self.threshold, len(top)))]
        return [i for s, i in top]
