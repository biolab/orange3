import numpy
from sklearn import neighbors
from sklearn.preprocessing import Imputer
import Orange.data
import Orange.classification
from Orange.data.continuizer import DomainContinuizer

def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)

def replace_nan(X):
    # Default scikit Imputer
    # Use Orange imputer when implemented
    if numpy.isnan(X).sum():
            imp = Imputer()
            X = imp.fit_transform(X)
    return X

class KNNLearner(Orange.classification.Fitter):
    def __init__(self, n_neighbors=5, metric="euclidean", normalize=True):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.normalize = normalize

    def _domain_continuizer(self, data):
        multinomial = continuous = None
        if any(map(is_discrete, data.domain.attributes)):
            multinomial = DomainContinuizer.FrequentIsBase
        if self.normalize:
            continuous = DomainContinuizer.NormalizeBySD
        if multinomial is not None or continuous is not None:
            return DomainContinuizer(multinomial_treatment=multinomial,
                                     normalize_continuous=continuous)
        else:
            return None

    def __call__(self, data):
        dc = self._domain_continuizer(data)
        if dc is not None:
            domain = dc(data)
            data = Orange.data.Table.from_table(domain, data)

        return super().__call__(data)

    def fit(self, X, Y, W):
        X = replace_nan(X)
        if self.metric == "mahalanobis":
            skclf = neighbors.KNeighborsClassifier(
                n_neighbors=self.n_neighbors, metric=self.metric, V = numpy.cov(X.T)
            )
        else:
            skclf = neighbors.KNeighborsClassifier(
                n_neighbors=self.n_neighbors, metric=self.metric
            )
        skclf.fit(X, Y.ravel())
        return KNNClassifier(skclf)


class KNNClassifier(Orange.classification.Model):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        X = replace_nan(X)
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
