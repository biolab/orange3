import sklearn.svm as skl_svm

from Orange.classification import SklLearner, SklModel

__all__ = ["SVMLearner", "LinearSVMLearner", "NuSVMLearner",
           "SVRLearner", "NuSVRLearner", "OneClassSVMLearner"]


class SVMClassifier(SklModel):

    def predict(self, X):
        value = self.clf.predict(X)
        if self.clf.probability:
            prob = self.clf.predict_proba(X)
            return value, prob
        return value


class SVMLearner(SklLearner):
    __wraps__ = skl_svm.SVC
    __returns__ = SVMClassifier
    name = 'svm'

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=0.001, cache_size=200, max_iter=-1,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_multiclass = True
        self.supports_weights = True


class LinearSVMLearner(SklLearner):
    __wraps__ = skl_svm.LinearSVC
    name = 'linear svm'

    def __init__(self, penalty='l2', loss='l2', dual=True, tol=0.0001,
                 C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=True, random_state=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_multiclass = True


class NuSVMClassifier(SklModel):

    def predict(self, X):
        value = self.clf.predict(X)
        if self.clf.probability:
            prob = self.clf.predict_proba(X)
            return value, prob
        return value


class NuSVMLearner(SklLearner):
    __wraps__ = skl_svm.NuSVC
    __returns__ = NuSVMClassifier
    name = 'nu svm'

    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_multiclass = True
        self.supports_weights = True


class SVRLearner(SklLearner):
    __wraps__ = skl_svm.SVR
    name = 'svr'

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_weights = True


class NuSVRLearner(SklLearner):
    __wraps__ = skl_svm.NuSVR
    name = 'nu svr'

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, tol=0.001,
                 cache_size=200, max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_weights = True


class OneClassSVMLearner(SklLearner):
    __wraps__ = skl_svm.OneClassSVM
    name = 'one class svm'

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=0.001, nu=0.5, shrinking=True, cache_size=200,
                 max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_weights = True

    def fit(self, X, Y=None, W=None):
        clf = self.__wraps__(**self.params)
        if W is not None:
            return self.__returns__(clf.fit(X, W.reshape(-1)))
        return self.__returns__(clf.fit(X))


if __name__ == '__main__':
    import Orange
    import numpy as np

    d1 = Orange.data.Table('iris')
    d1.shuffle()
    for learner in [SVMLearner, NuSVMLearner, LinearSVMLearner]:
        m = learner()
        print(m)
        cross = Orange.evaluation.CrossValidation(d1, m)
        prediction = cross.KFold(10)
        print(Orange.evaluation.CA(d1, prediction[0]))
        clf = m(d1)
        print(clf(d1[0].x, ret=clf.ValueProbs))

    d2 = Orange.data.Table('iris')
    d2.shuffle()
    n = int(0.7*d2.X.shape[0])
    train, test = d2[:n], d2[n:]
    for learner in [SVRLearner, NuSVRLearner]:
        m = learner()
        print(m)
        clf = m(train)
        print(1./test.Y.shape[0]*np.sum((clf(test)-test.Y.reshape(-1))**2))
