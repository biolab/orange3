import sklearn.svm as sklsvm

from Orange.classification import SklFitter, SklModel


class SVMClassifier(SklModel):

    def predict(self, X):
        value = self.clf.predict(X)
        if self.clf.probability:
            prob = self.clf.predict_proba(X)
            return value, prob
        return value


class SVMLearner(SklFitter):
    __wraps__ = sklsvm.SVC
    __returns__ = SVMClassifier

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=0.001, cache_size=200, max_iter=-1):
        self.params = vars()
        self.supports_multiclass = True
        self.supports_weights = True


class LinearSVMLearner(SklFitter):
    __wraps__ = sklsvm.LinearSVC

    def __init__(self, penalty='l2', loss='l2', dual=True, tol=0.0001,
                C=1.0, multi_class='ovr', fit_intercept=True,
                intercept_scaling=True, random_state=None):
        self.params = vars()
        self.supports_multiclass = True


class NuSVMClassifier(SklModel):

    def predict(self, X):
        value = self.clf.predict(X)
        if self.clf.probability:
            prob = self.clf.predict_proba(X)
            return value, prob
        return value


class NuSVMLearner(SklFitter):
    __wraps__ = sklsvm.NuSVC
    __returns__ = NuSVMClassifier

    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 max_iter=-1):
        self.params = vars()
        self.supports_multiclass = True
        self.supports_weights = True


class SVRLearner(SklFitter):
    __wraps__ = sklsvm.SVR

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                cache_size=200, max_iter=-1):
        self.params = vars()
        self.supports_weights = True


class NuSVRLearner(SklFitter):
    __wraps__ = sklsvm.NuSVR

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, tol=0.001,
                 cache_size=200, max_iter=-1):
        self.params = vars()
        self.supports_weights = True


class OneClassSVMLearner(SklFitter):
    __wraps__ = sklsvm.OneClassSVM

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=0.001, nu=0.5, shrinking=True, cache_size=200, max_iter=-1):
        self.params = vars()
        self.supports_weights = True

    def fit(self, X, Y=None, W=None):
        clf = self.__wraps__(**self.params)
        if W is not None:
            return self.__returns__(clf.fit(X, W.reshape(-1)))
        return self.__returns__(clf.fit(X))



if __name__ == '__main__':
    import Orange.data
    import Orange.evaluation
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
