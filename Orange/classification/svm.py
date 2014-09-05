from Orange import classification
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR, NuSVR, OneClassSVM

class SVMLearner(classification.SklFitter):

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=0.001, cache_size=200, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.supports_multiclass = True

    def fit(self, X, Y, W):
        clf = SVC(C=self.C, kernel=self.kernel, degree=self.degree,
                  gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking,
                  probability=self.probability, tol=self.tol,
                  cache_size=self.cache_size, max_iter=self.max_iter)
        if W.shape[1]>0:
            return SVMClassifier(clf.fit(X, Y.reshape(-1), W.reshape(-1)))
        return SVMClassifier(clf.fit(X, Y.reshape(-1)))

class SVMClassifier(classification.SklModel):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        if self.clf.probability:
            prob = self.clf.predict_proba(X)
            return value, prob
        return value


class LinearSVMLearner(classification.SklFitter):

    def __init__(self, penalty='l2', loss='l2', dual=True, tol=0.0001,
                C=1.0, multi_class='ovr', fit_intercept=True,
                intercept_scaling=True, random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.supports_multiclass = True

    def fit(self, X, Y, W):
        clf = LinearSVC(penalty=self.penalty, loss=self.loss, dual=self.dual,
                        tol=self.tol, C=self.C, multi_class=self.multi_class,
                        fit_intercept=self.fit_intercept,
                        intercept_scaling=self.intercept_scaling,
                        random_state=self.random_state)
        return LinearSVMClassifier(clf.fit(X, Y.reshape(-1)))

class LinearSVMClassifier(classification.SklModel):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        return value


class NuSVMLearner(classification.SklFitter):

    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 max_iter=-1):
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.supports_multiclass = True

    def fit(self, X, Y, W):
        clf = NuSVC(nu=self.nu, kernel=self.kernel, degree=self.degree,
                    gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking,
                    probability=self.probability, tol=self.tol, cache_size=self.cache_size,
                    max_iter=self.max_iter)
        if W.shape[1]>0:
            return NuSVMClassifier(clf.fit(X, Y.reshape(-1), W.reshape(-1)))
        return NuSVMClassifier(clf.fit(X, Y.reshape(-1)))

class NuSVMClassifier(classification.SklModel):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        if self.clf.probability:
            prob = self.clf.predict_proba(X)
            return value, prob
        return value


class SVRLearner(classification.SklFitter):

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                cache_size=200, max_iter=-1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking  = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter

    def fit(self, X, Y, W):
        clf = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                  coef0=self.coef0, tol=self.tol, C=self.C, epsilon=self.epsilon,
                  shrinking=self.shrinking, cache_size=self.cache_size,
                  max_iter=self.max_iter)
        if W.shape[1]>0:
            return SVRClassifier(clf.fit(X, Y.reshape(-1), W.reshape(-1)))
        return SVRClassifier(clf.fit(X, Y.reshape(-1)))

class SVRClassifier(classification.SklModel):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        return value


class NuSVRLearner(classification.SklFitter):

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, tol=0.001,
                 cache_size=200, max_iter=-1):
        self.nu = nu
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter

    def fit(self, X, Y, W):
        clf = NuSVR(nu=self.nu, C=self.C, kernel=self.kernel, degree=self.degree,
                    gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking,
                    tol=self.tol, cache_size=self.cache_size,
                    max_iter=self.max_iter)
        if W.shape[1]>0:
            return NuSVRClassifier(clf.fit(X, Y.reshape(-1), W.reshape(-1)))
        return NuSVRClassifier(clf.fit(X, Y.reshape(-1)))

class NuSVRClassifier(classification.SklModel):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        return value


class OneClassSVMLearner(classification.SklFitter):

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=0.001, nu=0.5, shrinking=True, cache_size=200, max_iter=-1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter

    def fit(self, X, Y, W):
        clf = OneClassSVM(kernel=self.kernel, degree=self.degree,
                          gamma=self.gamma, coef0=self.coef0, tol=self.tol,
                          nu=self.nu, shrinking=self.shrinking,
                          cache_size=self.cache_size, max_iter=self.max_iter)
        if W.shape[1]>0:
            return OneClassSVMClassifier(clf.fit(X, W.reshape(-1)))
        return OneClassSVMClassifier(clf.fit(X))

class OneClassSVMClassifier(classification.SklModel):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        return value


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
