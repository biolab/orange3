import sklearn.svm as skl_svm

from Orange.classification import SklLearner, SklModel
from Orange.preprocess import Continuize

__all__ = ["SVMLearner", "LinearSVMLearner", "NuSVMLearner",
           "SVRLearner", "NuSVRLearner", "OneClassSVMLearner"]


svm_pps = SklLearner.preprocessors + \
      [Continuize(normalize_continuous=Continuize.NormalizeBySD)]


class SVMClassifier(SklModel):

    def predict(self, X):
        value = self.skl_model.predict(X)
        if self.skl_model.probability:
            prob = self.skl_model.predict_proba(X)
            return value, prob
        return value


class SVMLearner(SklLearner):
    __wraps__ = skl_svm.SVC
    __returns__ = SVMClassifier
    name = 'svm'
    preprocessors = svm_pps

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
    preprocessors = svm_pps

    def __init__(self, penalty='l2', loss='l2', dual=True, tol=0.0001,
                 C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=True, random_state=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_multiclass = True


class NuSVMClassifier(SklModel):

    def predict(self, X):
        value = self.skl_model.predict(X)
        if self.skl_model.probability:
            prob = self.skl_model.predict_proba(X)
            return value, prob
        return value


class NuSVMLearner(SklLearner):
    __wraps__ = skl_svm.NuSVC
    __returns__ = NuSVMClassifier
    name = 'nu svm'
    preprocessors = svm_pps

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
    preprocessors = svm_pps

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_weights = True


class NuSVRLearner(SklLearner):
    __wraps__ = skl_svm.NuSVR
    name = 'nu svr'
    preprocessors = svm_pps

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, tol=0.001,
                 cache_size=200, max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_weights = True


class OneClassSVMLearner(SklLearner):
    __wraps__ = skl_svm.OneClassSVM
    name = 'one class svm'
    preprocessors = svm_pps

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

    data = Orange.data.Table('iris')
    learners = [SVMLearner(), NuSVMLearner(), LinearSVMLearner()]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.CA(res)):
        print("learner: {}\nCA: {}\n".format(l, ca))

    data = Orange.data.Table('housing')
    learners = [SVRLearner(), NuSVRLearner()]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.RMSE(res)):
        print("learner: {}\nRMSE: {}\n".format(l, ca))
