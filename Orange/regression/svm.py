import sklearn.svm as skl_svm

from Orange.regression import SklLearner
from Orange.preprocess import Normalize

__all__ = ["SVRLearner", "LinearSVRLearner", "NuSVRLearner"]

svm_pps = SklLearner.preprocessors + [Normalize()]


class SVRLearner(SklLearner):
    __wraps__ = skl_svm.SVR
    preprocessors = svm_pps

    def __init__(self, kernel='rbf', degree=3, gamma="auto", coef0=0.0,
                 tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LinearSVRLearner(SklLearner):
    __wraps__ = skl_svm.LinearSVR
    preprocessors = svm_pps

    def __init__(self, epsilon=0., tol=.0001, C=1., loss='epsilon_insensitive',
                 fit_intercept=True, intercept_scaling=1., dual=True,
                 random_state=None, max_iter=1000, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class NuSVRLearner(SklLearner):
    __wraps__ = skl_svm.NuSVR
    preprocessors = svm_pps

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma="auto",
                 coef0=0.0, shrinking=True, tol=0.001,
                 cache_size=200, max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('housing')
    learners = [SVRLearner(), LinearSVRLearner(), NuSVRLearner()]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.RMSE(res)):
        print("learner: {}\nRMSE: {}\n".format(l, ca))

