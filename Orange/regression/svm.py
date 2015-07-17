import sklearn.svm as skl_svm

from Orange.regression import SklLearner
from Orange.preprocess import Normalize

__all__ = ["SVRLearner", "NuSVRLearner"]

svm_pps = SklLearner.preprocessors + [Normalize()]


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


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('housing')
    learners = [SVRLearner(), NuSVRLearner()]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.RMSE(res)):
        print("learner: {}\nRMSE: {}\n".format(l, ca))

