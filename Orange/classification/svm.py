import sklearn.svm as skl_svm

from Orange.classification import SklLearner, SklModel
from Orange.preprocess import AdaptiveNormalize

__all__ = ["SVMLearner", "LinearSVMLearner", "NuSVMLearner"]

svm_pps = SklLearner.preprocessors + [AdaptiveNormalize()]


class SVMLearner(SklLearner):
    __wraps__ = skl_svm.SVC
    preprocessors = svm_pps

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma="auto",
                 coef0=0.0, shrinking=True, probability=False,
                 tol=0.001, cache_size=200, max_iter=-1,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LinearSVMLearner(SklLearner):
    __wraps__ = skl_svm.LinearSVC
    preprocessors = svm_pps

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True,
                 tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=True, random_state=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


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
    preprocessors = svm_pps

    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma="auto", coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('iris')
    learners = [SVMLearner(), NuSVMLearner(), LinearSVMLearner()]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.CA(res)):
        print("learner: {}\nCA: {}\n".format(l, ca))
