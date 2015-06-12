import sklearn.linear_model as skl_linear_model

from Orange.classification import SklLearner, SklModel
from Orange.preprocess import Normalize

__all__ = ["LogisticRegressionLearner"]


class LogisticRegressionClassifier(SklModel):
    pass


class LogisticRegressionLearner(SklLearner):
    __wraps__ = skl_linear_model.LogisticRegression
    __returns__ = LogisticRegressionClassifier
    name = 'logreg'
    preprocessors = SklLearner.preprocessors + [Normalize()]

    def __init__(self, penalty="l2", dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
