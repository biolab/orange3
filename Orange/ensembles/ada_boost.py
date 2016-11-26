import sklearn.ensemble as skl_ensemble
from Orange.base import SklLearner
from Orange.classification.base_classification import (SklLearnerClassification,
                                                       SklModelClassification)
from Orange.regression.base_regression import (SklLearnerRegression,
                                               SklModelRegression)

__all__ = ["SklAdaBoostLearner", "SklAdaBoostRegressionLearner"]


class SklAdaBoostClassifier(SklModelClassification):
    pass


class SklAdaBoostLearner(SklLearnerClassification):
    __wraps__ = skl_ensemble.AdaBoostClassifier
    __returns__ = SklAdaBoostClassifier
    name = 'skl adaBoost'

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 algorithm='SAMME.R', random_state=None, preprocessors=None):
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SklAdaBoostRegressor(SklModelRegression):
    pass


class SklAdaBoostRegressionLearner(SklLearnerRegression):
    __wraps__ = skl_ensemble.AdaBoostRegressor
    __returns__ = SklAdaBoostRegressor
    name = 'skl adaBoost regression'

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 loss='linear', random_state=None, preprocessors=None):
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
