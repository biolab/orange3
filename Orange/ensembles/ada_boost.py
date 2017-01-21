import sklearn.ensemble as skl_ensemble

from Orange.base import SklLearner
from Orange.classification.base_classification import (
    SklLearnerClassification, SklModelClassification
)
from Orange.regression.base_regression import (
    SklLearnerRegression, SklModelRegression
)

__all__ = ['SklAdaBoostClassificationLearner', 'SklAdaBoostRegressionLearner']


class SklAdaBoostClassifier(SklModelClassification):
    pass


class SklAdaBoostClassificationLearner(SklLearnerClassification):
    __wraps__ = skl_ensemble.AdaBoostClassifier
    __returns__ = SklAdaBoostClassifier

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 algorithm='SAMME.R', random_state=None, preprocessors=None):
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(base_estimator, Fitter):
            base_estimator = base_estimator.get_learner(
                base_estimator.CLASSIFICATION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SklAdaBoostRegressor(SklModelRegression):
    pass


class SklAdaBoostRegressionLearner(SklLearnerRegression):
    __wraps__ = skl_ensemble.AdaBoostRegressor
    __returns__ = SklAdaBoostRegressor

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 loss='linear', random_state=None, preprocessors=None):
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(base_estimator, Fitter):
            base_estimator = base_estimator.get_learner(
                base_estimator.REGRESSION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
