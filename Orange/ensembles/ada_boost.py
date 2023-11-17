import warnings

import sklearn.ensemble as skl_ensemble

from Orange.base import SklLearner
from Orange.classification.base_classification import (
    SklLearnerClassification, SklModelClassification
)
from Orange.regression.base_regression import (
    SklLearnerRegression, SklModelRegression
)
from Orange.util import OrangeDeprecationWarning


__all__ = ['SklAdaBoostClassificationLearner', 'SklAdaBoostRegressionLearner']


class SklAdaBoostClassifier(SklModelClassification):
    pass


def base_estimator_deprecation():
    warnings.warn(
        "`base_estimator` is deprecated (to be removed in 3.39): use `estimator` instead.",
        OrangeDeprecationWarning, stacklevel=3)


class SklAdaBoostClassificationLearner(SklLearnerClassification):
    __wraps__ = skl_ensemble.AdaBoostClassifier
    __returns__ = SklAdaBoostClassifier
    supports_weights = True

    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.,
                 algorithm='SAMME.R', random_state=None, preprocessors=None,
                 base_estimator="deprecated"):
        if base_estimator != "deprecated":
            base_estimator_deprecation()
            estimator = base_estimator
        del base_estimator
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(estimator, Fitter):
            estimator = estimator.get_learner(
                estimator.CLASSIFICATION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(estimator, SklLearner):
            estimator = estimator.__wraps__(**estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SklAdaBoostRegressor(SklModelRegression):
    pass


class SklAdaBoostRegressionLearner(SklLearnerRegression):
    __wraps__ = skl_ensemble.AdaBoostRegressor
    __returns__ = SklAdaBoostRegressor
    supports_weights = True

    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.,
                 loss='linear', random_state=None, preprocessors=None,
                 base_estimator="deprecated"):
        if base_estimator != "deprecated":
            base_estimator_deprecation()
            estimator = base_estimator
        del base_estimator
        from Orange.modelling import Fitter
        # If fitter, get the appropriate Learner instance
        if isinstance(estimator, Fitter):
            estimator = estimator.get_learner(
                estimator.REGRESSION)
        # If sklearn learner, get the underlying sklearn representation
        if isinstance(estimator, SklLearner):
            estimator = estimator.__wraps__(**estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
