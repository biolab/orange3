import sklearn.ensemble as skl_ensemble
from Orange.base import SklLearner
from Orange.classification.base_classification import (SklLearnerClassification,
                                                       SklModelClassification)
from Orange.regression.base_regression import (SklLearnerRegression,
                                               SklModelRegression)
from Orange import options

__all__ = ["SklAdaBoostLearner", "SklAdaBoostRegressionLearner"]


class SklAdaBoostClassifier(SklModelClassification):
    pass


class BaseAdaBoostOptions:
    base_estimator = options.ObjectOption('base_estimator')
    options = [
        options.IntegerOption('n_estimators', default=50, range=(1, 1000), step=10),
        options.FloatOption('learning_rate', default=1., range=(0.05, 1.), step=.05),
        options.IntegerOption('random_state', default=0, range=(0, 100)),
    ]


class SklAdaBoostLearner(SklLearnerClassification):
    __wraps__ = skl_ensemble.AdaBoostClassifier
    __returns__ = SklAdaBoostClassifier
    name = 'skl adaBoost'
    verbose_name = 'Ada Boost'

    options = [
        BaseAdaBoostOptions.base_estimator,
        options.ChoiceOption('algorithm', choices=('SAMME', 'SAMME.R')),
    ] + BaseAdaBoostOptions.options

    def __init__(self, base_estimator=None, **kwargs):
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.instance
        super().__init__(base_estimator=base_estimator, **kwargs)


class SklAdaBoostRegressor(SklModelRegression):
    pass


class SklAdaBoostRegressionLearner(SklLearnerRegression):
    __wraps__ = skl_ensemble.AdaBoostRegressor
    __returns__ = SklAdaBoostRegressor
    name = 'skl adaBoost regression'
    verbose_name = 'Ada Boost Regression'

    options = [
        BaseAdaBoostOptions.base_estimator,
        options.ChoiceOption('loss', choices=('linear', 'square', 'exponential'))
    ] + BaseAdaBoostOptions.options

    def __init__(self, base_estimator=None, **kwargs):
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.instance
        super().__init__(base_estimator=base_estimator, **kwargs)
