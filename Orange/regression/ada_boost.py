import sklearn.ensemble as skl_ensemble
from Orange.regression import SklLearner, SklModel

__all__ = ["AdaBoostRegressionLearner"]


class AdaBoostRegressor(SklModel):
    pass


class AdaBoostRegressionLearner(SklLearner):
    __wraps__ = skl_ensemble.AdaBoostRegressor
    __returns__ = AdaBoostRegressor
    name = 'adaBoost regression'

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 loss='linear', random_state=None, preprocessors=None):
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
