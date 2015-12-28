import sklearn.ensemble as skl_ensemble
from Orange.classification import SklLearner, SklModel

__all__ = ["AdaBoostLearner"]


class AdaBoostClassifier(SklModel):
    pass


class AdaBoostLearner(SklLearner):
    __wraps__ = skl_ensemble.AdaBoostClassifier
    __returns__ = AdaBoostClassifier
    name = 'adaBoost'

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.,
                 algorithm='SAMME.R', random_state=None, preprocessors=None):
        if isinstance(base_estimator, SklLearner):
            base_estimator = base_estimator.__wraps__(**base_estimator.params)
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
