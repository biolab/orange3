from sklearn.linear_model import SGDClassifier

from Orange.base import SklLearner
from Orange.preprocess import Normalize
from Orange.regression.linear import LinearModel

__all__ = ["SGDClassificationLearner"]


class SGDClassificationLearner(SklLearner):
    name = 'sgd'
    __wraps__ = SGDClassifier
    __returns__ = LinearModel
    preprocessors = SklLearner.preprocessors + [Normalize()]

    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001,
                 l1_ratio=0.15,fit_intercept=True, n_iter=5, shuffle=True,
                 epsilon=0.1, random_state=None, learning_rate='invscaling',
                 eta0=0.01, power_t=0.25, warm_start=False, average=False,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
