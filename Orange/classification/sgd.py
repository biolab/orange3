from sklearn.linear_model import SGDRegressor

from Orange.base import SklLearner


class SGDClassificationLearner(SklLearner):
    name = 'sgd'
    __wraps__ = SGDRegressor

    def __init__(self, loss='squared_loss',penalty='l2', alpha=0.0001,
                 l1_ratio=0.15,fit_intercept=True, n_iter=5, shuffle=True,
                 epsilon=0.1, random_state=None, learning_rate='invscaling',
                 eta0=0.01, power_t=0.25, warm_start=False, average=False,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
