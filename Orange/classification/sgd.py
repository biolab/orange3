from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Orange.base import SklLearner
from Orange.regression.linear import LinearModel


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

    def fit(self, X, Y, W):
        sk = self.__wraps__(**self.params)
        clf = Pipeline([('scaler', StandardScaler()), ('sgd', sk)])
        clf.fit(X, Y.ravel())
        return LinearModel(clf)
