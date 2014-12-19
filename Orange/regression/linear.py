
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

from ..classification import Fitter, Model
from Orange import classification

class LinearRegressionLearner(Fitter):
    def fit(self, X, Y, W):
        sk = sklearn.linear_model.LinearRegression()
        sk.fit(X, Y)
        return LinearModel(sk)


class RidgeRegressionLearner(Fitter):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y, W):
        sk = sklearn.linear_model.Ridge(alpha=self.alpha, fit_intercept=True)
        sk.fit(X, Y)
        return LinearModel(sk)


class LassoRegressionLearner(Fitter):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y, W):
        sk = sklearn.linear_model.Lasso(alpha=self.alpha, fit_intercept=True)
        sk.fit(X, Y)
        return LinearModel(sk)

class SGDRegressionLearner(classification.SklFitter):
    def __init__(self, loss='squared_loss', alpha=0.0001, epsilon=0.1, 
        eta0=0.01, l1_ratio=0.15, penalty='l2', power_t=0.25, 
        learning_rate='invscaling', n_iter=5, fit_intercept=True):
        self.params = vars()
    
    def fit(self, X, Y, W):
        sk = sklearn.linear_model.SGDRegressor(**self.params)
        clf = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler()), ('sgd', sk)])
        clf.fit(X,Y.ravel())
        return LinearModel(clf)

class LinearModel(Model):
    supports_multiclass = True

    def __init__(self, skmodel):
        self.skmodel = skmodel

    def predict(self, table):
        vals = self.skmodel.predict(table)
        if len(vals.shape) == 1:
            # Prevent IndexError for 1D array
            return vals
        elif vals.shape[1] == 1:
            return vals.ravel()
        else:

            return vals
    def __str__(self):
        return 'LinearModel {}'.format(self.skmodel)
