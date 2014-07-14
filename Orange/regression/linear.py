
import sklearn.linear_model

from ..classification import Fitter, Model


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


class LinearModel(Model):
    supports_multiclass = True

    def __init__(self, skmodel):
        self.skmodel = skmodel

    def predict(self, table):
        vals = self.skmodel.predict(table)
        if vals.shape[1] == 1:
            return vals.ravel()
        else:
            return vals
