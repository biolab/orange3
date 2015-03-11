import sklearn.linear_model as skl_linear_model
import sklearn.pipeline as skl_pipeline
import sklearn.preprocessing as skl_preprocessing

from Orange.classification import Learner, Model, SklLearner

__all__ = ["LinearRegressionLearner", "RidgeRegressionLearner",
           "LassoRegressionLearner", "SGDRegressionLearner",
           "ElasticNetLearner", "ElasticNetCVLearner"]


class LinearRegressionLearner(SklLearner):
    __wraps__ = skl_linear_model.LinearRegression
    name = 'linreg'

    def fit(self, X, Y, W):
        sk = skl_linear_model.LinearRegression()
        sk.fit(X, Y)
        return LinearModel(sk)


class RidgeRegressionLearner(SklLearner):
    __wraps__ = skl_linear_model.Ridge
    name = 'ridge'

    def __init__(self, alpha=1.0, fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=None,
                 tol=0.001, solver='auto'):
        self.params = vars()


class LassoRegressionLearner(SklLearner):
    __wraps__ = skl_linear_model.Lasso
    name = 'lasso'

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute='auto', copy_X=True, max_iter=1000,
                 tol=0.0001, warm_start=False, positive=False):
        self.params = vars()


class ElasticNetLearner(SklLearner):
    __wraps__ = skl_linear_model.ElasticNet
    name = 'elastic'

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute='auto', max_iter=1000,
                 copy_X=True, tol=0.0001, warm_start=False, positive=False):
        self.params = vars()


class ElasticNetCVLearner(SklLearner):
    __wraps__ = skl_linear_model.ElasticNetCV
    name = 'elasticCV'

    def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=0.0001, cv=None, copy_X=True,
                 verbose=0, n_jobs=1, positive=False):
        self.params = vars()


class SGDRegressionLearner(SklLearner):
    __wraps__ = skl_linear_model.SGDRegressor
    name = 'sgd'

    def __init__(self, loss='squared_loss', alpha=0.0001, epsilon=0.1,
                 eta0=0.01, l1_ratio=0.15, penalty='l2', power_t=0.25,
                 learning_rate='invscaling', n_iter=5, fit_intercept=True,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y, W):
        sk = skl_linear_model.SGDRegressor(**self.params)
        clf = skl_pipeline.Pipeline(
            [('scaler', skl_preprocessing.StandardScaler()), ('sgd', sk)])
        clf.fit(X, Y.ravel())
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
