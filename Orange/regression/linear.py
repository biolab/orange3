import numpy as np

import sklearn.linear_model as skl_linear_model
import sklearn.preprocessing as skl_preprocessing

from Orange.data import Variable, ContinuousVariable
from Orange.preprocess import Normalize
from Orange.preprocess.score import LearnerScorer
from Orange.regression import Learner, Model, SklLearner, SklModel


__all__ = ["LinearRegressionLearner", "RidgeRegressionLearner",
           "LassoRegressionLearner", "SGDRegressionLearner",
           "ElasticNetLearner", "ElasticNetCVLearner",
           "PolynomialLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data):
        data = Normalize()(data)
        model = self(data)
        return np.abs(model.coefficients)


class LinearRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_linear_model.LinearRegression

    def __init__(self, preprocessors=None):
        super().__init__(preprocessors=preprocessors)

    def fit(self, X, Y, W):
        model = super().fit(X, Y, W)
        return LinearModel(model.skl_model)


class RidgeRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.Ridge

    def __init__(self, alpha=1.0, fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=None,
                 tol=0.001, solver='auto', preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LassoRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.Lasso

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=0.0001, warm_start=False, positive=False,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class ElasticNetLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.ElasticNet

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=0.0001, warm_start=False, positive=False,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class ElasticNetCVLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.ElasticNetCV

    def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=0.0001, cv=None, copy_X=True,
                 verbose=0, n_jobs=1, positive=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SGDRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.SGDRegressor
    preprocessors = SklLearner.preprocessors + [Normalize()]

    def __init__(self, loss='squared_loss', penalty='l2', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=5, tol=None,
                 shuffle=True, epsilon=0.1, n_jobs=1, random_state=None,
                 learning_rate='invscaling', eta0=0.01, power_t=0.25,
                 class_weight=None, warm_start=False, average=False,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class PolynomialLearner(Learner):
    """Generate polynomial features and learn a prediction model

    Parameters
    ----------
    learner : LearnerRegression
        learner to be fitted on the transformed features
    degree : int
        degree of used polynomial
    preprocessors : List[Preprocessor]
        preprocessors to be applied on the data before learning
    """
    name = 'poly learner'
    preprocessors = SklLearner.preprocessors

    def __init__(self, learner=LinearRegressionLearner(), degree=2,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.degree = degree
        self.learner = learner

    def fit(self, X, Y, W):
        polyfeatures = skl_preprocessing.PolynomialFeatures(self.degree)
        X = polyfeatures.fit_transform(X)
        clf = self.learner
        if W is None or not self.supports_weights:
            model = clf.fit(X, Y, None)
        else:
            model = clf.fit(X, Y, sample_weight=W.reshape(-1))
        return PolynomialModel(model, polyfeatures)


class LinearModel(SklModel):
    @property
    def intercept(self):
        return self.skl_model.intercept_

    @property
    def coefficients(self):
        return self.skl_model.coef_

    def predict(self, X):
        vals = self.skl_model.predict(X)
        if len(vals.shape) == 1:
            # Prevent IndexError for 1D array
            return vals
        elif vals.shape[1] == 1:
            return vals.ravel()
        else:
            return vals

    def __str__(self):
        return 'LinearModel {}'.format(self.skl_model)


class PolynomialModel(Model):
    def __init__(self, model, polyfeatures):
        self.model = model
        self.polyfeatures = polyfeatures

    def predict(self, X):
        X = self.polyfeatures.fit_transform(X)
        return self.model.predict(X)

    def __str__(self):
        return 'PolynomialModel {}'.format(self.model)


PolynomialLearner.__returns__ = PolynomialModel
