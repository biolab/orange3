import warnings

import numpy as np
import dask.array as da

import sklearn.linear_model as skl_linear_model
import sklearn.preprocessing as skl_preprocessing

try:
    import dask_ml.linear_model as dask_linear_model
    from dask_glm.regularizers import ElasticNet
except ImportError:
    dask_linear_model = skl_linear_model
    ElasticNet = ...

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
        return np.abs(model.coefficients), model.domain.attributes


class LinearRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_linear_model.LinearRegression
    __penalty__ = None

    # Arguments are needed for signatures, pylint: disable=unused-argument
    def __init__(self, preprocessors=None, fit_intercept=True):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def _initialize_wrapped(self, X=None, Y=None):
        if isinstance(X, da.Array) or isinstance(Y, da.Array):
            if dask_linear_model is skl_linear_model:
                warnings.warn("dask_ml is not installed, using sklearn instead.")
            else:
                params = self.params.copy()
                penalty = self.__penalty__
                params["solver"] = "gradient_descent"

                if penalty is not None:
                    if penalty == "elasticnet":
                        penalty = ElasticNet(weight=params.pop("l1_ratio"))
                    params["penalty"] = penalty
                    params["solver"] = "admm"
                    params["C"] = 1 / params.pop("alpha")
                    params["max_iter"] = params["max_iter"] or 100
                    for key in ["copy_X", "precompute", "positive"]:
                        params.pop(key, None)

                return dask_linear_model.LinearRegression(**params)
        return self.__wraps__(**self.params)

    def fit(self, X, Y, W=None):
        model = super().fit(X, Y, W)
        return LinearModel(model.skl_model)


class RidgeRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.Ridge
    __penalty__ = "l2"

    # Arguments are needed for signatures, pylint: disable=unused-argument
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True,
                 max_iter=None, tol=0.001, solver='auto', preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LassoRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.Lasso
    __penalty__ = "l1"

    # Arguments are needed for signatures, pylint: disable=unused-argument
    def __init__(self, alpha=1.0, fit_intercept=True, precompute=False,
                 copy_X=True, max_iter=1000, tol=0.0001, warm_start=False,
                 positive=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class ElasticNetLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.ElasticNet
    __penalty__ = "elasticnet"

    # Arguments are needed for signatures, pylint: disable=unused-argument
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 precompute=False, max_iter=1000, copy_X=True, tol=0.0001,
                 warm_start=False, positive=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class ElasticNetCVLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.ElasticNetCV

    # Arguments are needed for signatures, pylint: disable=unused-argument
    def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
                 fit_intercept=True, precompute='auto', max_iter=1000,
                 tol=0.0001, cv=5, copy_X=True, verbose=0, n_jobs=1,
                 positive=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SGDRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.SGDRegressor
    preprocessors = SklLearner.preprocessors + [Normalize()]

    # Arguments are needed for signatures, pylint: disable=unused-argument
    def __init__(self, loss='squared_error', penalty='l2', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=5, tol=1e-3,
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
                 preprocessors=None, include_bias=True):
        super().__init__(preprocessors=preprocessors)
        self.degree = degree
        self.learner = learner
        self.include_bias = include_bias

    def fit(self, X, Y, W=None):
        polyfeatures = skl_preprocessing.PolynomialFeatures(
            self.degree, include_bias=self.include_bias)
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

    def __str__(self):
        return 'LinearModel {}'.format(self.skl_model)


class PolynomialModel(Model):
    def __init__(self, model, polyfeatures):
        super().__init__()
        self.model = model
        self.polyfeatures = polyfeatures

    def predict(self, X):
        X = self.polyfeatures.fit_transform(X)
        return self.model.predict(X)

    def __str__(self):
        return 'PolynomialModel {}'.format(self.model)


PolynomialLearner.__returns__ = PolynomialModel
