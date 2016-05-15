import numpy as np

import sklearn.linear_model as skl_linear_model
import sklearn.pipeline as skl_pipeline
import sklearn.preprocessing as skl_preprocessing

from Orange.data import Variable, ContinuousVariable
from Orange.preprocess import Continuize, Normalize, RemoveNaNColumns, SklImpute
from Orange.preprocess.score import LearnerScorer
from Orange.regression import Learner, Model, SklLearner, SklModel
from Orange import options


__all__ = ["LinearRegressionLearner", "RidgeRegressionLearner",
           "LassoRegressionLearner", "SGDRegressionLearner",
           "ElasticNetLearner", "ElasticNetCVLearner",
           "PolynomialLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data):
        data = Normalize(data)
        model = self(data)
        return np.abs(model.coefficients)


class LinearRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_linear_model.LinearRegression
    name = 'linreg'
    verbose_name = 'Linear'

    def fit(self, X, Y, W=None):
        model = super().fit(X, Y, W)
        return LinearModel(model.skl_model)


class LearnerOptions:
    l1_ratio = options.RatioOption('l1_ratio', default=.5, step=0.05,
                                   left_label='L1', right_label='L2')

    alpha = options.FloatOption('alpha', default=1., range=(0., 1000.), step=.1)
    tol = options.FloatOption('tol', default=1e-4, range=(1e-8, .1), step=1e-5)
    random_state = options.DisableableOption('random_state',
                                             option=options.IntegerOption(),
                                             disable_value=None)


class RidgeRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.Ridge
    name = 'ridge'
    verbose_name = 'Ridge'

    options = [
        LearnerOptions.alpha,
        LearnerOptions.tol,
        LearnerOptions.random_state,
    ]


class LassoRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.Lasso
    name = 'lasso'
    verbose_name = 'Lasso'

    options = RidgeRegressionLearner.options


class ElasticNetLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.ElasticNet
    name = 'elastic'
    verbose_name = 'ElasticNet'

    options = [
        LearnerOptions.l1_ratio,
    ] + RidgeRegressionLearner.options


class ElasticNetCVLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.ElasticNetCV
    name = 'elasticCV'

    def __init__(self, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=0.0001, cv=None, copy_X=True,
                 verbose=0, n_jobs=1, positive=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SGDRegressionLearner(LinearRegressionLearner):
    __wraps__ = skl_linear_model.SGDRegressor
    name = 'sgd'

    LOSSES = (
        options.Choice('squared_loss'),
        options.Choice('huber', related_options=('epsilon',)),
        options.Choice('epsilon_insensitive', related_options=('epsilon',)),
        options.Choice('squared_epsilon_insensitive', related_options=('epsilon',)),
    )

    PENALTIES = (
        options.Choice('none'),
        options.Choice('l1', related_options=('alpha', )),
        options.Choice('l2', related_options=('alpha', )),
        options.Choice('elasticnet', related_options=('alpha', 'l1_ratio')),
    )

    RATES = (
        options.Choice('constant', related_options=('eta0', ), label='eta0'),
        options.Choice('invscaling', verbose_name='Inverse scaling',
                       related_options=('eta0', 'power_t'), label='eta0/(t^p)'),
    )

    options = [
        LearnerOptions.alpha,
        LearnerOptions.l1_ratio,
        options.FloatOption('epsilon', default=.1, range=(0., 10.)),
        options.FloatOption('eta0', default=.01, verbose_name='eta0',
                            range=(1e-5, 1.), step=0.05),
        options.FloatOption('power_t', verbose_name='p',
                            default=.01, range=(0., 1.), step=0.05),

        options.ChoiceOption('loss', choices=LOSSES),
        options.ChoiceOption('penalty', choices=PENALTIES),
        options.ChoiceOption('learning_rate', choices=RATES),
        options.IntegerOption('n_iter', verbose_name='Number of iterations',
                              default=5, range=(1, 10 ** 4), step=10),
    ]

    def fit(self, X, Y, W=None):
        sk = self.instance
        clf = skl_pipeline.Pipeline(
            [('scaler', skl_preprocessing.StandardScaler()), ('sgd', sk)])
        clf.fit(X, Y.ravel())
        return LinearModel(clf)

    class GUI:
        main = (
            options.ChoiceGroup('loss', ('epsilon',)),
            options.ChoiceGroup('penalty', ('alpha', 'l1_ratio')),
            options.ChoiceGroup('learning_rate', ('eta0', 'power_t')),
            'n_iter'
        )


class PolynomialLearner(Learner):
    name = 'poly learner'
    preprocessors = [Continuize(),
                     RemoveNaNColumns(),
                     SklImpute()]

    def __init__(self, learner, degree=1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.degree = degree
        self.learner = learner

    def fit(self, X, Y, W=None):
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
