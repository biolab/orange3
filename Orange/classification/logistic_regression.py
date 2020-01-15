import warnings

import numpy as np

import sklearn.linear_model as skl_linear_model

import Orange
from Orange.classification import SklLearner, SklModel
from Orange.classification.base_classification import LinearModel
from Orange.preprocess import Normalize
from Orange.preprocess.score import LearnerScorer
from Orange.data import Variable, DiscreteVariable

__all__ = ["LogisticRegressionLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data):
        data = Normalize()(data)
        model = self(data)
        return np.abs(model.coefficients), model.domain.attributes


class LogisticRegressionClassifier(SklModel, LinearModel):
    @property
    def intercept(self):
        return self.skl_model.intercept_

    @property
    def coefficients(self):
        return self.skl_model.coef_


class LogisticRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_linear_model.LogisticRegression
    __returns__ = LogisticRegressionClassifier
    preprocessors = SklLearner.preprocessors

    def __init__(self, penalty="l2", dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, n_jobs=1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def __call__(self, data):
        if len(np.unique(data.Y)) > 1:
            return super().__call__(data)
        else:
            warnings.warn("Single class in data, returning Constant Model.")
            maj = Orange.classification.MajorityLearner()
            const = maj(data)
            return const
