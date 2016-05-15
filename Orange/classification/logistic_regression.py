import numpy as np

import sklearn.linear_model as skl_linear_model

from Orange.classification import SklLearner, SklModel
from Orange.preprocess import Normalize
from Orange.preprocess.score import LearnerScorer
from Orange.data import Variable, DiscreteVariable
from Orange import options

__all__ = ["LogisticRegressionLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data):
        data = Normalize(data)
        model = self(data)
        # Take the maximum attribute score across all classes
        return np.max(np.abs(model.coefficients), axis=0)


class LogisticRegressionClassifier(SklModel):
    @property
    def intercept(self):
        return self.skl_model.intercept_

    @property
    def coefficients(self):
        return self.skl_model.coef_


class LogisticRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_linear_model.LogisticRegression
    __returns__ = LogisticRegressionClassifier
    name = 'logreg'
    preprocessors = SklLearner.preprocessors

    PENALTIES = (('l1', 'Lasso (L1)'), ('l2', 'Ridge (L2)'))
    options = [
        options.ChoiceOption('penalty', verbose_name='Regularization',
                             choices=PENALTIES, default='l2'),
        options.FloatOption('C', default=1., verbose_name='C',
                            range=(.0001, 1000.), step=.001)
    ]
