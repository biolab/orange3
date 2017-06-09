import numpy as np

from Orange.classification.sgd import SGDClassificationLearner
from Orange.data import Variable
from Orange.modelling import SklFitter
from Orange.preprocess.score import LearnerScorer
from Orange.regression import SGDRegressionLearner

__all__ = ['SGDLearner']


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = Variable

    def score(self, data):
        model = self.get_learner(data)(data)
        return np.atleast_2d(np.abs(model.skl_model.coef_)).mean(0)


class SGDLearner(SklFitter, _FeatureScorerMixin):
    name = 'sgd'

    __fits__ = {'classification': SGDClassificationLearner,
                'regression': SGDRegressionLearner}

    def _change_kwargs(self, kwargs, problem_type):
        if problem_type is self.CLASSIFICATION:
            kwargs['loss'] = kwargs.get('classification_loss')
            kwargs['epsilon'] = kwargs.get('classification_epsilon')
        elif problem_type is self.REGRESSION:
            kwargs['loss'] = kwargs.get('regression_loss')
            kwargs['epsilon'] = kwargs.get('regression_epsilon')
        return kwargs
