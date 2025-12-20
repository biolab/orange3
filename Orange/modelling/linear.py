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
        return (np.atleast_2d(np.abs(model.skl_model.coef_)).mean(0),
                model.domain.attributes)


class SGDLearner(SklFitter, _FeatureScorerMixin):
    name = 'sgd'

    __fits__ = {'classification': SGDClassificationLearner,
                'regression': SGDRegressionLearner}

    def _change_kwargs(self, kwargs, problem_type):
        pref = "classification" if problem_type is self.CLASSIFICATION else "regression"
        return kwargs | {
            attr: kwargs[pattr]
            for attr, pattr in ((attr, f"{pref}_{attr}")
                                for attr in ('loss', 'epsilon'))
            if pattr in kwargs}
