from Orange.classification.sgd import SGDClassificationLearner
from Orange.modelling import Fitter
from Orange.regression import SGDRegressionLearner


class SGDLearner(Fitter):
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
