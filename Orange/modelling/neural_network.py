from Orange.classification import NNClassificationLearner
from Orange.modelling import SklFitter
from Orange.regression import NNRegressionLearner

__all__ = ['NNLearner']


class NNLearner(SklFitter):
    __fits__ = {'classification': NNClassificationLearner,
                'regression': NNRegressionLearner}
