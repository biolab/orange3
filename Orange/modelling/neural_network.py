from Orange.classification import NNClassificationLearner
from Orange.modelling import Fitter
from Orange.regression import NNRegressionLearner


class NNLearner(Fitter):
    __fits__ = {'classification': NNClassificationLearner,
                'regression': NNRegressionLearner}
