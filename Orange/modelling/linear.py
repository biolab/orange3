from Orange.classification.sgd import SGDClassificationLearner
from Orange.modelling import Fitter
from Orange.regression import SGDRegressionLearner


class SGDLearner(Fitter):
    name = 'sgd'

    __fits__ = {'classification': SGDClassificationLearner,
                'regression': SGDRegressionLearner}
