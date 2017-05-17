from Orange.classification import KNNLearner as KNNClassification
from Orange.modelling import SklFitter
from Orange.regression import KNNRegressionLearner

__all__ = ['KNNLearner']


class KNNLearner(SklFitter):
    __fits__ = {'classification': KNNClassification,
                'regression': KNNRegressionLearner}
