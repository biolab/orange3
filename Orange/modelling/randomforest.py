from Orange.base import RandomForestModel
from Orange.classification import RandomForestLearner as RFClassification
from Orange.modelling import SklFitter
from Orange.regression import RandomForestRegressionLearner as RFRegression

__all__ = ['RandomForestLearner']


class RandomForestLearner(SklFitter):
    name = 'random forest'

    __fits__ = {'classification': RFClassification,
                'regression': RFRegression}

    __returns__ = RandomForestModel
