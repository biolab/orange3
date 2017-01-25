from Orange.base import RandomForestModel
from Orange.classification import RandomForestLearner as RFClassification
from Orange.modelling import Fitter
from Orange.regression import RandomForestRegressionLearner as RFRegression


class RandomForestLearner(Fitter):
    name = 'random forest'

    __fits__ = {'classification': RFClassification,
                'regression': RFRegression}

    __returns__ = RandomForestModel
