from Orange.base import RandomForest, Model
from Orange.classification import RandomForestLearner as RFClassification
from Orange.modelling import Fitter, LearnerTypes
from Orange.regression import RandomForestRegressionLearner as RFRegression


class RandomForestModel(Model, RandomForest):
    pass


class RandomForestLearner(Fitter):
    name = 'random forest'

    __fits__ = LearnerTypes(classification=RFClassification,
                            regression=RFRegression)

    __returns__ = RandomForestModel
