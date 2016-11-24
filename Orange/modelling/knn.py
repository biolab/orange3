from Orange.classification import KNNLearner as KNNClassification
from Orange.modelling.fitter import Fitter, LearnerTypes
from Orange.regression import KNNRegressionLearner


class KNNLearner(Fitter):
    name = 'knn'

    __fits__ = LearnerTypes(classification=KNNClassification,
                            regression=KNNRegressionLearner)
