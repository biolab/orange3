from Orange.base import Model
from Orange.classification import KNNLearner
from Orange.modelling.fitter import Fitter, LearnerTypes
from Orange.regression import KNNRegressionLearner


class KNNLearner(Fitter):
    name = 'knn'

    __fits__ = LearnerTypes(classification=KNNLearner,
                            regression=KNNRegressionLearner)

    __returns__ = Model
