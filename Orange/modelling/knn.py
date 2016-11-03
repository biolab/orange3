from Orange.base import LearnerTypes, LearnerDispatcher, Model
from Orange.classification import KNNLearner
from Orange.regression import KNNRegressionLearner


class KNNLearner(LearnerDispatcher):
    name = 'knn'

    __dispatches__ = LearnerTypes(classification=KNNLearner,
                                  regression=KNNRegressionLearner)

    __returns__ = Model
