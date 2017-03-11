import sklearn.neural_network as skl_nn
from Orange.base import NNBase
from Orange.regression import SklLearner

__all__ = ["NNRegressionLearner"]


class NNRegressionLearner(NNBase, SklLearner):
    __wraps__ = skl_nn.MLPRegressor
