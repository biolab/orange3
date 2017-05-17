import sklearn.neural_network as skl_nn
from Orange.base import NNBase
from Orange.classification import SklLearner

__all__ = ["NNClassificationLearner"]


class NNClassificationLearner(NNBase, SklLearner):
    __wraps__ = skl_nn.MLPClassifier
