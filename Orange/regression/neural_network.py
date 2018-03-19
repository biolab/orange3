import sklearn.neural_network as skl_nn
from Orange.base import NNBase
from Orange.regression import SklLearner
from Orange.classification.neural_network import NIterCallbackMixin

__all__ = ["NNRegressionLearner"]


class MLPRegressorWCallback(skl_nn.MLPRegressor, NIterCallbackMixin):
    pass


class NNRegressionLearner(NNBase, SklLearner):
    __wraps__ = MLPRegressorWCallback

    def _initialize_wrapped(self):
        clf = SklLearner._initialize_wrapped(self)
        clf.orange_callback = getattr(self, "callback", None)
        return clf
