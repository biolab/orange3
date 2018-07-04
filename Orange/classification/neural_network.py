import sklearn.neural_network as skl_nn
from Orange.base import NNBase
from Orange.classification import SklLearner

__all__ = ["NNClassificationLearner"]


class NIterCallbackMixin:
    orange_callback = None

    @property
    def n_iter_(self):
        return self.__orange_n_iter

    @n_iter_.setter
    def n_iter_(self, v):
        self.__orange_n_iter = v
        if self.orange_callback:
            self.orange_callback(v)


class MLPClassifierWCallback(skl_nn.MLPClassifier, NIterCallbackMixin):
    pass


class NNClassificationLearner(NNBase, SklLearner):
    __wraps__ = MLPClassifierWCallback

    def _initialize_wrapped(self):
        clf = SklLearner._initialize_wrapped(self)
        clf.orange_callback = getattr(self, "callback", None)
        return clf
