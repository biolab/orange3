from Orange.classification import MajorityLearner
from Orange.modelling import Fitter
from Orange.regression import MeanLearner

__all__ = ['ConstantLearner']


class ConstantLearner(Fitter):
    __fits__ = {'classification': MajorityLearner, 'regression': MeanLearner}
