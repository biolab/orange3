from Orange.classification import MajorityLearner
from Orange.modelling import Fitter
from Orange.regression import MeanLearner


class ConstantLearner(Fitter):
    __fits__ = {'classification': MajorityLearner, 'regression': MeanLearner}
