from Orange.classification import SVMLearner, LinearSVMLearner, NuSVMLearner
from Orange.modelling import Fitter
from Orange.regression import SVRLearner, LinearSVRLearner, NuSVRLearner


class SVMFitter(Fitter):
    __fits__ = {'classification': SVMLearner, 'regression': SVRLearner}


class LinearSVMFitter(Fitter):
    __fits__ = {'classification': LinearSVMLearner, 'regression': LinearSVRLearner}


class NuSVMFitter(Fitter):
    __fits__ = {'classification': NuSVMLearner, 'regression': NuSVRLearner}
