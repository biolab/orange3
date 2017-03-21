from Orange.classification import (
    SVMLearner as SVCLearner,
    LinearSVMLearner as LinearSVCLearner,
    NuSVMLearner as NuSVCLearner,
)
from Orange.modelling import Fitter
from Orange.regression import SVRLearner, LinearSVRLearner, NuSVRLearner

__all__ = ['SVMLearner', 'LinearSVMLearner', 'NuSVMLearner']


class SVMLearner(Fitter):
    __fits__ = {'classification': SVCLearner, 'regression': SVRLearner}


class LinearSVMLearner(Fitter):
    __fits__ = {'classification': LinearSVCLearner, 'regression': LinearSVRLearner}


class NuSVMLearner(Fitter):
    __fits__ = {'classification': NuSVCLearner, 'regression': NuSVRLearner}
