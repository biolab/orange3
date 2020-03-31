from Orange.modelling import SklFitter
from Orange.regression import PLSRegressionLearner

__all__ = ['PLSRegressionLearner']


class PLSRegressionLearner(SklFitter):
    __fits__ = {'regression': PLSRegressionLearner}
