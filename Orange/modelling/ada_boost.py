from Orange.base import SklModel
from Orange.ensembles import (
    SklAdaBoostClassificationLearner, SklAdaBoostRegressionLearner
)
from Orange.modelling import SklFitter

__all__ = ['SklAdaBoostLearner']


class SklAdaBoostLearner(SklFitter):
    __fits__ = {'classification': SklAdaBoostClassificationLearner,
                'regression': SklAdaBoostRegressionLearner}

    __returns__ = SklModel
