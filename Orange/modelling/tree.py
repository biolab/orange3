from Orange.classification import SklTreeLearner
from Orange.classification import TreeLearner as ClassificationTreeLearner
from Orange.modelling import Fitter
from Orange.regression import TreeLearner as RegressionTreeLearner
from Orange.regression.tree import SklTreeRegressionLearner
from Orange.tree import TreeModel


class SklTreeLearner(Fitter):
    name = 'tree'

    __fits__ = {'classification': SklTreeLearner,
                'regression': SklTreeRegressionLearner}


class TreeLearner(Fitter):
    name = 'tree'

    __fits__ = {'classification': ClassificationTreeLearner,
                'regression': RegressionTreeLearner}

    __returns__ = TreeModel
