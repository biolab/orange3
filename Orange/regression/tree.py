import sklearn.tree as skl_tree

from Orange.classification import TreeLearner
from Orange.regression import SklLearner, SklModel
from Orange.preprocess import Continuize, RemoveNaNColumns, SklImpute
from Orange import options

__all__ = ["TreeRegressionLearner"]


class TreeRegressor(SklModel):
    pass


class TreeRegressionLearner(SklLearner):
    __wraps__ = skl_tree.DecisionTreeRegressor
    __returns__ = TreeRegressor
    name = 'regression tree'
    verbose_name = 'Decision Tree Regressor'
    preprocessors = [RemoveNaNColumns(),
                     SklImpute(),
                     Continuize()]

    CRITERIONS = (('mse', 'Mean square error'), )

    options = [
        options.ChoiceOption('criterion', choices=CRITERIONS),
    ] + TreeLearner.options[1:]

    GUI = TreeLearner.GUI
