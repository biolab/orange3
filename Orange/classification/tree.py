import sklearn.tree as skl_tree
from Orange.classification import SklLearner, SklModel
from Orange.preprocess import Continuize, RemoveNaNColumns, SklImpute

__all__ = ["TreeLearner"]


class TreeClassifier(SklModel):
    pass


class TreeLearner(SklLearner):
    __wraps__ = skl_tree.DecisionTreeClassifier
    __returns__ = TreeClassifier
    name = 'tree'
    preprocessors = [RemoveNaNColumns(),
                     SklImpute(),
                     Continuize()]

    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_weights = True
