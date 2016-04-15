import sklearn.tree as skl_tree
from Orange.regression import SklLearner, SklModel
from Orange.preprocess import Continuize, RemoveNaNColumns, SklImpute

__all__ = ["TreeRegressionLearner"]


class TreeRegressor(SklModel):
    pass


class TreeRegressionLearner(SklLearner):
    __wraps__ = skl_tree.DecisionTreeRegressor
    __returns__ = TreeRegressor
    name = 'regression tree'
    preprocessors = [RemoveNaNColumns(),
                     SklImpute(),
                     Continuize()]

    """
    criterion = The function to measure the quality of a split.
    splitter = The strategy used to choose the split at each node.
    max_depth = The maximum depth of the tree.
    min_samples_split = The minimum number of samples required to split an internal node.
    min_samples_leaf = The minimum number of samples required to be at a leaf node.
    max_features = The number of features to consider when looking for the best split:
                    If int, then consider max_features features at each split.
                    If float, then max_features is a percentage and int(max_features * n_features)
                        features are considered at each split.
                    If “auto”, then max_features=sqrt(n_features).
                    If “sqrt”, then max_features=sqrt(n_features).
                    If “log2”, then max_features=log2(n_features).
                    If None, then max_features=n_features.
    random_state = If int, random_state is the seed used by the random number generator;
    max_leaf_nodes = Grow a tree with max_leaf_nodes in best-first fashion.
    """

    def __init__(self, criterion="mse", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_weights = True
