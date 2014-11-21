from Orange.classification import SklFitter
from sklearn.tree import DecisionTreeClassifier


class ClassificationTreeLearner(SklFitter):
    __wraps__ = DecisionTreeClassifier

    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None):
        self.params = vars()
        self.supports_weights = True
