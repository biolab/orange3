from Orange.classification import SklFitter, SklModel
import sklearn.tree as skltree

import numpy as np
from collections import Counter


class ClassificationTreeLearner(SklFitter):
    __wraps__ = skltree.DecisionTreeClassifier

    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None):
        self.params = vars()
        self.supports_weights = True

    def distribute_items(self, X, Y, t, id, items):
        """Store example ids into leaves and compute class distributions."""
        if t.children_left[id] == skltree._tree.TREE_LEAF:
            self.items[id] = items
            self.distr[id] = Counter(Y[items].flatten())
        else:
            x = X[items, :]
            left = items[np.where(x[:, t.feature[id]] <= t.threshold[id])]
            right = items[np.where(x[:, t.feature[id]] > t.threshold[id])]
            self.distribute_items(X, Y, t, t.children_left[id], left)
            self.distribute_items(X, Y, t, t.children_right[id], right)
            self.distr[id] = self.distr[t.children_left[id]] + self.distr[t.children_right[id]]

    def fit(self, X, Y, W):
        clf = skltree.DecisionTreeClassifier(**self.params)
        if W is None:
            clf = clf.fit(X, Y)
        else:
            clf = clf.fit(X, Y, sample_weight=W.reshape(-1))
        t = clf.tree_
        self.items = [None]*t.node_count
        self.distr = [None]*t.node_count
        self.distribute_items(X, Y, t, 0, np.arange(len(X)))
        return ClassificationTreeClassifier(clf, self.items, self.distr)


class ClassificationTreeClassifier(SklModel):

    def __init__(self, clf, items, distr):
        super().__init__(clf)
        self.items = items
        self.distr = distr

    def get_distr(self, id):
        return self.distr[id]

    def get_items(self, id):
        """Return ids of examples belonging to node id."""
        t = self.clf.tree_
        if t.children_left[id] == skltree._tree.TREE_LEAF:
            return self.items[id]
        else:
            left = self.get_items(t.children_left[id])
            right = self.get_items(t.children_right[id])
        return np.hstack((left, right))
