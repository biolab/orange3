import unittest
import numpy as np
import Orange.data
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
import Orange.classification.tree
from collections import Counter

class TreeTest(unittest.TestCase):

    def test_classification(self):
        table = Orange.data.Table('iris')
        learn = Orange.classification.tree.ClassificationTreeLearner()
        clf = learn(table)
        Z = clf(table)
        self.assertTrue(np.all(table.Y.flatten() == Z))

    def test_items_in_nodes(self):
        table = Orange.data.Table('iris')
        learn = Orange.classification.tree.ClassificationTreeLearner()
        clf = learn(table)
        self.assertTrue(len(clf.get_items(0))==len(table))

    def test_distr_in_nodes(self):
        table = Orange.data.Table('iris')
        learn = Orange.classification.tree.ClassificationTreeLearner()
        clf = learn(table)
        self.assertTrue(clf.get_distr(0)==Counter(table.Y.flatten()))

class SklearnTreeTest(unittest.TestCase):

    def test_full_tree(self):
        table = Orange.data.Table('iris')
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(table.X, table.Y)
        Z = clf.predict(table.X)
        self.assertTrue(np.all(table.Y.flatten() == Z))

    def test_min_samples_split(self):
        table = Orange.data.Table('iris')
        lim = 5
        clf = tree.DecisionTreeClassifier(min_samples_split=lim)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        for i in range(t.node_count):
            if t.children_left[i] != TREE_LEAF:
                self.assertTrue(t.n_node_samples[i] >= lim)

    def test_min_samples_leaf(self):
        table = Orange.data.Table('iris')
        lim = 5
        clf = tree.DecisionTreeClassifier(min_samples_leaf=lim)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        for i in range(t.node_count):
            if t.children_left[i] == TREE_LEAF:
                self.assertTrue(t.n_node_samples[i] >= lim)

    def test_max_leaf_nodes(self):
        table = Orange.data.Table('iris')
        lim = 5
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=lim)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        self.assertTrue(t.node_count <= lim * 2 - 1)

    def test_criterion(self):
        table = Orange.data.Table('iris')
        clf = tree.DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(table.X, table.Y)

    def test_splitter(self):
        table = Orange.data.Table('iris')
        clf = tree.DecisionTreeClassifier(splitter="random")
        clf = clf.fit(table.X, table.Y)

    def test_weights(self):
        table = Orange.data.Table('iris')
        clf = tree.DecisionTreeClassifier(max_depth=2)
        clf = clf.fit(table.X, table.Y)
        clfw = tree.DecisionTreeClassifier(max_depth=2)
        clfw = clfw.fit(table.X, table.Y, sample_weight=np.arange(len(table)))
        self.assertFalse(len(clf.tree_.feature) == len(clfw.tree_.feature) and
                         np.all(clf.tree_.feature == clfw.tree_.feature))

    def test_impurity(self):
        table = Orange.data.Table('iris')
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        for i in range(t.node_count):
            if t.children_left[i] == TREE_LEAF:
                self.assertTrue(t.impurity[i] == 0)
            else:
                l, r = t.children_left[i], t.children_right[i]
                child_impurity = min(t.impurity[l], t.impurity[r])
                self.assertTrue(child_impurity <= t.impurity[i])

    def test_navigate_tree(self):
        table = Orange.data.Table('iris')
        clf = tree.DecisionTreeClassifier(max_depth=1)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_

        x = table.X[0]
        if x[t.feature[0]] <= t.threshold[0]:
            v = t.value[t.children_left[0]][0]
        else:
            v = t.value[t.children_right[0]][0]
        self.assertTrue(np.argmax(v) == clf.predict(table.X[0]))
