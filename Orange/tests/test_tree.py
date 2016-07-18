# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
import sklearn.tree as skl_tree
from sklearn.tree._tree import TREE_LEAF, Tree

from Orange.data import Table
from Orange.classification import TreeLearner
from Orange.regression import TreeRegressionLearner


class TestTreeLearner(unittest.TestCase):
    def test_classification(self):
        table = Table('iris')
        learn = TreeLearner()
        clf = learn(table)
        Z = clf(table)
        self.assertTrue(np.all(table.Y.flatten() == Z))

    def test_regression(self):
        table = Table('housing')
        learn = TreeRegressionLearner()
        model = learn(table)
        pred = model(table)
        self.assertTrue(np.all(table.Y.flatten() == pred))

    def test_get_tree_classification(self):
        table = Table('iris')
        learn = TreeLearner()
        clf = learn(table)
        self.assertIsInstance(clf.tree, Tree)

    def test_get_tree_regression(self):
        table = Table('housing')
        learn = TreeRegressionLearner()
        clf = learn(table)
        self.assertIsInstance(clf.tree, Tree)


class TestDecisionTreeClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')

    def test_full_tree(self):
        table = self.iris
        clf = skl_tree.DecisionTreeClassifier()
        clf = clf.fit(table.X, table.Y)
        Z = clf.predict(table.X)
        self.assertTrue(np.all(table.Y.flatten() == Z))

    def test_min_samples_split(self):
        table = self.iris
        lim = 5
        clf = skl_tree.DecisionTreeClassifier(min_samples_split=lim)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        for i in range(t.node_count):
            if t.children_left[i] != TREE_LEAF:
                self.assertGreaterEqual(t.n_node_samples[i], lim)

    def test_min_samples_leaf(self):
        table = self.iris
        lim = 5
        clf = skl_tree.DecisionTreeClassifier(min_samples_leaf=lim)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        for i in range(t.node_count):
            if t.children_left[i] == TREE_LEAF:
                self.assertGreaterEqual(t.n_node_samples[i], lim)

    def test_max_leaf_nodes(self):
        table = self.iris
        lim = 5
        clf = skl_tree.DecisionTreeClassifier(max_leaf_nodes=lim)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        self.assertLessEqual(t.node_count, lim * 2 - 1)

    def test_criterion(self):
        table = self.iris
        clf = skl_tree.DecisionTreeClassifier(criterion="entropy")
        clf = clf.fit(table.X, table.Y)

    def test_splitter(self):
        table = self.iris
        clf = skl_tree.DecisionTreeClassifier(splitter="random")
        clf = clf.fit(table.X, table.Y)

    def test_weights(self):
        table = self.iris
        clf = skl_tree.DecisionTreeClassifier(max_depth=2)
        clf = clf.fit(table.X, table.Y)
        clfw = skl_tree.DecisionTreeClassifier(max_depth=2)
        clfw = clfw.fit(table.X, table.Y, sample_weight=np.arange(len(table)))
        self.assertFalse(len(clf.tree_.feature) == len(clfw.tree_.feature) and
                         np.all(clf.tree_.feature == clfw.tree_.feature))

    def test_impurity(self):
        table = self.iris
        clf = skl_tree.DecisionTreeClassifier()
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_
        for i in range(t.node_count):
            if t.children_left[i] == TREE_LEAF:
                self.assertEqual(t.impurity[i], 0)
            else:
                l, r = t.children_left[i], t.children_right[i]
                child_impurity = min(t.impurity[l], t.impurity[r])
                self.assertLessEqual(child_impurity, t.impurity[i])

    def test_navigate_tree(self):
        table = self.iris
        clf = skl_tree.DecisionTreeClassifier(max_depth=1)
        clf = clf.fit(table.X, table.Y)
        t = clf.tree_

        x = table.X[0]
        if x[t.feature[0]] <= t.threshold[0]:
            v = t.value[t.children_left[0]][0]
        else:
            v = t.value[t.children_right[0]][0]
        self.assertEqual(np.argmax(v), clf.predict(table.X[0]))
