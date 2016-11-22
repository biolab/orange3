# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table
from Orange.classification import MajorityLearner


class TestMajorityLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.learn = MajorityLearner()

    def test_majority(self):
        nrows = 1000
        ncols = 10
        x = np.random.randint(1, 4, (nrows, ncols))
        y = np.random.randint(1, 4, (nrows, 1)) // 2
        t = Table(x, y)
        clf = self.learn(t)

        x2 = np.random.randint(1, 4, (nrows, ncols))
        y2 = clf(x2)
        self.assertEqual(y2.all(), 1)

    def test_weights(self):
        nrows = 100
        ncols = 10
        x = np.random.randint(1, 5, (nrows, ncols))
        y = np.array(70*[0] + 30*[1]).reshape((nrows, 1))
        heavy_class = 1
        w = (y == heavy_class) * 2 + 1
        t = Table(x, y, W=w)
        clf = self.learn(t)

        y2 = clf(x)
        self.assertEqual(y2.all(), heavy_class)

    def test_empty(self):
        clf = self.learn(self.iris[:0])
        y = clf(self.iris[0], clf.Probs)
        self.assertTrue(np.allclose(y, y.sum() / y.size))

    def test_missing(self):
        iris = Table('iris')
        learn = MajorityLearner()
        for e in iris[: len(iris) // 2: 2]:
            e.set_class("?")
        clf = learn(iris)
        y = clf(iris)
        self.assertTrue((y == 2).all())

        for e in iris:
            e.set_class("?")
        clf = learn(iris)
        y = clf(iris)
        self.assertEqual(y.all(), 1)

    def test_continuous(self):
        autompg = Table('auto-mpg')
        learn = MajorityLearner()
        self.assertRaises(ValueError, learn, autompg)

    def test_returns_random_class(self):
        iris = self.iris
        train = np.ones((150,), dtype='bool')
        train[0] = False
        majority = MajorityLearner()(iris[train])
        pred1 = majority(iris[0])
        self.assertIn(pred1, [1, 2])

        for i in range(1, 50):
            train[i] = train[50 + i] = train[100 + i] = False
            majority = MajorityLearner()(iris[train])
            pred2 = majority(iris[0])
            self.assertIn(pred2, [1, 2])
            if pred1 != pred2:
                break
        else:
            self.fail("Majority always returns the same value.")
