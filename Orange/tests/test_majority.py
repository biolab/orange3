import unittest

import numpy as np

from Orange.data import Table
from Orange.classification import MajorityLearner


class MajorityTest(unittest.TestCase):
    def test_majority(self):
        nrows = 1000
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(1, 3, (nrows, 1)) // 2
        y[0] = 4
        t = Table(x, y)
        learn = MajorityLearner()
        clf = learn(t)

        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue((y2 == 1).all())

    def test_weights(self):
        nrows = 100
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(1, 5, (nrows, 1))
        heavy = 3
        w = (y == heavy) * 123 + 1
        t = Table(x, y, W=w)
        learn = MajorityLearner()
        clf = learn(t)

        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue((y2 == heavy).all())

    def test_empty(self):
        iris = Table('iris')
        learn = MajorityLearner()
        clf = learn(iris[:0])
        y = clf(iris[0], clf.Probs)
        self.assertTrue(np.allclose(y, y.sum() / y.size))

    def test_missing(self):
        iris = Table('iris')
        learn = MajorityLearner()
        for e in iris[: len(iris) // 2 : 2]:
            e.set_class("?")
        clf = learn(iris)
        y = clf(iris)
        self.assertTrue((y == 2).all())

        iris = Table('iris')
        learn = MajorityLearner()
        for e in iris:
            e.set_class("?")
        clf = learn(iris)
        y = clf(iris)
        self.assertTrue((y == 0).all())

    def test_continuous(self):
        autompg = Table('auto-mpg')
        learn = MajorityLearner()
        self.assertRaises(ValueError, learn, autompg)
