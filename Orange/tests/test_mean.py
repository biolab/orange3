import unittest
import numpy as np

from Orange.data import Table
from Orange.regression import MeanLearner


class MeanTest(unittest.TestCase):
    def test_mean(self):
        nrows = 1000
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(0, 4, (nrows, 1)) / 3.0
        t = Table(x, y)
        learn = MeanLearner()
        clf = learn(t)

        true_mean = np.average(y)
        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue(np.allclose(y2, true_mean))

    def test_weights(self):
        nrows = 100
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(0, 4, (nrows, 1)) / 3.0
        heavy = 1
        w = ((y == heavy) * 123 + 1.0 ) / 124.0
        t = Table(x, y, W=w)
        learn = MeanLearner()
        clf = learn(t)

        expected_mean = np.average(y, weights=w)
        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue(np.allclose(y2, expected_mean))

    def test_empty(self):
        autompg = Table('auto-mpg')
        learn = MeanLearner()
        clf = learn(autompg[:0])
        y = clf(autompg[0])
        self.assertTrue(y == 0)

    def test_discrete(self):
        iris = Table('iris')
        learn = MeanLearner()
        self.assertRaises(ValueError, learn, iris)
