# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table
from Orange.regression import MeanLearner
from Orange.tests import test_filename


class TestMeanLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learn = MeanLearner()

    def test_mean(self):
        nrows = 1000
        ncols = 10
        x = np.random.randint(1, 4, (nrows, ncols))
        y = np.random.randint(0, 5, (nrows, 1)) / 3.0
        t = Table.from_numpy(None, x, y)
        clf = self.learn(t)

        true_mean = np.average(y)
        x2 = np.random.randint(1, 4, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue(np.allclose(y2, true_mean))

    def test_weights(self):
        nrows = 100
        ncols = 10
        x = np.random.randint(1, 4, (nrows, ncols))
        y = np.random.randint(0, 5, (nrows, 1)) / 3.0
        heavy = 1
        w = ((y == heavy) * 123 + 1.0) / 124.0
        t = Table.from_numpy(None, x, y, W=w)
        clf = self.learn(t)

        expected_mean = np.average(y, weights=w)
        x2 = np.random.randint(1, 4, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue(np.allclose(y2, expected_mean))

    def test_empty(self):
        autompg = Table(test_filename('datasets/imports-85.tab'))
        clf = self.learn(autompg[:0])
        y = clf(autompg[0])
        self.assertEqual(y, 0)

    def test_discrete(self):
        iris = Table('iris')
        self.assertRaises(ValueError, self.learn, iris)
