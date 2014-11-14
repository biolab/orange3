import unittest
import numpy as np

from Orange import data
import Orange.regression.mean as mean_


class MeanTest(unittest.TestCase):
    def test_mean(self):
        nrows = 1000
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(0, 4, (nrows, 1)) / 3.0
        t = data.Table(x, y)
        learn = mean_.MeanFitter()
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
        t = data.Table(x, y, W=w)
        learn = mean_.MeanFitter()
        clf = learn(t)

        expected_mean = np.average(y, weights=w)
        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue(np.allclose(y2, expected_mean))

    def test_empty(self):
        autompg = data.Table('auto-mpg')
        learn = mean_.MeanFitter()
        clf = learn(autompg[:0])
        y = clf(autompg[0])
        self.assertTrue(y == 0)

    def test_discrete(self):
        iris = data.Table('iris')
        learn = mean_.MeanFitter()
        self.assertRaises(ValueError, learn, iris)
