import unittest

import numpy as np

import Orange


class SGDRegressionTest(unittest.TestCase):
    def test_SGDRegression(self):
        nrows = 500
        ncols = 5
        x = np.sort(10 * np.random.rand(nrows, ncols))
        y = np.sum(np.sin(x), axis=1).reshape(nrows, 1)
        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = Orange.data.Table(x1, y1)
        learn = Orange.regression.SGDRegressionLearner()
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((abs(z.reshape(-1, 1) - y2) < 4.0).all())
