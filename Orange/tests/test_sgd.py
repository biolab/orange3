import unittest

import numpy as np

import Orange


class SGDRegressionTest(unittest.TestCase):
    def test_SGDRegression(self):
        nrows, ncols = 500, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Orange.data.Table(X, y)
        sgd = Orange.regression.SGDRegressionLearner()
        res = Orange.evaluation.CrossValidation(data, [sgd], k=3)
        self.assertTrue(Orange.evaluation.RMSE(res)[0] < 0.1)
