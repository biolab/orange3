# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
from Orange.regression import SGDRegressionLearner
from Orange.evaluation import CrossValidation, RMSE
from Orange.data import Table


class SGDRegressionTest(unittest.TestCase):
    def test_SGDRegression(self):
        nrows, ncols = 500, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Table(X, y)
        sgd = SGDRegressionLearner()
        res = CrossValidation(data, [sgd], k=3)
        self.assertLess(RMSE(res)[0], 0.1)
