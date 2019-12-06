# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table, ContinuousVariable, Domain
from Orange.regression import LinearRegressionLearner, PolynomialLearner
from Orange.evaluation import TestOnTrainingData, RMSE

class TestPolynomialLearner(unittest.TestCase):
    def test_PolynomialLearner(self):
        x = np.array([0.172, 0.167, 0.337, 0.420, 0.355, 0.710, 0.801, 0.876])
        y = np.array([0.784, 0.746, 0.345, 0.363, 0.366, 0.833, 0.490, 0.445])

        data = Table.from_numpy(None, x.reshape(-1, 1), y)
        data.domain = Domain([ContinuousVariable('x')],
                             class_vars=[ContinuousVariable('y')])

        linear = LinearRegressionLearner()
        polynomial2 = PolynomialLearner(linear, degree=2)
        polynomial3 = PolynomialLearner(linear, degree=3)

        tt = TestOnTrainingData()
        res = tt(data, [linear, polynomial2, polynomial3])
        rmse = RMSE(res)

        self.assertGreater(rmse[0], rmse[1])
        self.assertGreater(rmse[1], rmse[2])
