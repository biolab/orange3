# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table
from Orange.evaluation import CrossValidation, RMSE
from Orange.regression.linear_bfgs import LinearRegressionLearner


class TestLinearRegressionLearner(unittest.TestCase):
    def test_preprocessors(self):
        table = Table('housing')
        learners = [LinearRegressionLearner(preprocessors=[]),
                    LinearRegressionLearner()]
        results = CrossValidation(table, learners, k=3)
        rmse = RMSE(results)
        self.assertLess(rmse[0], rmse[1])
