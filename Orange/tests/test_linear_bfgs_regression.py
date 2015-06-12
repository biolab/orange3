import unittest

from Orange.data import Table
from Orange.evaluation import CrossValidation, RMSE
from Orange.regression.linear_bfgs import LinearRegressionLearner


class LinearRegressionTest(unittest.TestCase):
    def test_preprocessors(self):
        table = Table('housing')
        learners = [LinearRegressionLearner(preprocessors=[]),
                    LinearRegressionLearner()]
        results = CrossValidation(table, learners, k=3)
        rmse = RMSE(results)
        self.assertTrue(rmse[0] < rmse[1])
