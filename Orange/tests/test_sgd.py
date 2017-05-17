# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table
from Orange.classification import SGDClassificationLearner
from Orange.regression import SGDRegressionLearner
from Orange.evaluation import CrossValidation, RMSE, AUC


class TestSGDRegressionLearner(unittest.TestCase):
    def test_SGDRegression(self):
        nrows, ncols = 500, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Table(X, y)
        sgd = SGDRegressionLearner()
        res = CrossValidation(data, [sgd], k=3)
        self.assertLess(RMSE(res)[0], 0.1)

    def test_coefficients(self):
        lrn = SGDRegressionLearner()
        mod = lrn(Table("housing"))
        self.assertEqual(len(mod.coefficients), len(mod.domain.attributes))


class TestSGDClassificationLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')

    def test_SGDClassification(self):
        sgd = SGDClassificationLearner()
        res = CrossValidation(self.iris, [sgd], k=3)
        self.assertGreater(AUC(res)[0], 0.8)

    def test_coefficients(self):
        lrn = SGDClassificationLearner()
        mod = lrn(self.iris)
        self.assertEqual(len(mod.coefficients[0]), len(mod.domain.attributes))
