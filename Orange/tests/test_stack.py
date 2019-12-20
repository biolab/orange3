import unittest

from Orange.data import Table
from Orange.ensembles.stack import StackedFitter
from Orange.evaluation import CA, CrossValidation, MSE
from Orange.modelling import KNNLearner, TreeLearner


class TestStackedFitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')

    def test_classification(self):
        sf = StackedFitter([TreeLearner(), KNNLearner()])
        cv = CrossValidation(k=3)
        results = cv(self.iris, [sf])
        ca = CA(results)
        self.assertGreater(ca, 0.9)

    def test_regression(self):
        sf = StackedFitter([TreeLearner(), KNNLearner()])
        cv = CrossValidation(k=3, random_state=0)
        results = cv(self.housing[:50], [sf, TreeLearner(), KNNLearner()])
        mse = MSE()(results)
        self.assertLess(mse[0], mse[1])
        self.assertLess(mse[0], mse[2])
