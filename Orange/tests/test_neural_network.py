# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.data import Table
from Orange.classification import NNClassificationLearner
from Orange.modelling import NNLearner, ConstantLearner
from Orange.regression import NNRegressionLearner
from Orange.evaluation import CA, CrossValidation, MSE


class TestNNLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')
        cls.learner = NNLearner()

    def test_NN_classification(self):
        results = CrossValidation(self.iris, [NNClassificationLearner()], k=3)
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)

    def test_NN_regression(self):
        const = ConstantLearner()
        results = CrossValidation(self.housing, [NNRegressionLearner(), const],
                                  k=3)
        mse = MSE()
        res = mse(results)
        self.assertLess(res[0], 35)
        self.assertLess(res[0], res[1])

    def test_NN_model(self):
        results = CrossValidation(self.iris, [self.learner], k=3)
        self.assertGreater(CA(results), 0.90)
        results = CrossValidation(self.housing, [self.learner], k=3)
        mse = MSE()
        res = mse(results)
        self.assertLess(res[0], 35)

    def test_NN_classification_predict_single_instance(self):
        lrn = NNClassificationLearner()
        clf = lrn(self.iris)
        for ins in self.iris[::20]:
            clf(ins)
            _, _ = clf(ins, clf.ValueProbs)

    def test_NN_regression_predict_single_instance(self):
        lrn = NNRegressionLearner()
        clf = lrn(self.housing)
        for ins in self.housing[::20]:
            clf(ins)
