# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.classification import KNNLearner
from Orange.regression import KNNRegressionLearner
from Orange.evaluation import CA, CrossValidation, MSE


class TestKNNLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')

    def test_KNN(self):
        cv = CrossValidation(k=3)
        results = cv(self.iris, [KNNLearner()])
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        lrn = KNNLearner()
        clf = lrn(self.iris)
        for ins in self.iris[::20]:
            clf(ins)
            val, prob = clf(ins, clf.ValueProbs)

    def test_nan(self):
        lrn1 = KNNRegressionLearner(n_neighbors=1)
        lrn3 = KNNRegressionLearner(n_neighbors=3)
        X = np.arange(1, 7)[:, None]
        Y = np.array([np.nan, np.nan, np.nan, 1, 1, 1])
        attr = (ContinuousVariable("Feat 1"),)
        class_var = (ContinuousVariable("Class"),)
        domain = Domain(attr, class_var)
        data = Table(domain, X, Y)
        clf = lrn1(data)
        predictions = clf(data)
        self.assertEqual(predictions[0], 1.0)
        clf = lrn3(data)
        predictions = clf(data)
        self.assertEqual(predictions[3], 1.0)

    def test_random(self):
        nrows, ncols = 1000, 5
        x = np.random.randint(-20, 51, (nrows, ncols))
        y = np.random.randint(0, 9, (nrows, 1))
        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        attr = (ContinuousVariable('Feature 1'),
                ContinuousVariable('Feature 2'),
                ContinuousVariable('Feature 3'),
                ContinuousVariable('Feature 4'),
                ContinuousVariable('Feature 5'))
        class_vars = (DiscreteVariable('Target 1', values=tuple("abcdefghij")),)
        domain = Domain(attr, class_vars)
        t = Table(domain, x1, y1)
        lrn = KNNLearner()
        clf = lrn(t)
        z = clf(x2)
        correct = (z == y2.flatten())
        ca = np.mean(correct)
        self.assertGreater(ca, 0.1)
        self.assertLess(ca, 0.3)

    def test_KNN_mahalanobis(self):
        learners = [KNNLearner(metric="mahalanobis")]
        cv = CrossValidation(k=3)
        results = cv(self.iris, learners)
        ca = CA(results)
        self.assertGreater(ca, 0.8)

    def test_KNN_regression(self):
        learners = [KNNRegressionLearner(),
                    KNNRegressionLearner(metric="mahalanobis")]
        cv = CrossValidation(k=3)
        results = cv(self.housing, learners)
        mse = MSE(results)
        self.assertLess(mse[1], mse[0])

    def test_supports_weights(self):
        self.assertFalse(KNNLearner().supports_weights)
        self.assertFalse(KNNRegressionLearner().supports_weights)
