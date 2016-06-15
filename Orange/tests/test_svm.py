# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.classification import (SVMLearner, LinearSVMLearner,
                                   NuSVMLearner, OneClassSVMLearner)
from Orange.regression import (SVRLearner, NuSVRLearner)
from Orange.data import Table, Domain, ContinuousVariable
from Orange.evaluation import CrossValidation, CA, RMSE


class TestSVMLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table('ionosphere')
        cls.data.shuffle()

    def test_SVM(self):
        learn = SVMLearner()
        res = CrossValidation(self.data, [learn], k=2)
        self.assertGreater(CA(res)[0], 0.9)

    def test_LinearSVM(self):
        learn = LinearSVMLearner()
        res = CrossValidation(self.data, [learn], k=2)
        self.assertGreater(CA(res)[0], 0.8)
        self.assertLess(CA(res)[0], 0.9)

    def test_NuSVM(self):
        learn = NuSVMLearner(nu=0.01)
        res = CrossValidation(self.data, [learn], k=2)
        self.assertGreater(CA(res)[0], 0.9)

    def test_SVR(self):
        nrows, ncols = 200, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Table(X, y)
        learn = SVRLearner(kernel='rbf', gamma=0.1)
        res = CrossValidation(data, [learn], k=2)
        self.assertLess(RMSE(res)[0], 0.15)

    def test_NuSVR(self):
        nrows, ncols = 200, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Table(X, y)
        learn = NuSVRLearner(kernel='rbf', gamma=0.1)
        res = CrossValidation(data, [learn], k=2)
        self.assertLess(RMSE(res)[0], 0.1)

    def test_OneClassSVM(self):
        np.random.seed(42)
        domain = Domain((ContinuousVariable("c1"), ContinuousVariable("c2")))
        X_in = 0.3 * np.random.randn(40, 2)
        X_out = np.random.uniform(low=-4, high=4, size=(20, 2))
        X_all = Table(domain, np.r_[X_in + 2, X_in - 2, X_out])
        n_true_in = len(X_in) * 2
        n_true_out = len(X_out)

        nu = 0.2
        learner = OneClassSVMLearner(nu=nu)
        cls = learner(X_all)
        y_pred = cls(X_all)
        n_pred_out_all = np.sum(y_pred == -1)
        n_pred_in_true_in = np.sum(y_pred[:n_true_in] == 1)
        n_pred_out_true_out = np.sum(y_pred[- n_true_out:] == -1)

        self.assertEqual(np.absolute(y_pred).all(), 1)
        self.assertLessEqual(n_pred_out_all, len(X_all) * nu)
        self.assertLess(np.absolute(n_pred_out_all - n_true_out), 2)
        self.assertLess(np.absolute(n_pred_in_true_in - n_true_in), 4)
        self.assertLess(np.absolute(n_pred_out_true_out - n_true_out), 3)
