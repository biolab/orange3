import unittest

import numpy as np

import Orange
from Orange.classification import (SVMLearner, LinearSVMLearner, NuSVMLearner,
                                   SVRLearner, NuSVRLearner, OneClassSVMLearner)


class SVMTest(unittest.TestCase):

    def setUp(self):
        self.data = Orange.data.Table('ionosphere')
        self.data.shuffle()

    def test_SVM(self):
        learn = SVMLearner()
        res = Orange.evaluation.CrossValidation(self.data, [learn], k=2)
        self.assertGreater(Orange.evaluation.CA(res)[0], 0.9)

    def test_LinearSVM(self):
        learn = LinearSVMLearner()
        res = Orange.evaluation.CrossValidation(self.data, [learn], k=2)
        self.assertTrue(0.8 < Orange.evaluation.CA(res)[0] < 0.9)

    def test_NuSVM(self):
        learn = NuSVMLearner(nu=0.01)
        res = Orange.evaluation.CrossValidation(self.data, [learn], k=2)
        self.assertGreater(Orange.evaluation.CA(res)[0], 0.9)

    def test_SVR(self):
        nrows, ncols = 200, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Orange.data.Table(X, y)
        learn = SVRLearner(kernel='rbf', gamma=0.1)
        res = Orange.evaluation.CrossValidation(data, [learn], k=2)
        self.assertLess(Orange.evaluation.RMSE(res)[0], 0.15)

    def test_NuSVR(self):
        nrows, ncols = 200, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Orange.data.Table(X, y)
        learn = NuSVRLearner(kernel='rbf', gamma=0.1)
        res = Orange.evaluation.CrossValidation(data, [learn], k=2)
        self.assertLess(Orange.evaluation.RMSE(res)[0], 0.1)

    def test_OneClassSVM(self):
        # TODO: improve the test - what does it check?
        nrows, ncols = 200, 5
        X = 0.3 * np.random.randn(nrows, ncols)
        X = np.r_[X + 2, X - 2]
        table = Orange.data.Table(X, None)
        learn = OneClassSVMLearner(kernel="rbf")
        m = learn(table[:100])
        z = m(table[100:])
        self.assertTrue(0.1 < np.sum(z == 1) < 0.5 * len(z))
