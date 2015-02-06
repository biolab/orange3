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
        n = int(0.7*self.data.X.shape[0])
        learn = SVMLearner()
        clf = learn(self.data[:n])
        z = clf(self.data[n:])
        self.assertTrue(np.sum(z.reshape((-1, 1)) == self.data.Y[n:]) > 0.7*len(z))

    def test_LinearSVM(self):
        n = int(0.7*self.data.X.shape[0])
        learn = LinearSVMLearner()
        clf = learn(self.data[:n])
        z = clf(self.data[n:])
        self.assertTrue(np.sum(z.reshape((-1, 1)) == self.data.Y[n:]) > 0.7*len(z))

    def test_NuSVM(self):
        n = int(0.7*self.data.X.shape[0])
        learn = NuSVMLearner(nu=0.01)
        clf = learn(self.data[:n])
        z = clf(self.data[n:])
        self.assertTrue(np.sum(z.reshape((-1, 1)) == self.data.Y[n:]) > 0.7*len(z))

    def test_SVR(self):
        nrows = 500
        ncols = 5
        x = np.sort(10*np.random.rand(nrows, ncols))
        y = np.sum(np.sin(x), axis=1).reshape(nrows, 1)
        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = Orange.data.Table(x1, y1)
        learn = SVRLearner(kernel='rbf', C=1e3, gamma=0.1)
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((abs(z.reshape(-1, 1) - y2) < 4.0).all())

    def test_NuSVR(self):
        nrows = 500
        ncols = 5
        x = np.sort(10*np.random.rand(nrows, ncols))
        y = np.sum(np.sin(x), axis=1).reshape(nrows, 1)
        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = Orange.data.Table(x1, y1)
        learn = NuSVRLearner(kernel='rbf', C=1e3, gamma=0.1)
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((abs(z.reshape(-1, 1) - y2) < 4.0).all())

    def test_OneClassSVM(self):
        nrows = 100
        ncols = 5
        x1 = 0.3 * np.random.randn(nrows, ncols)
        t = Orange.data.Table(np.r_[x1 + 2, x1 - 2], None)
        x2 = 0.3 * np.random.randn(nrows, ncols)
        x2 = np.r_[x2 + 2, x2 - 2]
        learn = OneClassSVMLearner(kernel="rbf", nu=0.1, gamma=0.1)
        clf = learn(t)
        z = clf(x2)
        self.assertTrue(np.sum(z == 1) > 0.7*len(z))
