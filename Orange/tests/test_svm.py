# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from Orange.classification import SVMLearner, LinearSVMLearner, NuSVMLearner
from Orange.data import Table
from Orange.evaluation import CrossValidation, CA, RMSE
from Orange.regression import SVRLearner, NuSVRLearner
from Orange.tests import test_filename


class TestSVMLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table(test_filename('datasets/ionosphere.tab'))
        with cls.data.unlocked():
            cls.data.shuffle()

    def test_SVM(self):
        learn = SVMLearner()
        cv = CrossValidation(k=2)
        res = cv(self.data, [learn])
        self.assertGreater(CA(res)[0], 0.9)

    def test_LinearSVM(self):
        # This warning is irrelevant here
        warnings.filterwarnings("ignore", ".*", ConvergenceWarning)
        learn = LinearSVMLearner()
        cv = CrossValidation(k=2)
        res = cv(self.data, [learn])
        self.assertGreater(CA(res)[0], 0.8)
        self.assertLess(CA(res)[0], 0.9)

    def test_NuSVM(self):
        learn = NuSVMLearner(nu=0.01)
        cv = CrossValidation(k=2)
        res = cv(self.data, [learn])
        self.assertGreater(CA(res)[0], 0.9)

    def test_SVR(self):
        nrows, ncols = 200, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Table.from_numpy(None, X, y)
        learn = SVRLearner(kernel='rbf', gamma=0.1)
        cv = CrossValidation(k=2)
        res = cv(data, [learn])
        self.assertLess(RMSE(res)[0], 0.15)

    def test_LinearSVR(self):
        nrows, ncols = 200, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Table.from_numpy(None, X, y)
        learn = SVRLearner()
        cv = CrossValidation(k=2)
        res = cv(data, [learn])
        self.assertLess(RMSE(res)[0], 0.15)

    def test_NuSVR(self):
        nrows, ncols = 200, 5
        X = np.random.rand(nrows, ncols)
        y = X.dot(np.random.rand(ncols))
        data = Table.from_numpy(None, X, y)
        learn = NuSVRLearner(kernel='rbf', gamma=0.1)
        cv = CrossValidation(k=2)
        res = cv(data, [learn])
        self.assertLess(RMSE(res)[0], 0.1)


if __name__ == "__main__":
    unittest.main()
