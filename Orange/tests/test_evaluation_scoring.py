import unittest

import numpy as np

import Orange
from Orange.preprocess import discretization
from Orange.evaluation import Results, CA, TestOnTrainingData


class Scoring_CA_Test(unittest.TestCase):
    def test_init(self):
        res = Results(nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        np.testing.assert_almost_equal(CA(res), [1, 1])

        res.predicted[0][0] = 1
        np.testing.assert_almost_equal(CA(res), [0.99, 1])

        res.predicted[1] = 1 - res.predicted[1]
        np.testing.assert_almost_equal(CA(res), [0.99, 0])

    def test_call(self):
        res = Results(nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        ca = CA()
        np.testing.assert_almost_equal(ca(res), [1, 1])

        res.predicted[0][0] = 1
        np.testing.assert_almost_equal(ca(res), [0.99, 1])

        res.predicted[1] = 1 - res.predicted[1]
        np.testing.assert_almost_equal(ca(res), [0.99, 0])

    def test_bayes(self):
        x = np.random.random_integers(1, 3, (100, 5))
        col = np.random.randint(5)
        y = x[:, col].copy().reshape(100, 1)
        t = Orange.data.Table(x, y)
        t = discretization.DiscretizeTable(
            t, method=Orange.preprocess.discretization.EqualWidth(n=3))

        res = TestOnTrainingData(t, [Orange.classification.NaiveBayesLearner()])
        np.testing.assert_almost_equal(CA(res), [1])

        t.Y[-20:] = 4 - t.Y[-20:]
        res = TestOnTrainingData(t, [Orange.classification.NaiveBayesLearner()])
        self.assertGreaterEqual(CA(res)[0], 0.75)
        self.assertLess(CA(res)[0], 1)
