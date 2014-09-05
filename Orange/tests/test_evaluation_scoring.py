import unittest
import numpy as np

from Orange.data import discretization, Table
from Orange.evaluation import testing, scoring
from Orange.classification import naive_bayes
from Orange.feature.discretization import EqualWidth

class Scoring_CA_Test(unittest.TestCase):
    def test_init(self):
        res = testing.Results(nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        np.testing.assert_almost_equal(scoring.CA(res), [1, 1])

        res.predicted[0][0] = 1
        np.testing.assert_almost_equal(scoring.CA(res), [0.99, 1])

        res.predicted[1] = 1 - res.predicted[1]
        np.testing.assert_almost_equal(scoring.CA(res), [0.99, 0])

    def test_call(self):
        res = testing.Results(nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        ca = scoring.CA()
        np.testing.assert_almost_equal(ca(res), [1, 1])

        res.predicted[0][0] = 1
        np.testing.assert_almost_equal(ca(res), [0.99, 1])

        res.predicted[1] = 1 - res.predicted[1]
        np.testing.assert_almost_equal(ca(res), [0.99, 0])

    def test_bayes(self):
        x = np.random.random_integers(1, 3, (100, 5))
        col = np.random.randint(5)
        y = x[:, col].copy().reshape(100, 1)
        t = Table(x, y)
        t = discretization.DiscretizeTable(t, method=EqualWidth(n=3))

        res = testing.TestOnTrainingData(t, [naive_bayes.BayesLearner()])
        np.testing.assert_almost_equal(scoring.CA(res), [1])

        t.Y[-20:] = 4 - t.Y[-20:]
        res = testing.TestOnTrainingData(t, [naive_bayes.BayesLearner()])
        self.assertGreaterEqual(scoring.CA(res)[0], 0.75)
        self.assertLess(scoring.CA(res)[0], 1)
