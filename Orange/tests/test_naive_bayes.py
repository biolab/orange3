import unittest
import numpy as np

import Orange
import Orange.classification.naive_bayes as nb
from Orange.evaluation import scoring, testing


class NaiveBayesTest(unittest.TestCase):
    def test_NaiveBayes(self):
        nrows = 1000
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        col = np.random.randint(ncols)
        y = x[:nrows, col].reshape(nrows, 1) + 100

        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = Orange.data.Table(x1, y1)
        learn = nb.BayesLearner()
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((z.reshape((-1, 1)) == y2).all())

    def test_BayesStorage(self):
        nrows = 200
        ncols = 10
        x = np.random.random_integers(0, 3, (nrows, ncols))
        x[:, 0] = 3
        y = x[:, ncols // 2].reshape(nrows, 1)
        continuous_table = Orange.data.Table(x, y)
        table = Orange.data.discretization.DiscretizeTable(
            continuous_table, clean=False)
        bayes = nb.BayesStorageLearner()
        results = testing.CrossValidation(table, [bayes], k=10)
        ca = scoring.CA(results)
        self.assertGreater(ca, 0.95)
