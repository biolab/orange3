import unittest
import numpy as np

from Orange import data
from Orange import classification
from Orange.evaluation.testing import CrossValidation
from Orange.evaluation import scoring
from Orange.classification import naive_bayes, majority


class ScoringTest(unittest.TestCase):
    def testCA(self):
        nrows = 10000
        ncols = 5
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(1, 2, (nrows, 1))
        t = data.Table(x, y)

        pred = np.random.random_integers(2, 3, nrows)
        self.assertAlmostEqual(scoring.CA(t, pred), 0.25, delta=0.1)

        learn = naive_bayes.BayesLearner()
        clf = learn(t)
        pred = clf(x)
        self.assertAlmostEqual(scoring.CA(t, pred), 0.5, delta=0.1)

    def testAUC_binary(self):
        nrows = 1000
        ncols = 5
        x = np.random.random_integers(0, 1, (nrows, ncols))
        col1 = np.random.randint(ncols)
        col2 = np.random.randint(ncols)
        e = np.random.random_integers(0, 1, nrows)
        y = (x[:, col1] * x[:, col1] * e).reshape(nrows, 1)

        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = data.Table(x1, y1)
        learn = naive_bayes.BayesLearner()
        clf = learn(t)
        t = data.Table(x2, y2)
        prob = clf(x2, ret=classification.Model.Probs)
        auc = scoring.AUC_binary(t, prob)
        self.assertLess(auc, 1.0)
        self.assertGreater(auc, 0.7)
