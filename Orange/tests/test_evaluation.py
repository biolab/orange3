import unittest
import numpy as np

from Orange import data
from Orange.evaluation.cross import CrossValidation
from Orange.classification.naive_bayes import BayesLearner

class CrossValidationTest(unittest.TestCase):

    def test_KFold(self):
        nrows = 1000
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        col = np.random.randint(ncols)
        y = x[:nrows,col].reshape(nrows,1)+100
        t = data.Table(x, y)

        cv = CrossValidation(t, BayesLearner())
        z, prob = cv.KFold(3)
        self.assertTrue((z.reshape((-1,1))==y).all())

    def test_KFold_probs(self):
        nrows = 19
        ncols = 5
        x = np.random.random_integers(1, 3, (nrows, ncols))
        col = np.random.randint(ncols)
        y = np.random.random_integers(1, 19, (nrows, 1))
        t = data.Table(x, y)

        cv = CrossValidation(t, BayesLearner())
        z, prob = cv.KFold(3)
        self.assertEqual(prob.shape, (nrows, len(t.domain.class_var.values)))
