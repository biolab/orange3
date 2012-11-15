import unittest
import numpy as np

from Orange import data
import Orange.classification.naive_bayes as nb

class NaiveBayesTest(unittest.TestCase):

    def test_NaiveBayes(self):
        nrows = 1000
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        col = np.random.randint(ncols)
        y = x[:nrows,col].reshape(nrows,1)+100

        x1, x2 = np.split(x,2);
        y1, y2 = np.split(y,2);
        t = data.Table(x1, y1)
        learn = nb.BayesLearner()
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((z.reshape((-1,1))==y2).all())