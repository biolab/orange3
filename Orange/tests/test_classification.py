import unittest
import numpy as np

from Orange import data
import Orange.classification.majority as maj
import Orange.classification.naive_bayes as nb
import Orange.classification.linear_regression as lr

class MajorityTest(unittest.TestCase):

    def test_Majority(self):
        nrows = 1000
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(1, 3, (nrows, 1)) // 2
        y[0] = 4
        t = data.Table(x, y)
        learn = maj.MajorityLearner()
        clf = learn(t)

        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue((y2==1).all())

    def test_weights(self):
        nrows = 100
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(1, 5, (nrows, 1))
        heavy = 3
        w = (y==heavy)*123+1
        t = data.Table(x, y, W=w)
        learn = maj.MajorityLearner()
        clf = learn(t)

        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue((y2==heavy).all())


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


class LinearRegressionTest(unittest.TestCase):

    def test_LinearRegression(self):
        nrows = 1000
        ncols = 3
        x = np.random.random_integers(-20, 50, (nrows, ncols))
        c = np.random.rand(ncols, 1)*10-3
        e = np.random.rand(nrows, 1)-0.5
        y = np.dot(x,c) + e

        x1, x2 = np.split(x,2);
        y1, y2 = np.split(y,2);
        t = data.Table(x1, y1)
        learn = lr.LinearRegressionLearner()
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((abs(z.reshape(-1,1)-y2)<2.0).all())


class MultiClassTest(unittest.TestCase):

    # Naive Bayes implementation doesn't support multiple class variables
    def test_unsupported(self):
        nrows = 20
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))

        # multiple class variables
        y = np.random.random_integers(10, 11, (nrows, 2))
        t = data.Table(x, y)
        learn = nb.BayesLearner()
        with self.assertRaises(TypeError):
            clf = learn(t)

        # single class variable
        y = np.random.random_integers(10, 11, (nrows, 1))
        t = data.Table(x, y)
        learn = nb.BayesLearner()
        clf = learn(t)

    def test_supported(self):
        pass