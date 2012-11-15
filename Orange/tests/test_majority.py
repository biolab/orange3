import unittest
import numpy as np

from Orange import data
import Orange.classification.majority as maj

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

    def test_multiclass(self):
        nrows = 100
        ncols = 10
        nclass = 4
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(10, 19, (nrows, nclass))
        for i in range(nclass):
            length = nrows // 2
            start = np.random.randint(nrows-length+1)
            y[start:start+length,i] = np.tile(i,length)
        t = data.Table(x, y)
        learn = maj.MajorityLearner()
        clf = learn(t)

        x2 = np.random.random_integers(1, 3, (nrows, ncols))
        y2 = clf(x2)
        self.assertTrue(all(all(y2[:,i]==i) for i in range(nclass)))
