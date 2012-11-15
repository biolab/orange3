import unittest
import numpy as np

from Orange import data
import Orange.classification
import Orange.classification.dummies as dummies
import Orange.classification.majority as maj

class MultiClassTest(unittest.TestCase):

    def test_unsupported(self):
        nrows = 20
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))

        # multiple class variables
        y = np.random.random_integers(10, 11, (nrows, 2))
        t = data.Table(x, y)
        learn = dummies.DummyLearner()
        with self.assertRaises(TypeError):
            clf = learn(t)

        # single class variable
        y = np.random.random_integers(10, 11, (nrows, 1))
        t = data.Table(x, y)
        learn = dummies.DummyLearner()
        clf = learn(t)
        z = clf(x)
        self.assertEqual(z.ndim, 1)

    def test_supported(self):
        nrows = 20
        ncols = 10
        x = np.random.random_integers(1, 3, (nrows, ncols))
        y = np.random.random_integers(10, 11, (nrows, 2))
        t = data.Table(x, y)
        learn = dummies.DummyMulticlassLearner()
        clf = learn(t)
        z = clf(x)
        self.assertEqual(z.shape, y.shape)


class ModelTest(unittest.TestCase):

    def test_value_from_probs(self):
        nrows = 100
        ncols = 5
        x = np.random.random_integers(0, 1, (nrows, ncols))

        # single class variable
        y = np.random.random_integers(1, 3, (nrows, 2)) // 2    # majority = 1
        t = data.Table(x, y)
        learn = maj.MajorityLearner()
        clf = learn(t)
        clf.ret = Orange.classification.Model.Probs
        y2 = clf(x, ret=Orange.classification.Model.Value)
        self.assertTrue((y2==1).all())
        y2, probs = clf(x, ret=Orange.classification.Model.ValueProbs)
        self.assertTrue((y2==1).all())

        # multitarget
        y = np.random.random_integers(1, 5, (nrows, 2))
        y[:,0] = y[:,0] // 3        # majority = 1
        y[:,1] = (y[:,1]+4) // 3    # majority = 2
        t = data.Table(x, y)
        learn = maj.MajorityLearner()
        clf = learn(t)
        clf.ret = Orange.classification.Model.Probs
        y2 = clf(x, ret=Orange.classification.Model.Value)
        self.assertEqual(y2.shape, y.shape)
        self.assertTrue((y2[:,0]==1).all() and (y2[:,1]==2).all())
        y2, probs = clf(x, ret=Orange.classification.Model.ValueProbs)
        self.assertTrue((y2[:,0]==1).all() and (y2[:,1]==2).all())
