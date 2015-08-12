import unittest
import numpy as np
import Orange
from Orange.classification import SimpleRandomForestLearner as SimpRandForestCls
from Orange.regression import SimpleRandomForestLearner as SimpRandForestReg


class SimpleRandomForestTest(unittest.TestCase):
    def test_SimpleRandomForest_classification(self):
        data = Orange.data.Table('iris')
        lrn = SimpRandForestCls()
        clf = lrn(data)
        p = clf(data, clf.Probs)
        self.assertEqual(p.shape, (150, 3))
        self.assertGreaterEqual(p.min(), 0)
        self.assertLessEqual(p.max(), 1)
        np.testing.assert_almost_equal(p.sum(axis=1), np.ones(150))

    def test_SimpleRandomForest_regression(self):
        data = Orange.data.Table('housing')
        lrn = SimpRandForestReg()
        clf = lrn(data)
        p = clf(data)
        self.assertEqual(p.shape, (len(data),))


if __name__ == '__main__':
    unittest.main()
