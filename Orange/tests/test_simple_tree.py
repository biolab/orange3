import unittest
import pickle

import numpy as np

import Orange
from Orange.classification.simple_tree import SimpleTreeLearner


class SimpleTreeTest(unittest.TestCase):

    def setUp(self):
        self.N = 50
        self.Mi = 3
        self.Mf = 3
        self.cls_vals = 3
        np.random.seed(42)
        Xi = np.random.randint(0, 2, (self.N, self.Mi)).astype(np.float64)
        Xf = np.random.normal(0, 1, (self.N, self.Mf)).astype(np.float64)
        X = np.hstack((Xi, Xf))
        y_cls = np.random.randint(0, self.cls_vals, self.N).astype(np.float64)
        y_reg = np.random.normal(0, 1, self.N).astype(np.float64)

        X[np.random.random(X.shape) < 0.1] = np.nan
        y_cls[np.random.random(self.N) < 0.1] = np.nan
        y_reg[np.random.random(self.N) < 0.1] = np.nan

        di = [Orange.data.domain.DiscreteVariable(
            'd{}'.format(i), [0, 1]) for i in range(self.Mi)]
        df = [Orange.data.domain.ContinuousVariable(
            'c{}'.format(i)) for i in range(self.Mf)]
        dcls = Orange.data.domain.DiscreteVariable('yc', [0, 1, 2])
        dreg = Orange.data.domain.ContinuousVariable('yr')
        domain_cls = Orange.data.domain.Domain(di + df, dcls)
        domain_reg = Orange.data.domain.Domain(di + df, dreg)

        self.data_cls = Orange.data.Table.from_numpy(domain_cls, X, y_cls)
        self.data_reg = Orange.data.Table.from_numpy(domain_reg, X, y_reg)

    def test_SimpleTree_classification(self):
        Orange.data.Variable._clear_all_caches()
        lrn = SimpleTreeLearner()
        clf = lrn(self.data_cls)
        p = clf(self.data_cls, clf.Probs)
        self.assertEqual(p.shape, (self.N, self.cls_vals))
        self.assertAlmostEqual(p.min(), 0)
        self.assertAlmostEqual(p.max(), 1)
        np.testing.assert_almost_equal(p.sum(axis=1), np.ones(self.N))

    def test_SimpleTree_classification_pickle(self):
        lrn = SimpleTreeLearner()
        clf = lrn(self.data_cls)
        p = clf(self.data_cls, clf.Probs)
        clf_ = pickle.loads(pickle.dumps(clf))
        p_ = clf_(self.data_cls, clf.Probs)
        np.testing.assert_almost_equal(p, p_)

    def test_SimpleTree_classification_tree(self):
        lrn = SimpleTreeLearner(min_instances=6, max_majority=0.7)
        clf = lrn(self.data_cls)
        self.assertEqual(clf.dumps_tree(
            clf.node), '{ 1 4 -1.17364 { 1 5 0.37564 { 2 0.00 0.00 0.56 } { 2 0.00 3.00 1.14 } } { 1 4 -0.41863 { 1 5 0.14592 { 2 3.54 0.54 0.70 } { 2 2.46 0.46 2.47 } } { 1 4 0.24404 { 1 4 0.00654 { 1 3 -0.15750 { 2 1.00 0.00 0.45 } { 2 1.00 3.00 0.48 } } { 2 1.00 5.00 0.70 } } { 1 5 0.32635 { 2 0.52 2.52 4.21 } { 2 2.48 3.48 1.30 } } } } }')

    def test_SimpleTree_regression(self):
        lrn = SimpleTreeLearner()
        clf = lrn(self.data_reg)
        p = clf(self.data_reg)
        self.assertEqual(p.shape, (self.N,))

    def test_SimpleTree_regression_pickle(self):
        pass

    def test_SimpleTree_regression_tree(self):
        lrn = SimpleTreeLearner(min_instances=5)
        clf = lrn(self.data_reg)
        self.assertEqual(clf.dumps_tree(
            clf.node), '{ 0 2 { 1 4 0.13895 { 1 4 -0.32607 { 2 4.60993 1.71141 } { 2 4.96454 3.56122 } } { 2 7.09220 -4.32343 } } { 1 4 -0.35941 { 0 0 { 1 5 -0.20027 { 2 3.54255 0.95095 } { 2 5.50000 -5.56049 } } { 2 7.62411 2.03615 } } { 1 5 0.40797 { 1 3 0.83459 { 2 3.71094 0.27028 } { 2 5.18490 3.70920 } } { 2 5.77083 5.93398 } } } }')

    def test_SimpleTree_single_instance(self):
        data = Orange.data.Table('iris')
        lrn = SimpleTreeLearner()
        clf = lrn(data)
        for ins in data[::20]:
            clf(ins)
            val, prob = clf(ins, clf.ValueProbs)
            self.assertEqual(sum(prob[0]), 1)


if __name__ == '__main__':
    unittest.main()
