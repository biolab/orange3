import unittest
import numpy as np

import Orange
import Orange.classification.knn as knn
from Orange.evaluation.testing import CrossValidation
from Orange.evaluation.scoring import CA

class KNNTest(unittest.TestCase):
    def test_KNN(self):
        table = Orange.data.Table('iris')
        learn = knn.KNNLearner()
        results = CrossValidation(table, [learn], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        data = Orange.data.Table('iris')
        learn = knn.KNNLearner()
        clf = learn(data)
        for ins in data[::20]:
            clf(ins)
            val, prob = clf(ins, clf.ValueProbs)

    def test_random(self):
        nrows, ncols = 1000, 5
        x = np.random.random_integers(-20, 50, (nrows, ncols))
        y = np.random.random_integers(-2, 2, (nrows, 1))
        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = Orange.data.Table(x1, y1)
        learn = knn.KNNLearner()
        clf = learn(t)
        z = clf(x2)
        correct = (z == y2.flatten())
        ca = sum(correct)/len(correct)
        self.assertTrue(0.1 < ca < 0.3)

    def test_mahalanobis(self):
        table = Orange.data.Table('iris')
        learn = knn.KNNLearner(metric='mahalanobis')
        results = CrossValidation(table, [learn], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)