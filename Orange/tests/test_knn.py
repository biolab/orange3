# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.classification import KNNLearner
from Orange.evaluation import CA, CrossValidation


class KNNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.learn = KNNLearner()

    def test_KNN(self):
        results = CrossValidation(self.iris, [self.learn], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.8)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        clf = self.learn(self.iris)
        for ins in self.iris[::20]:
            clf(ins)
            val, prob = clf(ins, clf.ValueProbs)

    def test_random(self):
        nrows, ncols = 1000, 5
        x = np.random.random_integers(-20, 50, (nrows, ncols))
        y = np.random.random_integers(-2, 2, (nrows, 1))
        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        attr = (ContinuousVariable('Feature 1'),
                ContinuousVariable('Feature 2'),
                ContinuousVariable('Feature 3'),
                ContinuousVariable('Feature 4'),
                ContinuousVariable('Feature 5'))
        class_vars = (DiscreteVariable('Target 1'),)
        domain = Domain(attr, class_vars)
        t = Table(domain, x1, y1)
        clf = self.learn(t)
        z = clf(x2)
        correct = (z == y2.flatten())
        ca = sum(correct)/len(correct)
        self.assertGreater(ca, 0.1)
        self.assertLess(ca, 0.3)
