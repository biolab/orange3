# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
from Orange.data import Table, Domain, ContinuousVariable
from Orange.classification import EllipticEnvelopeLearner


class TestEllipticEnvelopeLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        domain = Domain((ContinuousVariable("c1"), ContinuousVariable("c2")))
        cls.n_true_in, cls.n_true_out = 80, 20
        cls.X_in = 0.3 * np.random.randn(cls.n_true_in, 2)
        cls.X_out = np.random.uniform(low=-4, high=4,
                                      size=(cls.n_true_out, 2))
        cls.X_all = Table(domain, np.r_[cls.X_in, cls.X_out])
        cls.cont = cls.n_true_out / (cls.n_true_in + cls.n_true_out)
        cls.learner = EllipticEnvelopeLearner(contamination=cls.cont)
        cls.model = cls.learner(cls.X_all)

    def test_EllipticEnvelope(self):
        y_pred = self.model(self.X_all)
        n_pred_out_all = np.sum(y_pred == -1)
        n_pred_in_true_in = np.sum(y_pred[:self.n_true_in] == 1)
        n_pred_out_true_o = np.sum(y_pred[- self.n_true_out:] == -1)

        self.assertTrue(all(np.absolute(y_pred) == 1))
        self.assertGreaterEqual(len(self.X_all) * self.cont, n_pred_out_all)
        self.assertGreater(1, np.absolute(n_pred_out_all - self.n_true_out))
        self.assertGreater(2, np.absolute(n_pred_in_true_in - self.n_true_in))
        self.assertGreater(2, np.absolute(n_pred_out_true_o - self.n_true_out))

    def test_mahalanobis(self):
        n = len(self.X_all)
        y_pred = self.model(self.X_all)
        y_mahal = self.model.mahalanobis(self.X_all)
        y_mahal, y_pred = zip(*sorted(zip(y_mahal, y_pred), reverse=True))
        self.assertTrue(all(i == -1 for i in y_pred[:int(self.cont * n)]))
        self.assertTrue(all(i == 1 for i in y_pred[int(self.cont * n):]))
