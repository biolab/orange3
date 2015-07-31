import unittest

import numpy as np
from Orange.data import Table, Domain, ContinuousVariable
from Orange.classification import EllipticEnvelopeLearner


class EllipticEnvelopeTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        domain = Domain((ContinuousVariable("c1"), ContinuousVariable("c2")))
        self.n_true_in, self.n_true_out = 80, 20
        self.X_in = 0.3 * np.random.randn(self.n_true_in, 2)
        self.X_out = np.random.uniform(low=-4, high=4,
                                       size=(self.n_true_out, 2))
        self.X_all = Table(domain, np.r_[self.X_in, self.X_out])
        self.cont = self.n_true_out / (self.n_true_in + self.n_true_out)
        self.learner = EllipticEnvelopeLearner(contamination=self.cont)
        self.model = self.learner(self.X_all)

    def test_EllipticEnvelope(self):
        y_pred = self.model(self.X_all)
        n_pred_out_all = np.sum(y_pred == -1)
        n_pred_in_true_in = np.sum(y_pred[:self.n_true_in] == 1)
        n_pred_out_true_o = np.sum(y_pred[- self.n_true_out:] == -1)

        self.assertTrue(all(np.absolute(y_pred) == 1))
        self.assertTrue(n_pred_out_all <= len(self.X_all) * self.cont)
        self.assertTrue(np.absolute(n_pred_out_all - self.n_true_out) < 1)
        self.assertTrue(np.absolute(n_pred_in_true_in - self.n_true_in) < 2)
        self.assertTrue(np.absolute(n_pred_out_true_o - self.n_true_out) < 2)

    def test_mahalanobis(self):
        n = len(self.X_all)
        y_pred = self.model(self.X_all)
        y_mahal = self.model.mahalanobis(self.X_all)
        y_mahal, y_pred = zip(*sorted(zip(y_mahal, y_pred), reverse=True))
        self.assertTrue(all(i == -1 for i in y_pred[:int(self.cont * n)]))
        self.assertTrue(all(i == 1 for i in y_pred[int(self.cont * n):]))
