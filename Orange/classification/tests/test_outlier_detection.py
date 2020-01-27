# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
from Orange.data import Table, Domain, ContinuousVariable
from Orange.classification import EllipticEnvelopeLearner, \
    IsolationForestLearner, LocalOutlierFactorLearner, OneClassSVMLearner


class TestOneClassSVMLearner(unittest.TestCase):
    def test_OneClassSVM(self):
        np.random.seed(42)
        domain = Domain((ContinuousVariable("c1"), ContinuousVariable("c2")))
        X_in = 0.3 * np.random.randn(40, 2)
        X_out = np.random.uniform(low=-4, high=4, size=(20, 2))
        X_all = Table(domain, np.r_[X_in + 2, X_in - 2, X_out])
        n_true_in = len(X_in) * 2
        n_true_out = len(X_out)

        nu = 0.2
        learner = OneClassSVMLearner(nu=nu)
        cls = learner(X_all)
        y_pred = cls(X_all)
        n_pred_out_all = np.sum(y_pred == -1)
        n_pred_in_true_in = np.sum(y_pred[:n_true_in] == 1)
        n_pred_out_true_out = np.sum(y_pred[- n_true_out:] == -1)

        self.assertEqual(np.absolute(y_pred).all(), 1)
        self.assertLessEqual(n_pred_out_all, len(X_all) * nu)
        self.assertLess(np.absolute(n_pred_out_all - n_true_out), 2)
        self.assertLess(np.absolute(n_pred_in_true_in - n_true_in), 4)
        self.assertLess(np.absolute(n_pred_out_true_out - n_true_out), 3)

    def test_OneClassSVM_ignores_y(self):
        domain = Domain((ContinuousVariable("x1"), ContinuousVariable("x2")),
                        class_vars=(ContinuousVariable("y1"), ContinuousVariable("y2")))
        X = np.random.random((40, 2))
        Y = np.random.random((40, 2))
        table = Table(domain, X, Y)
        classless_table = table.transform(Domain(table.domain.attributes))
        learner = OneClassSVMLearner()
        classless_model = learner(classless_table)
        model = learner(table)
        pred1 = classless_model(classless_table)
        pred2 = classless_model(table)
        pred3 = model(classless_table)
        pred4 = model(table)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)
        np.testing.assert_array_equal(pred3, pred4)


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

    def test_EllipticEnvelope_ignores_y(self):
        domain = Domain((ContinuousVariable("x1"), ContinuousVariable("x2")),
                        (ContinuousVariable("y1"), ContinuousVariable("y2")))
        X = np.random.random((40, 2))
        Y = np.random.random((40, 2))
        table = Table(domain, X, Y)
        classless_table = table.transform(Domain(table.domain.attributes))
        learner = EllipticEnvelopeLearner()
        classless_model = learner(classless_table)
        model = learner(table)
        pred1 = classless_model(classless_table)
        pred2 = classless_model(table)
        pred3 = model(classless_table)
        pred4 = model(table)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)
        np.testing.assert_array_equal(pred3, pred4)


class TestLocalOutlierFactorLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")

    def test_LocalOutlierFactor(self):
        detector = LocalOutlierFactorLearner(contamination=0.1)
        detect = detector(self.iris)
        is_inlier = detect(self.iris)
        self.assertEqual(len(np.where(is_inlier == -1)[0]), 14)


class TestIsolationForestLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")

    def test_IsolationForest(self):
        detector = IsolationForestLearner(contamination=0.1)
        detect = detector(self.iris)
        is_inlier = detect(self.iris)
        self.assertEqual(len(np.where(is_inlier == -1)[0]), 15)


if __name__ == "__main__":
    unittest.main()
