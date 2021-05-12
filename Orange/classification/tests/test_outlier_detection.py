# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from Orange.classification import EllipticEnvelopeLearner, \
    IsolationForestLearner, LocalOutlierFactorLearner, OneClassSVMLearner
from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.table import DomainTransformationError


class _TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")

    def assert_domain_equal(self, domain1, domain2):
        for var1, var2 in zip(domain1.variables + domain1.metas,
                              domain2.variables + domain2.metas):
            self.assertEqual(type(var1), type(var2))
            self.assertEqual(var1.name, var2.name)
            if var1.is_discrete:
                self.assertEqual(var1.values, var2.values)

    def assert_table_equal(self, table1, table2):
        if table1 is None or table2 is None:
            self.assertIs(table1, table2)
            return
        self.assert_domain_equal(table1.domain, table2.domain)
        np.testing.assert_array_equal(table1.X, table2.X)
        np.testing.assert_array_equal(table1.Y, table2.Y)
        np.testing.assert_array_equal(table1.metas, table2.metas)

    def assert_table_appended_outlier(self, table1, table2, offset=1):
        np.testing.assert_array_equal(table1.ids, table2.ids)
        np.testing.assert_array_equal(table1.X, table2.X)
        np.testing.assert_array_equal(table1.Y, table2.Y)
        np.testing.assert_array_equal(table1.metas, table2.metas[:, :-offset])
        metas = table2.metas[:, -offset]
        self.assertEqual(sum(metas == 1) + sum(metas == 0), len(metas))
        dom = table2.domain
        domain = Domain(dom.attributes, dom.class_vars, dom.metas[:-offset])
        self.assert_domain_equal(table1.domain, domain)
        self.assertEqual(table2.domain.metas[-offset].name, "Outlier")
        self.assertIsNotNone(table2.domain.metas[-offset].compute_value)

class TestOneClassSVMLearner(_TestDetector):
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
        n_pred_out_all = np.sum(y_pred.metas == 0)
        n_pred_in_true_in = np.sum(y_pred.metas[:n_true_in] == 1)
        n_pred_out_true_out = np.sum(y_pred.metas[- n_true_out:] == 0)

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

        np.testing.assert_array_equal(pred1.metas, pred2.metas)
        np.testing.assert_array_equal(pred2.metas, pred3.metas)
        np.testing.assert_array_equal(pred3.metas, pred4.metas)

    def test_transform(self):
        detector = OneClassSVMLearner(nu=0.1)
        detect = detector(self.iris)
        pred = detect(self.iris)
        self.assert_table_appended_outlier(self.iris, pred)
        pred2 = self.iris.transform(pred.domain)
        self.assert_table_equal(pred, pred2)


class TestEllipticEnvelopeLearner(_TestDetector):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
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
        n_pred_out_all = np.sum(y_pred.metas == 0)
        n_pred_in_true_in = np.sum(y_pred.metas[:self.n_true_in] == 1)
        n_pred_out_true_o = np.sum(y_pred.metas[- self.n_true_out:] == 0)

        self.assertGreaterEqual(len(self.X_all) * self.cont, n_pred_out_all)
        self.assertGreater(1, np.absolute(n_pred_out_all - self.n_true_out))
        self.assertGreater(2, np.absolute(n_pred_in_true_in - self.n_true_in))
        self.assertGreater(2, np.absolute(n_pred_out_true_o - self.n_true_out))

    def test_mahalanobis(self):
        n = len(self.X_all)
        pred = self.model(self.X_all)

        y_pred = pred[:, self.model.outlier_var].metas
        y_mahal = pred[:, self.model.mahal_var].metas
        y_mahal, y_pred = zip(*sorted(zip(y_mahal, y_pred), reverse=True))
        self.assertTrue(all(i == 0 for i in y_pred[:int(self.cont * n)]))
        self.assertTrue(all(i == 1 for i in y_pred[int(self.cont * n):]))

    def test_single_data_to_model_domain(self):
        with patch.object(self.model, "data_to_model_domain",
                          wraps=self.model.data_to_model_domain) as call:
            self.model(self.X_all)
            self.assertEqual(call.call_count, 1)

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

        np.testing.assert_array_equal(pred1.metas, pred2.metas)
        np.testing.assert_array_equal(pred2.metas, pred3.metas)
        np.testing.assert_array_equal(pred3.metas, pred4.metas)

    def test_transform(self):
        detector = EllipticEnvelopeLearner()
        detect = detector(self.iris)
        pred = detect(self.iris)
        self.assert_table_appended_outlier(self.iris, pred, offset=2)
        self.assertEqual(pred.domain.metas[-1].name, "Mahalanobis")
        self.assertIsNotNone(pred.domain.metas[-1].compute_value)
        pred2 = self.iris.transform(pred.domain)
        self.assert_table_equal(pred, pred2)


class TestLocalOutlierFactorLearner(_TestDetector):
    def test_LocalOutlierFactor(self):
        detector = LocalOutlierFactorLearner(contamination=0.1)
        detect = detector(self.iris)
        pred = detect(self.iris)
        self.assertEqual(len(np.where(pred.metas == 0)[0]), 14)

    def test_transform(self):
        detector = LocalOutlierFactorLearner(contamination=0.1)
        detect = detector(self.iris)
        pred = detect(self.iris)
        self.assert_table_appended_outlier(self.iris, pred)
        pred2 = self.iris.transform(pred.domain)
        self.assert_table_equal(pred, pred2)


class TestIsolationForestLearner(_TestDetector):
    def test_IsolationForest(self):
        detector = IsolationForestLearner(contamination=0.1)
        detect = detector(self.iris)
        pred = detect(self.iris)
        self.assertEqual(len(np.where(pred.metas == 0)[0]), 15)

    def test_transform(self):
        detector = IsolationForestLearner(contamination=0.1)
        detect = detector(self.iris)
        pred = detect(self.iris)
        self.assert_table_appended_outlier(self.iris, pred)
        pred2 = self.iris.transform(pred.domain)
        self.assert_table_equal(pred, pred2)


class TestOutlierModel(_TestDetector):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.detector = LocalOutlierFactorLearner(contamination=0.1)

    def test_unique_name(self):
        domain = Domain((ContinuousVariable("Outlier"),))
        table = Table(domain, np.random.random((40, 1)))
        detect = self.detector(table)
        pred = detect(table)
        self.assertEqual(pred.domain.metas[0].name, "Outlier (1)")

    def test_predict(self):
        detect = self.detector(self.iris)
        subset = self.iris[:, :3]
        pred = detect(subset)
        self.assert_table_appended_outlier(subset, pred)

    def test_predict_all_nan(self):
        detect = self.detector(self.iris[:, :2])
        subset = self.iris[:, 2:]
        self.assertRaises(DomainTransformationError, detect, subset)

    def test_transform(self):
        detect = self.detector(self.iris)
        pred = detect(self.iris)
        self.assert_table_appended_outlier(self.iris, pred)
        pred2 = self.iris.transform(pred.domain)
        self.assert_table_equal(pred, pred2)

    def test_transformer(self):
        detect = self.detector(self.iris)
        pred = detect(self.iris)
        var = pred.domain.metas[0]
        np.testing.assert_array_equal(pred[:, "Outlier"].metas.ravel(),
                                      var.compute_value(self.iris))

    def test_pickle_model(self):
        detect = self.detector(self.iris)
        f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        pickle.dump(detect, f)
        f.close()

    def test_pickle_prediction(self):
        detect = self.detector(self.iris)
        pred = detect(self.iris)
        f = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        pickle.dump(pred, f)
        f.close()

    def test_fit_callback(self):
        callback = Mock()
        self.detector(self.iris, callback)
        args = [x[0][0] for x in callback.call_args_list]
        self.assertEqual(min(args), 0)
        self.assertEqual(max(args), 1)
        self.assertListEqual(args, sorted(args))

    def test_predict_callback(self):
        callback = Mock()
        detect = self.detector(self.iris)
        detect(self.iris, callback)
        args = [x[0][0] for x in callback.call_args_list]
        self.assertEqual(min(args), 0)
        self.assertEqual(max(args), 1)
        self.assertListEqual(args, sorted(args))


if __name__ == "__main__":
    unittest.main()
