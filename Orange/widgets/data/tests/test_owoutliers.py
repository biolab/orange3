# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access

import unittest
from unittest.mock import patch, Mock

from Orange.data import Table
from Orange.classification import LocalOutlierFactorLearner
from Orange.widgets.data.owoutliers import OWOutliers, run
from Orange.widgets.tests.base import WidgetTest, simulate


class TestRun(unittest.TestCase):
    def test_results(self):
        iris = Table("iris")
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)
        res = run(iris, LocalOutlierFactorLearner(), state)
        self.assertIsInstance(res.inliers, Table)
        self.assertIsInstance(res.outliers, Table)
        self.assertIsInstance(res.annotated_data, Table)

        self.assertEqual(iris.domain, res.inliers.domain)
        self.assertEqual(iris.domain, res.outliers.domain)
        self.assertIn("Outlier", res.annotated_data.domain)

        self.assertEqual(len(res.inliers), 145)
        self.assertEqual(len(res.outliers), 5)
        self.assertEqual(len(res.annotated_data), 150)

    def test_no_data(self):
        res = run(None, LocalOutlierFactorLearner(), Mock())
        self.assertIsNone(res.inliers)
        self.assertIsNone(res.outliers)
        self.assertIsNone(res.annotated_data)


class TestOWOutliers(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWOutliers)
        self.iris = Table("iris")
        self.heart_disease = Table("heart_disease")

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_outputs(self):
        """Check widget's data and the output with data on the input"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        inliers = self.get_output(self.widget.Outputs.inliers)
        outliers = self.get_output(self.widget.Outputs.outliers)
        data = self.get_output(self.widget.Outputs.data)
        self.assertIn(len(inliers), [135, 136])
        self.assertIn(len(outliers), [14, 15])
        self.assertEqual(len(data), 150)
        self.assertEqual(len(inliers.domain.attributes), 4)
        self.assertEqual(len(outliers.domain.attributes), 4)
        self.assertEqual(len(data.domain.attributes), 4)
        self.assertEqual(len(inliers.domain.class_vars), 1)
        self.assertEqual(len(outliers.domain.class_vars), 1)
        self.assertEqual(len(data.domain.class_vars), 1)
        self.assertEqual(len(inliers.domain.metas), 0)
        self.assertEqual(len(outliers.domain.metas), 0)
        self.assertEqual(len(data.domain.metas), 1)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.inliers))
        self.assertIsNone(self.get_output(self.widget.Outputs.outliers))
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_output_empirical_covariance(self):
        simulate.combobox_activate_index(self.widget.method_combo,
                                         self.widget.Covariance)
        self.send_signal(self.widget.Inputs.data, self.iris)
        inliers = self.get_output(self.widget.Outputs.inliers)
        outliers = self.get_output(self.widget.Outputs.outliers)
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(inliers), 135)
        self.assertEqual(len(outliers), 15)
        self.assertEqual(len(data), 150)
        self.assertEqual(len(inliers.domain.attributes), 4)
        self.assertEqual(len(outliers.domain.attributes), 4)
        self.assertEqual(len(data.domain.attributes), 4)
        self.assertEqual(len(inliers.domain.class_vars), 1)
        self.assertEqual(len(outliers.domain.class_vars), 1)
        self.assertEqual(len(data.domain.class_vars), 1)
        self.assertEqual(len(inliers.domain.metas), 0)
        self.assertEqual(len(outliers.domain.metas), 0)
        self.assertEqual(len(data.domain.metas), 2)
        self.assertEqual([m.name for m in data.domain.metas],
                         ["Outlier", "Mahalanobis"])

    def test_methods(self):
        def callback():
            self.widget.send_report()
            self.assertIsNotNone(self.get_output(self.widget.Outputs.inliers))
            self.assertIsNotNone(self.get_output(self.widget.Outputs.outliers))
            self.assertIsNotNone(self.get_output(self.widget.Outputs.data))

        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        simulate.combobox_run_through_all(self.widget.method_combo,
                                          callback=callback)

    @patch("Orange.classification.outlier_detection._OutlierModel.predict")
    def test_memory_error(self, mocked_predict: Mock):
        """
        Handling memory error.
        GH-2374
        """
        self.assertFalse(self.widget.Error.memory_error.is_shown())
        mocked_predict.side_effect = MemoryError
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.memory_error.is_shown())

    @patch("Orange.classification.outlier_detection._OutlierModel.predict")
    def test_singular_cov_error(self, mocked_predict: Mock):
        self.assertFalse(self.widget.Error.singular_cov.is_shown())
        mocked_predict.side_effect = ValueError
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.singular_cov.is_shown())

    def test_nans(self):
        """Widget does not crash with nans"""
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.inliers))
        simulate.combobox_activate_index(self.widget.method_combo,
                                         self.widget.Covariance)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.inliers))
        self.assertFalse(self.widget.Error.singular_cov.is_shown())

    @patch("Orange.widgets.data.owoutliers.OWOutliers.MAX_FEATURES", 3)
    @patch("Orange.widgets.data.owoutliers.OWOutliers.commit", Mock())
    def test_covariance_enabled(self):
        cov_item = self.widget.method_combo.model().item(self.widget.Covariance)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Warning.disabled_cov.is_shown())
        self.assertFalse(cov_item.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.iris[:, :2])
        self.assertFalse(self.widget.Warning.disabled_cov.is_shown())
        self.assertTrue(cov_item.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Warning.disabled_cov.is_shown())
        self.assertFalse(cov_item.isEnabled())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.disabled_cov.is_shown())
        self.assertTrue(cov_item.isEnabled())

    @patch("Orange.widgets.data.owoutliers.OWOutliers.report_items")
    def test_report(self, mocked_report: Mock):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.widget.send_report()
        mocked_report.assert_called()
        mocked_report.reset_mock()

        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()
        mocked_report.assert_not_called()

    def test_migrate_settings(self):
        settings = {"cont": 20, "empirical_covariance": True,
                    "gamma": 0.04, "nu": 30, "outlier_method": 0,
                    "support_fraction": 0.5, "__version__": 1}

        widget = self.create_widget(OWOutliers, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.iris)
        self.assertEqual(widget.svm_editor.nu, 30)
        self.assertEqual(widget.svm_editor.gamma, 0.04)

        self.assertEqual(widget.cov_editor.cont, 20)
        self.assertEqual(widget.cov_editor.empirical_covariance, True)
        self.assertEqual(widget.cov_editor.support_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
