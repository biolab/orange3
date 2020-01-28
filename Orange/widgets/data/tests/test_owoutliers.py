# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access

import unittest
from unittest.mock import patch, Mock

from Orange.data import Table
from Orange.widgets.data.owoutliers import OWOutliers
from Orange.widgets.tests.base import WidgetTest, simulate


class TestOWOutliers(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWOutliers)
        self.iris = Table("iris")
        self.heart_disease = Table("heart_disease")

    def test_outputs(self):
        """Check widget's data and the output with data on the input"""
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

        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_run_through_all(self.widget.method_combo,
                                          callback=callback)

    def test_memory_error(self):
        """
        Handling memory error.
        GH-2374
        """
        data = Table("iris")[::3]
        self.assertFalse(self.widget.Error.memory_error.is_shown())
        with unittest.mock.patch(
                "Orange.widgets.data.owoutliers.OWOutliers.detect_outliers",
                side_effect=MemoryError):
            self.send_signal("Data", data)
            self.assertTrue(self.widget.Error.memory_error.is_shown())

    def test_nans(self):
        """Widget does not crash with nans"""
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.inliers))
        simulate.combobox_activate_index(self.widget.method_combo,
                                         self.widget.Covariance)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.inliers))
        self.assertFalse(self.widget.Error.singular_cov.is_shown())

    def test_in_out_summary(self):
        info = self.widget.info
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.brief, "")

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(info._StateInfo__input_summary.brief, "150")
        self.assertEqual(info._StateInfo__output_summary.brief, "135")

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.brief, "")

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
