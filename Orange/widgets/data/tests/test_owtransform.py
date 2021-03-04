# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
from unittest.mock import Mock

from numpy import testing as npt

from Orange.data import Table
from Orange.preprocess import Discretize
from Orange.widgets.data.owtransform import OWTransform
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owpca import OWPCA


class TestOWTransform(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTransform)
        self.data = Table("iris")
        self.disc_data = Discretize()(self.data)

    def test_output(self):
        # send data and template data
        self.send_signal(self.widget.Inputs.data, self.data[::15])
        self.send_signal(self.widget.Inputs.template_data, self.disc_data)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertTableEqual(output, self.disc_data[::15])
        self.assertEqual("Input data with 10 instances and 4 features.",
                         self.widget.input_label.text())
        self.assertEqual("Template domain applied.",
                         self.widget.template_label.text())
        self.assertEqual("Output data includes 4 features.",
                         self.widget.output_label.text())

        # remove template data
        self.send_signal(self.widget.Inputs.template_data, None)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsNone(output)
        self.assertEqual("Input data with 10 instances and 4 features.",
                         self.widget.input_label.text())
        self.assertEqual("No template data on input.",
                         self.widget.template_label.text())
        self.assertEqual("", self.widget.output_label.text())

        # send template data
        self.send_signal(self.widget.Inputs.template_data, self.disc_data)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertTableEqual(output, self.disc_data[::15])
        self.assertEqual("Input data with 10 instances and 4 features.",
                         self.widget.input_label.text())
        self.assertEqual("Template domain applied.",
                         self.widget.template_label.text())
        self.assertEqual("Output data includes 4 features.",
                         self.widget.output_label.text())

        # remove data
        self.send_signal(self.widget.Inputs.data, None)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsNone(output)
        self.assertEqual("No data on input.", self.widget.input_label.text())
        self.assertEqual("Template data includes 4 features.",
                         self.widget.template_label.text())
        self.assertEqual("", self.widget.output_label.text())

        # remove template data
        self.send_signal(self.widget.Inputs.template_data, None)
        self.assertEqual("No data on input.", self.widget.input_label.text())
        self.assertEqual("No template data on input.",
                         self.widget.template_label.text())
        self.assertEqual("", self.widget.output_label.text())

    def assertTableEqual(self, table1, table2):
        self.assertIs(table1.domain, table2.domain)
        npt.assert_array_equal(table1.X, table2.X)
        npt.assert_array_equal(table1.Y, table2.Y)
        npt.assert_array_equal(table1.metas, table2.metas)

    def test_input_pca_output(self):
        owpca = self.create_widget(OWPCA)
        self.send_signal(owpca.Inputs.data, self.data, widget=owpca)
        owpca.components_spin.setValue(2)
        pca_out = self.get_output(owpca.Outputs.transformed_data, widget=owpca)

        self.send_signal(self.widget.Inputs.data, self.data[::10])
        self.send_signal(self.widget.Inputs.template_data, pca_out)
        output = self.get_output(self.widget.Outputs.transformed_data)
        npt.assert_array_equal(pca_out.X[::10], output.X)

    def test_error_transforming(self):
        data = self.data[::10]
        data.transform = Mock(side_effect=Exception())
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.template_data, self.disc_data)
        self.assertTrue(self.widget.Error.error.is_shown())
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsNone(output)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.error.is_shown())

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()


if __name__ == "__main__":
    import unittest
    unittest.main()
