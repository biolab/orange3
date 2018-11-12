# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.preprocess import Discretize
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.data.owtransform import OWTransform
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owpca import OWPCA


class TestOWTransform(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTransform)
        self.data = Table("iris")
        self.preprocessor = Discretize()

    def test_output(self):
        # send data and preprocessor
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.preprocessor, self.preprocessor)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsInstance(output, Table)
        self.assertEqual("Input data with 150 instances and 4 features.",
                         self.widget.input_label.text())
        self.assertEqual("Preprocessor Discretize() applied.",
                         self.widget.preprocessor_label.text())
        self.assertEqual("Output data includes 4 features.",
                         self.widget.output_label.text())

        # remove preprocessor
        self.send_signal(self.widget.Inputs.preprocessor, None)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsNone(output)
        self.assertEqual("Input data with 150 instances and 4 features.",
                         self.widget.input_label.text())
        self.assertEqual("No preprocessor on input.", self.widget.preprocessor_label.text())
        self.assertEqual("", self.widget.output_label.text())

        # send preprocessor
        self.send_signal(self.widget.Inputs.preprocessor, self.preprocessor)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsInstance(output, Table)
        self.assertEqual("Input data with 150 instances and 4 features.",
                         self.widget.input_label.text())
        self.assertEqual("Preprocessor Discretize() applied.",
                         self.widget.preprocessor_label.text())
        self.assertEqual("Output data includes 4 features.",
                         self.widget.output_label.text())

        # remove data
        self.send_signal(self.widget.Inputs.data, None)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsNone(output)
        self.assertEqual("No data on input.", self.widget.input_label.text())
        self.assertEqual("Preprocessor Discretize() on input.",
                         self.widget.preprocessor_label.text())
        self.assertEqual("", self.widget.output_label.text())

        # remove preprocessor
        self.send_signal(self.widget.Inputs.preprocessor, None)
        self.assertEqual("No data on input.", self.widget.input_label.text())
        self.assertEqual("No preprocessor on input.",
                         self.widget.preprocessor_label.text())
        self.assertEqual("", self.widget.output_label.text())

    def test_input_pca_preprocessor(self):
        owpca = self.create_widget(OWPCA)
        self.send_signal(owpca.Inputs.data, self.data, widget=owpca)
        owpca.components_spin.setValue(2)
        pp = self.get_output(owpca.Outputs.preprocessor, widget=owpca)
        self.assertIsNotNone(pp, Preprocess)

        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.preprocessor, pp)
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(output.X.shape, (len(self.data), 2))

    def test_error_transforming(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.preprocessor, Preprocess())
        self.assertTrue(self.widget.Error.pp_error.is_shown())
        output = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsNone(output)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.pp_error.is_shown())

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()
