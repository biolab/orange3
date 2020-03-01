# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import Table, DiscreteVariable
from Orange.preprocess import transformation
from Orange.widgets.data import owcontinuize
from Orange.widgets.data.owcontinuize import OWContinuize
from Orange.widgets.tests.base import WidgetTest
from orangewidget.widget import StateInfo


class TestOWContinuize(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWContinuize)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        widget = self.widget
        widget.multinomial_treatment = 1

        self.send_signal(self.widget.Inputs.data, data)
        widget.unconditional_commit()
        imp_data = self.get_output(self.widget.Outputs.data)
        np.testing.assert_equal(imp_data.X, data.X)
        np.testing.assert_equal(imp_data.Y, data.Y)

        widget.continuous_treatment = 1
        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(data.domain))
        widget.unconditional_commit()
        imp_data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(imp_data), 0)

        self.send_signal(self.widget.Inputs.data, None)
        widget.unconditional_commit()
        imp_data = self.get_output(self.widget.Outputs.data)
        self.assertIsNone(imp_data)

    def test_summary(self):
        """Check if status bar is updated when data is received"""
        data = Table("iris")
        input_sum = self.widget.info.set_input_summary = Mock()
        output_sum = self.widget.info.set_output_summary = Mock()

        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(int(StateInfo.format_number(len(data))))
        output = self.get_output(self.widget.Outputs.data)
        output_sum.assert_called_with(int(StateInfo.format_number(len(output))))

        input_sum.reset_mock()
        output_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_once()
        self.assertEqual(input_sum.call_args[0][0].brief, "")
        output_sum.assert_called_once()
        self.assertEqual(output_sum.call_args[0][0].brief, "")

    def test_one_column_equal_values(self):
        """
        No crash on a column with equal values and with selected option
        normalize by standard deviation.
        GH-2144
        """
        table = Table("iris")
        table = table[:, 1]
        table[:] = 42.0
        self.send_signal(self.widget.Inputs.data, table)
        # Normalize.NormalizeBySD
        self.widget.continuous_treatment = 2
        self.widget.unconditional_commit()

    def test_one_column_nan_values_normalize_sd(self):
        """
        No crash on a column with NaN values and with selected option
        normalize by standard deviation (Not the same issue which is
        tested above).
        GH-2144
        """
        table = Table("iris")
        table[:, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        # Normalize.NormalizeBySD
        self.widget.continuous_treatment = 2
        self.widget.unconditional_commit()
        table = Table("iris")
        table[1, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.unconditional_commit()


    def test_one_column_nan_values_normalize_span(self):
        """
        No crash on a column with NaN values and with selected option
        normalize by span.
        GH-2144
        """
        table = Table("iris")
        table[:, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        # Normalize.NormalizeBySpan
        self.widget.continuous_treatment = 1
        self.widget.unconditional_commit()
        table = Table("iris")
        table[1, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.unconditional_commit()


class TestOWContinuizeUtils(unittest.TestCase):
    def test_dummy_coding_zero_based(self):
        var = DiscreteVariable("foo", values=list("abc"))

        varb, varc = owcontinuize.dummy_coding(var)

        self.assertEqual(varb.name, "foo=b")
        self.assertIsInstance(varb.compute_value, transformation.Indicator)
        self.assertEqual(varb.compute_value.value, 1)
        self.assertIs(varb.compute_value.variable, var)

        self.assertEqual(varc.name, "foo=c")
        self.assertIsInstance(varc.compute_value, transformation.Indicator)
        self.assertEqual(varc.compute_value.value, 2)
        self.assertIs(varc.compute_value.variable, var)

        varb, varc = owcontinuize.dummy_coding(var, zero_based=False)

        self.assertEqual(varb.name, "foo=b")
        self.assertIsInstance(varb.compute_value, transformation.Indicator1)
        self.assertEqual(varb.compute_value.value, 1)
        self.assertIs(varb.compute_value.variable, var)

        self.assertEqual(varc.name, "foo=c")
        self.assertIsInstance(varc.compute_value, transformation.Indicator1)
        self.assertEqual(varc.compute_value.value, 2)
        self.assertIs(varb.compute_value.variable, var)

    def test_dummy_coding_base_value(self):
        var = DiscreteVariable("foo", values=list("abc"))

        varb, varc = owcontinuize.dummy_coding(var, base_value=0)

        self.assertEqual(varb.name, "foo=b")
        self.assertIsInstance(varb.compute_value, transformation.Indicator)
        self.assertEqual(varb.compute_value.value, 1)
        self.assertEqual(varc.name, "foo=c")
        self.assertIsInstance(varc.compute_value, transformation.Indicator)
        self.assertEqual(varc.compute_value.value, 2)

        varb, varc = owcontinuize.dummy_coding(var, base_value=1)

        self.assertEqual(varb.name, "foo=a")
        self.assertIsInstance(varb.compute_value, transformation.Indicator)
        self.assertEqual(varb.compute_value.value, 0)
        self.assertEqual(varc.name, "foo=c")
        self.assertIsInstance(varc.compute_value, transformation.Indicator)
        self.assertEqual(varc.compute_value.value, 2)

    def test_one_hot_coding(self):
        var = DiscreteVariable("foo", values=list("abc"))

        vars = owcontinuize.one_hot_coding(var)
        for i, (c, nvar) in enumerate(zip("abc", vars)):
            self.assertEqual(nvar.name, f"foo={c}")
            self.assertIsInstance(nvar.compute_value, transformation.Indicator)
            self.assertEqual(nvar.compute_value.value, i)
            self.assertIs(nvar.compute_value.variable, var)

        vars = owcontinuize.one_hot_coding(var, zero_based=False)
        for i, (c, nvar) in enumerate(zip("abc", vars)):
            self.assertEqual(nvar.name, f"foo={c}")
            self.assertIsInstance(nvar.compute_value, transformation.Indicator1)
            self.assertEqual(nvar.compute_value.value, i)
            self.assertIs(nvar.compute_value.variable, var)


if __name__ == "__main__":
    unittest.main()
