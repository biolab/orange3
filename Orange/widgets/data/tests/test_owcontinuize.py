# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import Table, DiscreteVariable, ContinuousVariable, Domain
from Orange.preprocess import transformation
from Orange.widgets.data import owcontinuize
from Orange.widgets.data.owcontinuize import OWContinuize
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.state_summary import format_summary_details


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
        input_sum.assert_called_with(len(data),
                                     format_summary_details(data))
        output = self.get_output(self.widget.Outputs.data)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))

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

    def test_disable_normalize_sparse(self):
        def assert_enabled(enabled):
            for button, (method, supports_sparse) in \
                    zip(buttons, w.continuous_treats):
                self.assertEqual(button.isEnabled(), enabled or supports_sparse,
                                 msg=f"Error in {method}")
            buttons[w.Normalize.Leave].click()
            buttons[w.Normalize.Standardize].click()

        w = self.widget
        buttons = w.controls.continuous_treatment.buttons
        iris = Table("iris")
        sparse_iris = iris.to_sparse()

        # input dense
        self.send_signal(w.Inputs.data, iris)
        assert_enabled(True)
        self.assertEqual(w.continuous_treatment, w.Normalize.Standardize)

        # input sparse
        self.send_signal(w.Inputs.data, sparse_iris)
        self.assertEqual(w.continuous_treatment, w.Normalize.Scale)
        assert_enabled(False)
        self.assertEqual(w.continuous_treatment, w.Normalize.Leave)

        # remove data
        self.send_signal(w.Inputs.data, None)
        assert_enabled(True)

        # input sparse
        buttons[w.Normalize.Normalize11].click()
        self.send_signal(w.Inputs.data, sparse_iris)
        self.assertEqual(w.continuous_treatment, w.Normalize.Leave)
        assert_enabled(False)

        # input dense
        self.send_signal(w.Inputs.data, iris)
        assert_enabled(True)

    def test_migrate_settings_to_v2(self):
        Normalize = OWContinuize.Normalize

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=0))
        self.assertEqual(widget.continuous_treatment, Normalize.Leave)

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=1, zero_based=True))
        self.assertEqual(widget.continuous_treatment, Normalize.Normalize01)

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=1, zero_based=False))
        self.assertEqual(widget.continuous_treatment, Normalize.Normalize11)

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=2))
        self.assertEqual(widget.continuous_treatment, Normalize.Standardize)

    def test_normalizations(self):
        buttons = self.widget.controls.continuous_treatment.buttons
        Normalize = self.widget.Normalize

        domain = Domain([ContinuousVariable(name) for name in "xyz"])
        col0 = np.arange(0, 10, 2).reshape(5, 1)
        col1 = np.ones((5, 1))
        col2 = np.arange(-2, 3).reshape(5, 1)
        means = np.array([4, 1, 0])
        sds = np.sqrt(np.array([16 + 4 + 0 + 4 + 16, 5, 4 + 1 + 0 + 1 + 4]) / 5)

        x = np.hstack((col0, col1, col2))
        data = Table.from_numpy(domain, x)
        self.send_signal(OWContinuize.Inputs.data, data)

        buttons[Normalize.Leave].click()
        out = self.get_output(self.widget.Outputs.data)
        np.testing.assert_equal(out.X, x)

        buttons[Normalize.Standardize].click()
        out = self.get_output(self.widget.Outputs.data)
        np.testing.assert_almost_equal(out.X, (x - means) / sds)

        buttons[Normalize.Center].click()
        out = self.get_output(self.widget.Outputs.data)
        np.testing.assert_almost_equal(out.X, x - means)

        buttons[Normalize.Scale].click()
        out = self.get_output(self.widget.Outputs.data)
        np.testing.assert_almost_equal(out.X, x / sds)

        buttons[Normalize.Normalize01].click()
        out = self.get_output(self.widget.Outputs.data)
        col = (np.arange(5) / 4).reshape(5, 1)
        np.testing.assert_almost_equal(
            out.X,
            np.hstack((col, np.zeros((5, 1)), col))
        )

        buttons[Normalize.Normalize11].click()
        out = self.get_output(self.widget.Outputs.data)
        col = (np.arange(5) / 2).reshape(5, 1) - 1
        np.testing.assert_almost_equal(
            out.X,
            np.hstack((col, np.zeros((5, 1)), col))
        )

    def test_send_report(self):
        self.widget.send_report()


class TestOWContinuizeUtils(unittest.TestCase):
    def test_dummy_coding_zero_based(self):
        var = DiscreteVariable("foo", values=tuple("abc"))

        varb, varc = owcontinuize.dummy_coding(var)

        self.assertEqual(varb.name, "foo=b")
        self.assertIsInstance(varb.compute_value, transformation.Indicator)
        self.assertEqual(varb.compute_value.value, 1)
        self.assertIs(varb.compute_value.variable, var)

        self.assertEqual(varc.name, "foo=c")
        self.assertIsInstance(varc.compute_value, transformation.Indicator)
        self.assertEqual(varc.compute_value.value, 2)
        self.assertIs(varc.compute_value.variable, var)

    def test_dummy_coding_base_value(self):
        var = DiscreteVariable("foo", values=tuple("abc"))

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
        var = DiscreteVariable("foo", values=tuple("abc"))

        new_vars = owcontinuize.one_hot_coding(var)
        for i, (c, nvar) in enumerate(zip("abc", new_vars)):
            self.assertEqual(nvar.name, f"foo={c}")
            self.assertIsInstance(nvar.compute_value, transformation.Indicator)
            self.assertEqual(nvar.compute_value.value, i)
            self.assertIs(nvar.compute_value.variable, var)


if __name__ == "__main__":
    unittest.main()
