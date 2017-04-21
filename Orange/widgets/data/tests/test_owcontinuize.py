# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.widgets.data.owcontinuize import OWContinuize
from Orange.widgets.tests.base import WidgetTest


class TestOWContinuize(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWContinuize)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        widget = self.widget
        widget.multinomial_treatment = 1

        self.send_signal("Data", data)
        widget.unconditional_commit()
        imp_data = self.get_output("Data")
        np.testing.assert_equal(imp_data.X, data.X)
        np.testing.assert_equal(imp_data.Y, data.Y)

        widget.continuous_treatment = 1
        self.send_signal("Data", Table(data.domain))
        widget.unconditional_commit()
        imp_data = self.get_output("Data")
        self.assertEqual(len(imp_data), 0)

        self.send_signal("Data", None)
        widget.unconditional_commit()
        imp_data = self.get_output("Data")
        self.assertIsNone(imp_data)

    def test_one_column_equal_values(self):
        """
        No crash on a column with equal values and with selected option
        normalize by standard deviation.
        GH-2144
        """
        table = Table("iris")
        table = table[:, 1]
        table[:] = 42.0
        self.send_signal("Data", table)
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
        self.send_signal("Data", table)
        # Normalize.NormalizeBySD
        self.widget.continuous_treatment = 2
        self.widget.unconditional_commit()
        table = Table("iris")
        table[1, 2] = np.NaN
        self.send_signal("Data", table)
        self.widget.unconditional_commit()


    def test_one_column_nan_values_normalize_span(self):
        """
        No crash on a column with NaN values and with selected option
        normalize by span.
        GH-2144
        """
        table = Table("iris")
        table[:, 2] = np.NaN
        self.send_signal("Data", table)
        # Normalize.NormalizeBySpan
        self.widget.continuous_treatment = 1
        self.widget.unconditional_commit()
        table = Table("iris")
        table[1, 2] = np.NaN
        self.send_signal("Data", table)
        self.widget.unconditional_commit()
