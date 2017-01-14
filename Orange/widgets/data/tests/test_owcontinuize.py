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
