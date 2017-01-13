# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.widgets.data.owimpute import OWImpute
from Orange.widgets.tests.base import WidgetTest


class TestOWImpute(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWImpute)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        widget = self.widget
        widget.default_method_index = widget.MODEL_BASED_IMPUTER
        widget.default_method = widget.METHODS[widget.default_method_index]

        self.send_signal("Data", data)
        widget.unconditional_commit()
        imp_data = self.get_output("Data")
        np.testing.assert_equal(imp_data.X, data.X)
        np.testing.assert_equal(imp_data.Y, data.Y)

        self.send_signal("Data", Table(data.domain))
        widget.unconditional_commit()
        imp_data = self.get_output("Data")
        self.assertEqual(len(imp_data), 0)
