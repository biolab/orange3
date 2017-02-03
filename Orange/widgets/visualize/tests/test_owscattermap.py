# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.visualize.owscattermap import OWScatterMap


class TestOWScatterMap(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWScatterMap)

    def test_input_single_instance(self):
        """Check widget for single instance on the input"""
        self.send_signal("Data", self.data[0:1])

    def test_input_constant_feature(self):
        """Check widget for input data with constant feature"""
        data = self.data.copy()
        data.X[:, 0] = 1
        self.send_signal("Data", data)

    def test_input_missing_value(self):
        """Check widget for input data with missing value"""
        data = self.data.copy()
        data.X[0, 0] = np.nan
        self.send_signal("Data", data)
        self.assertFalse(self.widget.Error.no_values.is_shown())

    def test_input_missing_values(self):
        """Check widget for input data with feature with no values"""
        data = self.data.copy()
        data.X[:, 0] = np.nan
        self.send_signal("Data", data)
        self.assertTrue(self.widget.Error.no_values.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.no_values.is_shown())
