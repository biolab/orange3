# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.visualize.owheatmap import OWHeatMap
from Orange.widgets.tests.base import WidgetTest


class TestOWHeatMap(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWHeatMap)
        self.iris = Table("iris")
        self.housing = Table("housing")

    def test_input_data(self):
        """Check widget's data with data on the input"""
        for data in (self.iris, self.housing):
            self.assertEqual(self.widget.data, None)
            self.send_signal("Data", data)
            self.assertEqual(self.widget.data, data)
            self.send_signal("Data", None)
