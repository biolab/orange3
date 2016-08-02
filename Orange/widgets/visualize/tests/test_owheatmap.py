# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.preprocess import Continuize
from Orange.widgets.visualize.owheatmap import OWHeatMap
from Orange.widgets.tests.base import WidgetTest


class TestOWHeatMap(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWHeatMap)
        self.iris = Table("iris")
        self.housing = Table("housing")
        self.titanic = Table("titanic")

    def test_input_data(self):
        """Check widget's data with data on the input"""
        for data in (self.iris, self.housing):
            self.assertEqual(self.widget.data, None)
            self.send_signal("Data", data)
            self.assertEqual(self.widget.data, data)
            self.assertFalse(self.widget.Error.active)
            self.assertFalse(self.widget.Warning.active)
            self.assertFalse(self.widget.Information.active)
            self.send_signal("Data", None)

    def test_error_message(self):
        self.send_signal("Data", self.titanic)
        self.assertTrue(self.widget.Error.active)
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Error.active)

    def test_information_message(self):
        self.widget.sort_rows = self.widget.OrderedClustering
        continuizer = Continuize()
        cont_titanic = continuizer(self.titanic)
        self.send_signal("Data", cont_titanic)
        self.assertTrue(self.widget.Information.active)
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Information.active)
