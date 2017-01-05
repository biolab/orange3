# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from Orange.data import Table
from Orange.widgets.data.owconcatenate import OWConcatenate
from Orange.widgets.tests.base import WidgetTest


class TestOWConcatenate(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWConcatenate)
        self.iris = Table("iris")

    def test_single_input(self):
        self.assertIsNone(self.get_output("Data"))
        self.send_signal("Primary Data", self.iris)
        output = self.get_output("Data")
        self.assertEqual(list(output), list(self.iris))
        self.send_signal("Primary Data", None)
        self.assertIsNone(self.get_output("Data"))
        self.send_signal("Additional Data", self.iris)
        output = self.get_output("Data")
        self.assertEqual(list(output), list(self.iris))
        self.send_signal("Additional Data", None)
        self.assertIsNone(self.get_output("Data"))