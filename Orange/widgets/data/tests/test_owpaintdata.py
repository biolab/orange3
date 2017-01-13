# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.data.owpaintdata import OWPaintData
from Orange.widgets.tests.base import WidgetTest


class TestOWPaintData(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPaintData)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        self.send_signal("Data", data)
        self.send_signal("Data", Table(data.domain))
