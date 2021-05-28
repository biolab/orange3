# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
from Orange.data import Table
from Orange.widgets.data.owdatainfo import OWDataInfo
from Orange.widgets.tests.base import WidgetTest


class TestOWDataInfo(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDataInfo)

    def test_data(self):
        """No crash on iris"""
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(data.domain))

    def test_data_attributes(self):
        """No crash on data attributes of different types"""
        data = Table("iris")
        data.attributes = {"att 1": 1, "att 2": True, "att 3": 3}
        self.send_signal(self.widget.Inputs.data, data)
