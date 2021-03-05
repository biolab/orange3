# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
from unittest.mock import Mock

from orangewidget.widget import StateInfo

from Orange.data import Table
from Orange.widgets.data.owdatainfo import OWDataInfo
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.state_summary import format_summary_details


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

    def test_summary(self):
        """Check if the status bar is updated when data is received"""
        data = Table("iris")
        input_sum = self.widget.info.set_input_summary = Mock()
        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(len(data), format_summary_details(data))
        input_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_once()
        self.assertIsInstance(input_sum.call_args[0][0], StateInfo.Empty)
