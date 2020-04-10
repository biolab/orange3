# pylint: disable=unsubscriptable-object
import unittest
from unittest.mock import Mock

from Orange.data import Table
from Orange.widgets.data.owpurgedomain import OWPurgeDomain
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.state_summary import format_summary_details


class TestOWPurgeDomain(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPurgeDomain)
        self.iris = Table("iris")

    def test_summary(self):
        """Check if the status bar is updated when data is received"""
        data = self.iris
        input_sum = self.widget.info.set_input_summary = Mock()
        output_sum = self.widget.info.set_output_summary = Mock()

        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(len(data), format_summary_details(data))
        output = self.get_output(self.widget.Outputs.data)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))
        input_sum.reset_mock()
        output_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_once()
        self.assertEqual(input_sum.call_args[0][0].brief, "")
        output_sum.assert_called_once()
        self.assertEqual(output_sum.call_args[0][0].brief, "")

    def test_minimum_size(self):
        pass


if __name__ == "__main__":
    unittest.main()
