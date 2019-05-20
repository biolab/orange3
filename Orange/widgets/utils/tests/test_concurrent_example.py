# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import unittest
from unittest.mock import Mock

from Orange.data import Table
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.utils.tests.concurrent_example import (
    OWConcurrentWidget
)


class TestOWConcurrentWidget(WidgetTest, ProjectionWidgetTestMixin,
                             WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWConcurrentWidget)

    def test_button_no_data(self):
        self.widget.run_button.click()
        self.assertEqual(self.widget.run_button.text(), "Start")

    def test_button_with_data(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(self.widget.run_button.text(), "Stop")
        self.wait_until_stop_blocking()
        self.assertEqual(self.widget.run_button.text(), "Start")

    def test_button_toggle(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.run_button.click()
        self.assertEqual(self.widget.run_button.text(), "Resume")

    def test_plot_once(self):
        table = Table("heart_disease")
        self.widget.setup_plot = Mock()
        self.widget.commit = Mock()
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.assert_called_once()
        self.wait_until_stop_blocking()
        self.widget.setup_plot.reset_mock()
        self.widget.commit.reset_mock()
        self.send_signal(self.widget.Inputs.data_subset, table[::10])
        self.widget.setup_plot.assert_not_called()
        self.widget.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
