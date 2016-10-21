# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.data.owtable import OWDataTable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin


class TestOWDataTable(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWDataTable)

    def test_input_data(self):
        """Check number of tabs with data on the input"""
        self.send_signal("Data", self.data, 1)
        self.assertEqual(self.widget.tabs.count(), 1)
        self.send_signal("Data", self.data, 2)
        self.assertEqual(self.widget.tabs.count(), 2)
        self.send_signal("Data", None, 1)
        self.assertEqual(self.widget.tabs.count(), 1)

    def test_data_model(self):
        self.send_signal("Data", self.data, 1)
        self.assertEqual(self.widget.tabs.widget(0).model().rowCount(),
                         len(self.data))

    def _select_data(self):
        self.widget.selected_cols = list(range(len(self.data.domain)))
        self.widget.selected_rows = list(range(0, len(self.data.domain), 10))
        self.widget.set_selection()
        return self.widget.selected_rows
