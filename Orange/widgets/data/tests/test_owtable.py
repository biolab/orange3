import unittest
from unittest.mock import Mock, patch

from Orange.widgets.data.owtable import OWDataTable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.data import Table, Domain


class TestOWDataTable(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data  # pylint: disable=no-member

    def setUp(self):
        self.widget = self.create_widget(OWDataTable)

    def test_input_data(self):
        """Check number of tabs with data on the input"""
        self.send_signal(self.widget.Inputs.data, self.data, 1)
        self.assertEqual(self.widget.tabs.count(), 1)
        self.send_signal(self.widget.Inputs.data, self.data, 2)
        self.assertEqual(self.widget.tabs.count(), 2)
        self.send_signal(self.widget.Inputs.data, None, 1)
        self.assertEqual(self.widget.tabs.count(), 1)

    def test_data_model(self):
        self.send_signal(self.widget.Inputs.data, self.data, 1)
        self.assertEqual(self.widget.tabs.widget(0).model().rowCount(),
                         len(self.data))

    def test_reset_select(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self._select_data()
        self.send_signal(self.widget.Inputs.data, Table('heart_disease'))
        self.assertListEqual([], self.widget.selected_cols)
        self.assertListEqual([], self.widget.selected_rows)

    def _select_data(self):
        self.widget.selected_cols = list(range(len(self.data.domain)))
        self.widget.selected_rows = list(range(0, len(self.data.domain), 10))
        self.widget.set_selection()
        return self.widget.selected_rows

    def test_attrs_appear_in_corner_text(self):
        iris = Table("iris")
        domain = iris.domain
        new_domain = Domain(
            domain.attributes[1:], iris.domain.class_var, domain.attributes[:1])
        new_domain.metas[0].attributes = {"c": "foo"}
        new_domain.attributes[0].attributes = {"a": "bar", "c": "baz"}
        new_domain.class_var.attributes = {"b": "foo"}
        self.widget.set_corner_text = Mock()
        self.send_signal(self.widget.Inputs.data, iris.transform(new_domain))
        # false positive, pylint: disable=unsubscriptable-object
        self.assertEqual(
            self.widget.set_corner_text.call_args[0][1], "\na\nb\nc")

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget, 'unconditional_commit') as commit:
            self.widget.auto_commit = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.data)
            commit.assert_called()

    def test_pending_selection(self):
        widget = self.create_widget(OWDataTable, stored_settings=dict(
            selected_rows=[5, 6, 7, 8, 9],
            selected_cols=list(range(len(self.data.domain)))))
        self.send_signal(widget.Inputs.data, None, 1)
        self.send_signal(widget.Inputs.data, self.data, 1)
        output = self.get_output(widget.Outputs.selected_data)
        self.assertEqual(5, len(output))


if __name__ == "__main__":
    unittest.main()
