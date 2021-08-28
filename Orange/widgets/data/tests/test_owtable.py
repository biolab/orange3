# pylint: disable=protected-access
import unittest


from unittest.mock import Mock, patch
from AnyQt.QtCore import Qt

from orangewidget.tests.utils import excepthook_catch
from orangewidget.widget import StateInfo

from Orange.widgets.data.owtable import OWDataTable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.data import Table, Domain
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.data.sql.table import SqlTable
from Orange.tests.sql.base import DataBaseTest as dbt


class TestOWDataTable(WidgetTest, WidgetOutputsTestMixin, dbt):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls,
                                    output_all_on_no_selection=True)

        cls.signal_name = "Data"
        cls.signal_data = cls.data  # pylint: disable=no-member

    def setUp(self):
        self.widget = self.create_widget(OWDataTable)

    def setUpDB(self):
        # pylint: disable=attribute-defined-outside-init
        conn, self.iris = self.create_iris_sql_table()
        self.table = SqlTable(conn, self.iris)

    def tearDownDB(self):
        self.drop_iris_sql_table()

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
        self.widget.selected_cols = list(range(len(self.data.domain.variables)))
        self.widget.selected_rows = list(range(0, len(self.data), 10))
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
        with patch.object(self.widget.commit, 'now') as commit:
            self.widget.auto_commit = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.data)
            commit.assert_called()

    def test_pending_selection(self):
        widget = self.create_widget(OWDataTable, stored_settings=dict(
            selected_rows=[5, 6, 7, 8, 9],
            selected_cols=list(range(len(self.data.domain.variables)))))
        self.send_signal(widget.Inputs.data, None, 1)
        self.send_signal(widget.Inputs.data, self.data, 1)
        output = self.get_output(widget.Outputs.selected_data)
        self.assertEqual(5, len(output))

    def test_sorting(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.selected_rows = [0, 1, 2, 3, 4]
        self.widget.selected_cols = list(range(len(self.data.domain.variables)))
        self.widget.set_selection()

        output = self.get_output(self.widget.Outputs.selected_data)
        output, _ = output.get_column_view(0)
        output_original = output.tolist()

        self.widget.tabs.currentWidget().sortByColumn(1, Qt.AscendingOrder)

        output = self.get_output(self.widget.Outputs.selected_data)
        output, _ = output.get_column_view(0)
        output_sorted = output.tolist()

        # the two outputs should not be the same.
        self.assertTrue(output_original != output_sorted)

        # check if output after sorting is actually sorted.
        self.assertTrue(sorted(output_original) == output_sorted)
        self.assertTrue(sorted(output_sorted) == output_sorted)

    def test_summary(self):
        """Check if status bar is updated when data is received"""
        info = self.widget.info
        no_input, no_output = "No data on input", "No data on output"

        self.assertIsInstance(info._StateInfo__input_summary, StateInfo.Empty)
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
        self.assertIsInstance(info._StateInfo__output_summary, StateInfo.Empty)
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

        data = Table("zoo")
        self.send_signal(self.widget.Inputs.data, data, 1)
        summary, details = f"{len(data)}", format_summary_details(data)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)

        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data, 2)
        summary, details = f"{len(data)}", format_summary_details(data)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)

        self.send_signal(self.widget.Inputs.data, None, 1)
        summary, details = f"{len(data)}", format_summary_details(data)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)

        self.send_signal(self.widget.Inputs.data, None, 2)
        self.assertIsInstance(info._StateInfo__input_summary, StateInfo.Empty)
        self.assertEqual(info._StateInfo__input_summary.details, no_input)

    def test_info(self):
        info_text = self.widget.info_text
        no_input = "No data."
        self.assertEqual(info_text.text(), no_input)

    def test_show_distributions(self):
        w = self.widget
        data = Table("heart_disease")[::3].copy()
        self.send_signal(w.Inputs.data, data, 0)
        # run through the delegate paint routines
        with excepthook_catch():
            w.grab()
        w.controls.show_distributions.toggle()
        with excepthook_catch():
            w.grab()
        w.controls.color_by_class.toggle()
        with excepthook_catch():
            w.grab()
        w.controls.show_distributions.toggle()

    def test_whole_rows(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data, 0)
        self.assertTrue(w.select_rows)  # default value
        with excepthook_catch():
            w.controls.select_rows.toggle()
        self.assertFalse(w.select_rows)
        w.selected_cols = [0, 1]
        w.selected_rows = [0, 1, 2, 3]
        w.set_selection()
        out = self.get_output(w.Outputs.selected_data)
        self.assertEqual(out.domain,
                         Domain([self.data.domain.attributes[0]], self.data.domain.class_var))
        with excepthook_catch():
            w.controls.select_rows.toggle()
        out = self.get_output(w.Outputs.selected_data)
        self.assertTrue(w.select_rows)
        self.assertEqual(out.domain,
                         self.data.domain)

    def test_show_attribute_labels(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data, 0)
        self.assertTrue(w.show_attribute_labels)  # default value
        with excepthook_catch():
            w.controls.show_attribute_labels.toggle()
        self.assertFalse(w.show_attribute_labels)


if __name__ == "__main__":
    unittest.main()
