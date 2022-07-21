# pylint: disable=protected-access
import unittest
from unittest.mock import patch

from AnyQt.QtCore import Qt

from Orange.data.dask import DaskTable
from orangewidget.tests.utils import excepthook_catch

from Orange.widgets.data.owtable import OWTable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.data import Table, Domain
from Orange.data.sql.table import SqlTable
from Orange.tests.sql.base import DataBaseTest as dbt
from Orange.tests.test_dasktable import open_as_dask


class TestOWTable(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls,
                                    output_all_on_no_selection=True)

        cls.signal_name = OWTable.Inputs.data
        cls.signal_data = cls.data  # pylint: disable=no-member

    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWTable)

    def test_input_data(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIs(self.widget.input.table, self.data)
        self.assertIs(self.widget.view.model().source, self.data)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.input)
        self.assertIsNone(self.widget.view.model())

    def test_input_data_empty(self):
        self.send_signal(self.widget.Inputs.data, self.data[:0])
        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output), 0)

    def test_data_model(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(self.widget.view.model().rowCount(), len(self.data))

    def test_reset_select(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self._select_data()
        self.send_signal(self.widget.Inputs.data, Table('heart_disease'))
        self.assertListEqual([], self.widget.stored_selection["columns"])
        self.assertListEqual([], self.widget.stored_selection["rows"])

    def _select_data(self):
        self.widget.set_selection(
            list(range(0, len(self.data), 10)),
            list(range(len(self.data.domain.variables))),
        )
        return self.widget.stored_selection["rows"]

    def test_attrs_appear_in_corner_text(self):
        domain = self.data.domain
        new_domain = Domain(
            domain.attributes[1:], domain.class_var, domain.attributes[:1])
        new_domain.metas[0].attributes = {"c": "foo"}
        new_domain.attributes[0].attributes = {"a": "bar", "c": "baz"}
        new_domain.class_var.attributes = {"b": "foo"}
        self.send_signal(self.widget.Inputs.data, self.data.transform(new_domain))
        self.assertEqual(self.widget.view.cornerText(), "\na\nb\nc")

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as commit:
            self.widget.auto_commit = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.data)
            commit.assert_called()

    def test_pending_selection(self):
        widget = self.create_widget(OWTable, stored_settings={
            "stored_selection": {
                "rows": [5, 6, 7, 8, 9],
                "columns": list(range(len(self.data.domain.variables)))
            }
        })
        self.send_signal(widget.Inputs.data, None)
        self.send_signal(widget.Inputs.data, self.data)
        output = self.get_output(widget.Outputs.selected_data)
        self.assertEqual(5, len(output))

    def test_pending_sorted_selection(self):
        rows = [5, 6, 7, 8, 9, 55, 56, 57, 58, 59]
        widget = self.create_widget(OWTable, stored_settings={
            "stored_selection": {
                "rows": rows,
                "columns": list(range(len(self.data.domain.variables)))
            },
            "stored_sort": [("sepal length", 1), ("sepal width", -1)]
        })
        self.send_signal(widget.Inputs.data, None)
        self.send_signal(widget.Inputs.data, self.data)
        self.assertEqual(widget.view.horizontalHeader().sortIndicatorOrder(),
                         Qt.DescendingOrder)
        self.assertEqual(widget.view.horizontalHeader().sortIndicatorSection(), 2)
        output = self.get_output(widget.Outputs.selected_data)
        self.assertEqual(len(rows), len(output))
        sepal_width = output.get_column("sepal width").tolist()
        sepal_length = output.get_column("sepal length").tolist()
        self.assertSequenceEqual(sepal_width, sorted(sepal_width, reverse=True))
        dd = list(zip(sepal_length, sepal_width))
        dd_sorted = sorted(dd, key=lambda t: t[0])
        dd_sorted = sorted(dd_sorted, key=lambda t: t[1], reverse=True)
        self.assertSequenceEqual(dd, dd_sorted)
        ids = self.data[rows].ids
        self.assertSetEqual(set(output.ids), set(ids))

    def test_missing_sort_column_shows_warning(self):
        widget = self.create_widget(OWTable, stored_settings={
            "stored_sort": [("sepal length", 1), ("no such column", -1)]
        })
        self.send_signal(widget.Inputs.data, self.data)
        self.assertTrue(widget.Warning.missing_sort_columns.is_shown())
        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Warning.missing_sort_columns.is_shown())

    def test_sorting(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.set_selection(
            [0, 1, 2, 3, 4],
            list(range(len(self.data.domain.variables)))
        )
        output = self.get_output(self.widget.Outputs.selected_data)
        output = output.get_column(0)
        output_original = output.tolist()

        self.widget.view.sortByColumn(1, Qt.AscendingOrder)
        self.assertEqual(self.widget.stored_sort, [('sepal length', 1)])
        output = self.get_output(self.widget.Outputs.selected_data)
        output = output.get_column(0)
        output_sorted = output.tolist()

        # the two outputs should not be the same.
        self.assertTrue(output_original != output_sorted)

        # check if output after sorting is actually sorted.
        self.assertTrue(sorted(output_original) == output_sorted)
        self.assertTrue(sorted(output_sorted) == output_sorted)

        self.widget.restore_order()
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(output.get_column(0).tolist(), output_original)

        # Check that output is the same with no sorting and cleared selection.
        self.widget.set_selection([], [])
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertIs(output, self.data)

    def test_sort_basket_column(self):
        data = self.data.to_sparse()
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.view.sortByColumn(0, Qt.AscendingOrder)
        self.assertEqual(self.widget.stored_sort, [("iris", 1)])
        self.widget.view.sortByColumn(1, Qt.AscendingOrder)
        self.assertEqual(self.widget.stored_sort,
                         [("iris", 1), ("\\BASKET(FEATURES)", 1)])
        # test restore
        w = self.create_widget(OWTable, stored_settings={
            "stored_sort": self.widget.stored_sort
        })
        self.send_signal(w.Inputs.data, data)
        self.assertEqual(w.stored_sort, self.widget.stored_sort)
        output_a = self.get_output(self.widget.Outputs.selected_data)
        output_b = self.get_output(w.Outputs.selected_data)
        self.assertEqual(output_a.ids.tolist(), output_b.ids.tolist())

    def test_info(self):
        info_text = self.widget.info_text
        no_input = "No data."
        self.assertEqual(info_text.text(), no_input)

    def test_show_distributions(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
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
        self.send_signal(w.Inputs.data, self.data)
        self.assertTrue(w.select_rows)  # default value
        with excepthook_catch():
            w.controls.select_rows.toggle()
        self.assertFalse(w.select_rows)
        w.set_selection([0, 1, 2, 3], [0, 1])
        out = self.get_output(w.Outputs.selected_data)
        self.assertEqual(out.domain,
                         Domain([self.data.domain.attributes[0]], self.data.domain.class_var))
        with excepthook_catch():
            w.controls.select_rows.toggle()
        out = self.get_output(w.Outputs.selected_data)
        self.assertTrue(w.select_rows)
        self.assertEqual(out.domain, self.data.domain)

    def test_show_attribute_labels(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        self.assertTrue(w.show_attribute_labels)  # default value
        with excepthook_catch():
            w.controls.show_attribute_labels.toggle()
        self.assertFalse(w.show_attribute_labels)

    def test_subset_input(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        with patch.object(w.signalManager, "send") as m:
            self.send_signal(w.Inputs.data_subset, self.data[[0, 1, 5]])
            m.assert_not_called()
        w.view.grab()  # cover delegate painting methods

        model = w.view.model()
        self.assertTrue(model.index(0, 0).data(model.SubsetRole))
        self.assertFalse(model.index(2, 0).data(model.SubsetRole))
        self.assertTrue(model.headerData(0, Qt.Vertical, model.SubsetRole))
        self.assertFalse(model.headerData(2, Qt.Vertical, model.SubsetRole))

        with patch.object(w.signalManager, "send") as m:
            self.send_signal(w.Inputs.data_subset, None)
            m.assert_not_called()

        w.view.grab()

        model = w.view.model()
        self.assertFalse(model.index(0, 0).data(model.SubsetRole))
        self.assertFalse(model.headerData(0, Qt.Vertical, model.SubsetRole))

    def test_dask(self):
        w = self.widget
        with open_as_dask("zoo") as zoo:
            self.send_signal(w.Inputs.data, zoo)
            selected = self.get_output(w.Outputs.selected_data)
            self.assertIsInstance(selected, DaskTable)
            annotated = self.get_output(w.Outputs.annotated_data)
            self.assertIsInstance(annotated, DaskTable)


class TestOWTableSQL(TestOWTable, dbt):
    def setUpDB(self):
        # pylint: disable=attribute-defined-outside-init
        conn, iris = self.create_iris_sql_table()
        data = SqlTable(conn, iris, inspect_values=True)
        self.data = data.transform(Domain(data.domain.attributes[:-1],
                                          data.domain.attributes[-1]))

    def tearDownDB(self):
        self.drop_iris_sql_table()

    @dbt.run_on(["postgres", "mssql"])
    def test_input_data(self):
        super().test_input_data()

    @unittest.skip("no data output")
    def test_input_data_empty(self):
        super().test_input_data_empty()

    @unittest.skip("approx_len messes up row count")
    def test_data_model(self):
        super().test_data_model()

    @dbt.run_on(["postgres", "mssql"])
    def test_unconditional_commit_on_new_signal(self):
        super().test_unconditional_commit_on_new_signal()

    @dbt.run_on(["postgres", "mssql"])
    def test_reset_select(self):
        super().test_reset_select()

    @dbt.run_on(["postgres", "mssql"])
    def test_attrs_appear_in_corner_text(self):
        super().test_attrs_appear_in_corner_text()

    @unittest.skip("no data output")
    def test_pending_selection(self):
        super().test_pending_selection()

    @unittest.skip("sorting not implemented")
    def test_sorting(self):
        super().test_sorting()

    @unittest.skip("does nothing")
    def test_info(self):
        super().test_info()

    @dbt.run_on(["postgres", "mssql"])
    def test_show_distributions(self):
        super().test_show_distributions()

    @unittest.skip("no data output")
    def test_whole_rows(self):
        super().test_whole_rows()

    @dbt.run_on(["postgres", "mssql"])
    def test_show_attribute_labels(self):
        super().test_show_distributions()


if __name__ == "__main__":
    unittest.main()
