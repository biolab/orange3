import os
import unittest
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from AnyQt import QtCore
from AnyQt.QtCore import QItemSelectionModel, Qt
from AnyQt.QtWidgets import QListView

from Orange.data import (
    Table,
    table_to_frame,
    Domain,
    ContinuousVariable,
    DiscreteVariable,
    TimeVariable,
    StringVariable,
)
from Orange.data.tests.test_aggregate import create_sample_data
from Orange.widgets.data.owgroupby import OWGroupBy
from Orange.widgets.tests.base import WidgetTest


class TestOWGroupBy(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWGroupBy)
        self.iris = Table("iris")

        self.data = create_sample_data()

    def test_none_data(self):
        self.send_signal(self.widget.Inputs.data, None)

        self.assertEqual(self.widget.agg_table_model.rowCount(), 0)
        self.assertEqual(self.widget.gb_attrs_model.rowCount(), 0)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_data(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        self.assertEqual(self.widget.agg_table_model.rowCount(), 5)
        self.assertEqual(self.widget.gb_attrs_model.rowCount(), 5)

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(output))

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_data_domain_changed(self):
        self.send_signal(self.widget.Inputs.data, self.iris[:, -2:])
        self.assert_aggregations_equal(["Mean", "Mode"])

        self.send_signal(self.widget.Inputs.data, self.iris[:, -3:])
        self.assert_aggregations_equal(["Mean", "Mean", "Mode"])
        self.select_table_rows(self.widget.agg_table_view, [0])

    @staticmethod
    def _set_selection(view: QListView, indices: List[int]):
        view.clearSelection()
        sm = view.selectionModel()
        model = view.model()
        for ind in indices:
            sm.select(model.index(ind, 0), QItemSelectionModel.Select)

    def test_groupby_attr_selection(self):
        gb_view = self.widget.controls.gb_attrs
        self.send_signal(self.widget.Inputs.data, self.iris)

        self._set_selection(gb_view, [1])  # sepal length
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(35, len(output))

        # select iris attribute with index 0
        self._set_selection(gb_view, [0])
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(output))

        # select iris and sepal length attribute
        self._set_selection(gb_view, [0, 1])
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(57, len(output))

    def assert_enabled_cbs(self, enabled_true):
        enabled_actual = set(
            name for name, cb in self.widget.agg_checkboxes.items() if cb.isEnabled()
        )
        self.assertSetEqual(enabled_true, enabled_actual)

    @staticmethod
    def select_table_rows(table, rows):
        table.clearSelection()
        indexes = [table.model().index(r, 0) for r in rows]
        mode = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        for i in indexes:
            table.selectionModel().select(i, mode)

    def test_attr_table_row_selection(self):
        # fmt: off
        continuous_aggs = {
            "Mean", "Median", "Q1", "Q3", "Min. value", "Max. value", "Mode", "Sum",
            "Standard deviation", "Variance", "Count defined", "Count", "Concatenate",
            "Span", "First value", "Last value", "Random value", "Proportion defined",
        }
        discrete_aggs = {
            "Mode", "Count defined", "Count", "Concatenate", "First value",
            "Last value", "Random value", "Proportion defined"
        }
        string_aggs = {
            "Count defined", "Count", "Concatenate", "First value",
            "Last value", "Random value", "Proportion defined"
        }
        # fmt: on
        self.send_signal(self.widget.Inputs.data, self.data)

        model = self.widget.agg_table_model
        table = self.widget.agg_table_view

        self.assertListEqual(
            ["a", "b", "cvar", "dvar", "svar"],
            [model.data(model.index(i, 0)) for i in range(model.rowCount())],
        )

        self.select_table_rows(table, [0])
        self.assert_enabled_cbs(continuous_aggs)
        self.select_table_rows(table, [0, 1])
        self.assert_enabled_cbs(continuous_aggs)
        self.select_table_rows(table, [2])
        self.assert_enabled_cbs(continuous_aggs)
        self.select_table_rows(table, [3])  # discrete variable
        self.assert_enabled_cbs(discrete_aggs)
        self.select_table_rows(table, [4])  # string variable
        self.assert_enabled_cbs(string_aggs)
        self.select_table_rows(table, [3, 4])  # discrete + string variable
        self.assert_enabled_cbs(string_aggs | discrete_aggs)
        self.select_table_rows(table, [2, 3, 4])  # cont + disc + str variable
        self.assert_enabled_cbs(string_aggs | discrete_aggs | continuous_aggs)

    def assert_aggregations_equal(self, expected_text):
        model = self.widget.agg_table_model
        agg_text = [model.data(model.index(i, 1)) for i in range(model.rowCount())]
        self.assertListEqual(expected_text, agg_text)

    def test_aggregations_change(self):
        table = self.widget.agg_table_view
        d = self.data.domain

        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["Mean", "Mean", "Mean", "Mode", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0])
        self.widget.agg_checkboxes["Median"].click()
        self.assert_aggregations_equal(
            ["Mean, Median", "Mean", "Mean", "Mode", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1])
        self.widget.agg_checkboxes["Mode"].click()
        self.assert_aggregations_equal(
            ["Mean, Median, Mode", "Mean, Mode", "Mean", "Mode", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1])
        # median is partially checked and will become checked
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["Median"].checkState()
        )
        self.widget.agg_checkboxes["Median"].click()
        self.assertEqual(Qt.Checked, self.widget.agg_checkboxes["Median"].checkState())
        self.assert_aggregations_equal(
            [
                "Mean, Median, Mode",
                "Mean, Median, Mode",
                "Mean",
                "Mode",
                "Concatenate",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median", "Mode"},
                d["b"]: {"Mean", "Median", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["Median"].click()
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["Median"].checkState()
        )
        self.assert_aggregations_equal(
            ["Mean, Mode", "Mean, Mode", "Mean", "Mode", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 3])
        # median is unchecked and will change to partially checked
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["Median"].checkState()
        )
        self.widget.agg_checkboxes["Median"].click()
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["Median"].checkState()
        )
        self.assert_aggregations_equal(
            ["Mean, Median, Mode", "Mean, Mode", "Mean", "Mode", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["Median"].click()
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["Median"].checkState()
        )
        self.assert_aggregations_equal(
            ["Mean, Mode", "Mean, Mode", "Mean", "Mode", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["Count"].click()
        self.assertEqual(Qt.Checked, self.widget.agg_checkboxes["Count"].checkState())
        self.assert_aggregations_equal(
            [
                "Mean, Mode, Count",
                "Mean, Mode",
                "Mean",
                "Mode, Count",
                "Concatenate",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode", "Count"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        # test the most complicated scenario: numeric with mode, numeric without
        # mode and discrete
        self.select_table_rows(table, [0])
        self.widget.agg_checkboxes["Mode"].click()
        self.assert_aggregations_equal(
            ["Mean, Count", "Mean, Mode", "Mean", "Mode, Count", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Count"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1, 4])
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["Mode"].checkState()
        )
        self.widget.agg_checkboxes["Mode"].click()
        # must stay partially checked since one Continuous can still have mode
        # as a aggregation and string cannot have it
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["Mode"].checkState()
        )
        self.assert_aggregations_equal(
            [
                "Mean, Mode, Count",
                "Mean, Mode",
                "Mean",
                "Mode, Count",
                "Concatenate",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode", "Count"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        # since now all that can have Mode have it as an aggregation it can be
        # unchecked on the next click
        self.widget.agg_checkboxes["Mode"].click()
        self.assertEqual(Qt.Unchecked, self.widget.agg_checkboxes["Mode"].checkState())
        self.assert_aggregations_equal(
            ["Mean, Count", "Mean", "Mean", "Mode, Count", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Count"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["Mode"].click()
        self.widget.agg_checkboxes["Count defined"].click()
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["Mode"].checkState()
        )
        self.assert_aggregations_equal(
            [
                "Mean, Mode, Count defined and 1 more",
                "Mean, Mode, Count defined",
                "Mean",
                "Mode, Count",
                "Concatenate, Count defined",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode", "Count", "Count defined"},
                d["b"]: {"Mean", "Mode", "Count defined"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Mode"},
                d["svar"]: {"Concatenate", "Count defined",},
            },
            self.widget.aggregations,
        )

    def test_aggregation(self):
        """Test aggregation results"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self._set_selection(self.widget.controls.gb_attrs, [1])  # a var
        output = self.get_output(self.widget.Outputs.data)

        np.testing.assert_array_almost_equal(
            output.X, [[1, 2.143, 0.317, 0], [2, 2, 2, 0]], decimal=3
        )
        np.testing.assert_array_equal(
            output.metas,
            np.array(
                [
                    [
                        "sval1 sval2 sval2 sval1 sval2 sval1",
                        1.0,
                    ],
                    [
                        "sval2 sval1 sval2 sval1 sval2 sval1",
                        2.0,
                    ],
                ],
                dtype=object,
            ),
        )

        # select all aggregations for all features except a and b
        self._set_selection(self.widget.controls.gb_attrs, [1, 2])
        self.select_table_rows(self.widget.agg_table_view, [2, 3, 4])
        # select all aggregations
        for cb in self.widget.agg_checkboxes.values():
            cb.click()
            while not cb.isChecked():
                cb.click()

        self.select_table_rows(self.widget.agg_table_view, [0, 1])
        # unselect all aggregations for attr a and b
        for cb in self.widget.agg_checkboxes.values():
            while cb.isChecked():
                cb.click()

        expected_columns = [
            "cvar - Mean",
            "cvar - Median",
            "cvar - Q1",
            "cvar - Q3",
            "cvar - Min. value",
            "cvar - Max. value",
            "cvar - Mode",
            "cvar - Standard deviation",
            "cvar - Variance",
            "cvar - Sum",
            "cvar - Span",
            "cvar - First value",
            "cvar - Last value",
            "cvar - Count defined",
            "cvar - Count",
            "cvar - Proportion defined",
            "dvar - Mode",
            "dvar - First value",
            "dvar - Last value",
            "dvar - Count defined",
            "dvar - Count",
            "dvar - Proportion defined",
            "svar - First value",
            "svar - Last value",
            "svar - Count defined",
            "svar - Count",
            "svar - Proportion defined",
            "cvar - Concatenate",
            "dvar - Concatenate",
            "svar - Concatenate",
            "a",  # groupby variables are last two in metas
            "b",
        ]

        # fmt: off
        expected_df = pd.DataFrame([
            [.15, .15, .125, .175, .1, .2, .1, .07, .005, .3, .1, 0.1, 0.2, 2, 2, 1,
             "val1", "val1", "val2", 2, 2, 1,
             "sval1", "sval2", 2, 2, 1,
             "0.1 0.2", "val1 val2", "sval1 sval2",
             1, 1],
            [.3, .3, .3, .3, .3, .3, .3, np.nan, np.nan, .3, 0, .3, .3, 1, 2, 0.5,
             "val2", "val2", "val2", 1, 2, 0.5,
             "", "sval2", 2, 2, 1,
             "0.3", "val2", "sval2",
             1, 2],
            [.433, .4, .35, .5, .3, .6, .3, 0.153, 0.023, 1.3, .3, .3, .6, 3, 3, 1,
             "val1", "val1", "val1", 3, 3, 1,
             "sval1", "sval1", 3, 3, 1,
             "0.3 0.4 0.6", "val1 val2 val1", "sval1 sval2 sval1",
             1, 3],
            [1.5, 1.5, 1.25, 1.75, 1, 2, 1, 0.707, 0.5, 3, 1, 1, 2, 2, 2, 1,
             "val1", "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "1.0 2.0", "val2 val1", "sval2 sval1",
             2, 1],
            [-0.5, -0.5, -2.25, 1.25, -4, 3, -4, 4.95, 24.5, -1, 7, 3, -4, 2, 2, 1,
             "val1", "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "3.0 -4.0", "val2 val1", "sval2 sval1",
             2, 2],
            [5, 5, 5, 5, 5, 5, 5, 0, 0, 10, 0, 5, 5, 2, 2, 1,
             "val1", "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "5.0 5.0", "val2 val1", "sval2 sval1",
             2, 3]
            ], columns=expected_columns
        )
        # fmt: on

        output_df = table_to_frame(
            self.get_output(self.widget.Outputs.data), include_metas=True
        )
        # remove random since it is not possible to test
        output_df = output_df.loc[:, ~output_df.columns.str.endswith("Random value")]

        pd.testing.assert_frame_equal(
            output_df,
            expected_df,
            check_dtype=False,
            check_column_type=False,
            check_categorical=False,
            atol=1e-3,
        )

    def test_metas_results(self):
        """Test if variable that is in meta in input table remains in metas"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self._set_selection(self.widget.controls.gb_attrs, [0, 1])

        output = self.get_output(self.widget.Outputs.data)
        self.assertIn(self.data.domain["svar"], output.domain.metas)

    def test_context(self):
        d = self.data.domain
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["Mean", "Mean", "Mean", "Mode", "Concatenate"]
        )

        self.select_table_rows(self.widget.agg_table_view, [0, 2])
        self.widget.agg_checkboxes["Median"].click()
        self.assert_aggregations_equal(
            ["Mean, Median", "Mean", "Mean, Median", "Mode", "Concatenate"]
        )

        self._set_selection(self.widget.controls.gb_attrs, [1, 2])
        self.assertListEqual([d["a"], d["b"]], self.widget.gb_attrs)
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean", "Median"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        # send new data and previous data to check if context restored correctly
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["Mean, Median", "Mean", "Mean, Median", "Mode", "Concatenate"]
        )
        self._set_selection(self.widget.controls.gb_attrs, [1, 2])
        self.assertListEqual([d["a"], d["b"]], self.widget.gb_attrs)
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean", "Median"},
                d["dvar"]: {"Mode"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

    def test_context_time_variable(self):
        """
        Test migrate_context which removes sum for TimeVariable since
        GroupBy does not support it anymore for TimeVariable
        """
        tv = TimeVariable("T", have_time=True, have_date=True)
        data = Table.from_numpy(
            Domain([DiscreteVariable("G", values=["G1", "G2"]), tv]),
            np.array([[0.0, 0.0], [0, 10], [0, 20], [1, 500], [1, 1000]]),
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.aggregations[tv].add("Sum")
        self.widget.aggregations[tv].add("Median")
        self.send_signal(self.widget.Inputs.data, self.iris)

        widget = self.create_widget(
            OWGroupBy,
            stored_settings=self.widget.settingsHandler.pack_data(self.widget),
        )
        self.send_signal(widget.Inputs.data, data, widget=widget)
        self.assertSetEqual(widget.aggregations[tv], {"Mean", "Median"})

    @patch(
        "Orange.data.aggregate.OrangeTableGroupBy.aggregate",
        Mock(side_effect=ValueError("Test unexpected err")),
    )
    def test_unexpected_error(self):
        """Test if exception in aggregation shown correctly"""

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        self.assertTrue(self.widget.Error.unexpected_error.is_shown())
        self.assertEqual(
            str(self.widget.Error.unexpected_error),
            "Test unexpected err",
        )

    def test_time_variable(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        test10_path = os.path.join(
            cur_dir, "..", "..", "..", "tests", "datasets", "test10.tab"
        )
        data = Table.from_file(test10_path)

        # time variable as a group by variable
        self.send_signal(self.widget.Inputs.data, data)
        self._set_selection(self.widget.controls.gb_attrs, [3])
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(output))

        # time variable as a grouped variable
        attributes = [data.domain["c2"], data.domain["d2"]]
        self.send_signal(self.widget.Inputs.data, data[:, attributes])
        self._set_selection(self.widget.controls.gb_attrs, [1])  # d2
        # check all aggregations
        self.assert_aggregations_equal(["Mean", "Mode"])
        self.select_table_rows(self.widget.agg_table_view, [0])  # c2
        for cb in self.widget.agg_checkboxes.values():
            if cb.text() != "Mean":
                cb.click()
        self.assert_aggregations_equal(["Mean, Median, Q1 and 14 more", "Mode"])
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(2, len(output))

    def test_time_variable_results(self):
        data = Table.from_numpy(
            Domain(
                [
                    DiscreteVariable("G", values=["G1", "G2", "G3"]),
                    TimeVariable("T", have_time=True, have_date=True),
                ]
            ),
            np.array([[0.0, 0], [0, 10], [0, 20], [1, 500], [1, 1000], [2, 1]]),
        )
        self.send_signal(self.widget.Inputs.data, data)

        # disable aggregating G
        self.select_table_rows(self.widget.agg_table_view, [0])  # T
        self.widget.agg_checkboxes["Mode"].click()
        # select all possible aggregations for T
        self.select_table_rows(self.widget.agg_table_view, [1])  # T
        for cb in self.widget.agg_checkboxes.values():
            if cb.text() != "Mean":
                cb.click()
        self.assert_aggregations_equal(["", "Mean, Median, Q1 and 14 more"])

        expected_df = pd.DataFrame(
            {
                "T - Mean": [
                    "1970-01-01 00:00:10",
                    "1970-01-01 00:12:30",
                    "1970-01-01 00:00:01",
                ],
                "T - Median": [
                    "1970-01-01 00:00:10",
                    "1970-01-01 00:12:30",
                    "1970-01-01 00:00:01",
                ],
                "T - Q1": [
                    "1970-01-01 00:00:05",
                    "1970-01-01 00:10:25",
                    "1970-01-01 00:00:01",
                ],
                "T - Q3": [
                    "1970-01-01 00:00:15",
                    "1970-01-01 00:14:35",
                    "1970-01-01 00:00:01",
                ],
                "T - Min. value": [
                    "1970-01-01 00:00:00",
                    "1970-01-01 00:08:20",
                    "1970-01-01 00:00:01",
                ],
                "T - Max. value": [
                    "1970-01-01 00:00:20",
                    "1970-01-01 00:16:40",
                    "1970-01-01 00:00:01",
                ],
                "T - Mode": [
                    "1970-01-01 00:00:00",
                    "1970-01-01 00:08:20",
                    "1970-01-01 00:00:01",
                ],
                "T - Standard deviation": [10, 353.5533905932738, np.nan],
                "T - Variance": [100, 125000, np.nan],
                "T - Span": [20, 500, 0],
                "T - First value": [
                    "1970-01-01 00:00:00",
                    "1970-01-01 00:08:20",
                    "1970-01-01 00:00:01",
                ],
                "T - Last value": [
                    "1970-01-01 00:00:20",
                    "1970-01-01 00:16:40",
                    "1970-01-01 00:00:01",
                ],
                "T - Count defined": [3, 2, 1],
                "T - Count": [3, 2, 1],
                "T - Proportion defined": [1, 1, 1],
                "T - Concatenate": [
                    "1970-01-01 00:00:00 1970-01-01 00:00:10 1970-01-01 00:00:20",
                    "1970-01-01 00:08:20 1970-01-01 00:16:40",
                    "1970-01-01 00:00:01",
                ],
                "G": ["G1", "G2", "G3"],
            }
        )
        df_col = [
            "T - Mean",
            "T - Median",
            "T - Q1",
            "T - Q3",
            "T - Mode",
            "T - Min. value",
            "T - Max. value",
            "T - First value",
            "T - Last value",
        ]
        expected_df[df_col] = expected_df[df_col].apply(pd.to_datetime)
        output = self.get_output(self.widget.Outputs.data)
        output_df = table_to_frame(output, include_metas=True)
        # remove random since it is not possible to test
        output_df = output_df.loc[:, ~output_df.columns.str.endswith("Random value")]

        pd.testing.assert_frame_equal(
            output_df,
            expected_df,
            check_dtype=False,
            check_column_type=False,
            check_categorical=False,
            atol=1e-3,
        )
        expected_attributes = (
            TimeVariable("T - Mean", have_date=1, have_time=1),
            TimeVariable("T - Median", have_date=1, have_time=1),
            TimeVariable("T - Q1", have_date=1, have_time=1),
            TimeVariable("T - Q3", have_date=1, have_time=1),
            TimeVariable("T - Min. value", have_date=1, have_time=1),
            TimeVariable("T - Max. value", have_date=1, have_time=1),
            TimeVariable("T - Mode", have_date=1, have_time=1),
            ContinuousVariable(name="T - Standard deviation"),
            ContinuousVariable(name="T - Variance"),
            ContinuousVariable(name="T - Span"),
            TimeVariable("T - First value", have_date=1, have_time=1),
            TimeVariable("T - Last value", have_date=1, have_time=1),
            TimeVariable("T - Random value", have_date=1, have_time=1),
            ContinuousVariable(name="T - Count defined"),
            ContinuousVariable(name="T - Count"),
            ContinuousVariable(name="T - Proportion defined"),
        )
        expected_metas = (
            StringVariable(name="T - Concatenate"),
            DiscreteVariable(name="G", values=("G1", "G2", "G3")),
        )
        self.assertTupleEqual(output.domain.attributes, expected_attributes)
        self.assertTupleEqual(output.domain.metas, expected_metas)

    def test_tz_time_variable_results(self):
        """ Test results in case of timezoned time variable"""
        tv = TimeVariable("T", have_time=True, have_date=True)
        data = Table.from_numpy(
            Domain([DiscreteVariable("G", values=["G1", "G2"]), tv]),
            np.array([[0.0, tv.parse("1970-01-01 01:00:00+01:00")],
                      [0, tv.parse("1970-01-01 01:00:10+01:00")],
                     [0, tv.parse("1970-01-01 01:00:20+01:00")]]),
        )

        self.send_signal(self.widget.Inputs.data, data)

        # disable aggregating G
        self.select_table_rows(self.widget.agg_table_view, [0])  # T
        self.widget.agg_checkboxes["Mode"].click()
        # select all possible aggregations for T
        self.select_table_rows(self.widget.agg_table_view, [1])  # T
        for cb in self.widget.agg_checkboxes.values():
            if cb.text() != "Mean":
                cb.click()
        self.assert_aggregations_equal(["", "Mean, Median, Q1 and 14 more"])

        expected_df = pd.DataFrame(
            {
                "T - Mean": ["1970-01-01 00:00:10"],
                "T - Median": ["1970-01-01 00:00:10"],
                "T - Q1": ["1970-01-01 00:00:05"],
                "T - Q3": ["1970-01-01 00:00:15"],
                "T - Min. value": ["1970-01-01 00:00:00"],
                "T - Max. value": ["1970-01-01 00:00:20"],
                "T - Mode": ["1970-01-01 00:00:00"],
                "T - Standard deviation": [10],
                "T - Variance": [100],
                "T - Span": [20, ],
                "T - First value": ["1970-01-01 00:00:00"],
                "T - Last value": ["1970-01-01 00:00:20"],
                "T - Count defined": [3],
                "T - Count": [3],
                "T - Proportion defined": [1],
                "T - Concatenate": [
                    "1970-01-01 00:00:00 1970-01-01 00:00:10 1970-01-01 00:00:20",
                ],
                "G": ["G1"],
            }
        )
        df_col = [
            "T - Mean",
            "T - Median",
            "T - Q1",
            "T - Q3",
            "T - Min. value",
            "T - Max. value",
            "T - Mode",
            "T - First value",
            "T - Last value",
        ]
        expected_df[df_col] = expected_df[df_col].apply(pd.to_datetime)
        output_df = table_to_frame(
            self.get_output(self.widget.Outputs.data), include_metas=True
        )
        # remove random since it is not possible to test
        output_df = output_df.loc[:, ~output_df.columns.str.endswith("Random value")]

        pd.testing.assert_frame_equal(
            output_df,
            expected_df,
            check_dtype=False,
            check_column_type=False,
            check_categorical=False,
            atol=1e-3,
        )

    def test_only_nan_in_group(self):
        data = Table(
            Domain([ContinuousVariable("A"), ContinuousVariable("B")]),
            np.array([[1, np.nan], [2, 1], [1, np.nan], [2, 1]]),
        )
        self.send_signal(self.widget.Inputs.data, data)

        # select feature A as group-by
        self._set_selection(self.widget.controls.gb_attrs, [0])
        # select all aggregations for feature B
        self.select_table_rows(self.widget.agg_table_view, [1])
        for cb in self.widget.agg_checkboxes.values():
            while not cb.isChecked():
                cb.click()

        # unselect all aggregations for attr A
        self.select_table_rows(self.widget.agg_table_view, [0])
        for cb in self.widget.agg_checkboxes.values():
            while cb.isChecked():
                cb.click()

        expected_columns = [
            "B - Mean",
            "B - Median",
            "B - Q1",
            "B - Q3",
            "B - Min. value",
            "B - Max. value",
            "B - Mode",
            "B - Standard deviation",
            "B - Variance",
            "B - Sum",
            "B - Span",
            "B - First value",
            "B - Last value",
            "B - Random value",
            "B - Count defined",
            "B - Count",
            "B - Proportion defined",
            "B - Concatenate",
            "A",
        ]
        n = np.nan
        expected_df = pd.DataFrame(
            [
                [n, n, n, n, n, n, n, n, n, 0, n, n, n, n, 0, 2, 0, "", 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 1, 1, 1, 2, 2, 1, "1.0 1.0", 2],
            ],
            columns=expected_columns,
        )
        output_df = table_to_frame(
            self.get_output(self.widget.Outputs.data), include_metas=True
        )
        pd.testing.assert_frame_equal(
            output_df,
            expected_df,
            check_dtype=False,
            check_column_type=False,
            check_categorical=False,
        )

    def test_hidden_attributes(self):
        domain = self.iris.domain
        data = self.iris.transform(domain.copy())

        data.domain.attributes[0].attributes["hidden"] = True
        self.send_signal(self.widget.Inputs.data, data)
        self.assertListEqual([data.domain["iris"]], self.widget.gb_attrs)

        data = self.iris.transform(domain.copy())
        data.domain.class_vars[0].attributes["hidden"] = True
        self.send_signal(self.widget.Inputs.data, data)
        # iris is hidden now so sepal length is selected
        self.assertListEqual([data.domain["sepal length"]], self.widget.gb_attrs)

        d = domain.copy()
        data = self.iris.transform(Domain(d.attributes[:3], metas=d.attributes[3:]))
        data.domain.metas[0].attributes["hidden"] = True
        self.send_signal(self.widget.Inputs.data, data)
        # sepal length still selected because of context
        self.assertListEqual([data.domain["sepal length"]], self.widget.gb_attrs)

        # test case when one of two selected attributes is hidden
        self._set_selection(self.widget.controls.gb_attrs, [0, 1])  # sep l, sep w
        data.domain.attributes[0].attributes["hidden"] = True
        self.send_signal(self.widget.Inputs.data, data)
        # sepal length is hidden - only sepal width remain selected
        self.assertListEqual([data.domain["sepal width"]], self.widget.gb_attrs)


if __name__ == "__main__":
    unittest.main()
