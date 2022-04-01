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
        self.assertEqual(len(output), 35)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_data_domain_changed(self):
        self.send_signal(self.widget.Inputs.data, self.iris[:, -2:])
        self.assert_aggregations_equal(["Mean", "Concatenate"])

        self.send_signal(self.widget.Inputs.data, self.iris[:, -3:])
        self.assert_aggregations_equal(["Mean", "Mean", "Concatenate"])
        self.select_table_rows(self.widget.agg_table_view, [0])

    @staticmethod
    def _set_selection(view: QListView, indices: List[int]):
        view.clearSelection()
        sm = view.selectionModel()
        model = view.model()
        for ind in indices:
            sm.select(model.index(ind), QItemSelectionModel.Select)

    def test_groupby_attr_selection(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 35)

        # select iris attribute with index 0
        self._set_selection(self.widget.gb_attrs_view, [0])
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 3)

        # select iris attribute with index 0
        self._set_selection(self.widget.gb_attrs_view, [0, 1])
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 57)

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
        self.send_signal(self.widget.Inputs.data, self.data)

        model = self.widget.agg_table_model
        table = self.widget.agg_table_view

        self.assertListEqual(
            ["a", "b", "cvar", "dvar", "svar"],
            [model.data(model.index(i, 0)) for i in range(model.rowCount())],
        )

        self.select_table_rows(table, [0])
        self.assert_enabled_cbs(
            {
                "Mean",
                "Median",
                "Mode",
                "Standard deviation",
                "Variance",
                "Sum",
                "Min. value",
                "Max. value",
                "Count defined",
                "Count",
                "Concatenate",
                "Span",
                "First value",
                "Last value",
                "Random value",
                "Proportion defined",
            }
        )
        self.select_table_rows(table, [0, 1])
        self.assert_enabled_cbs(
            {
                "Mean",
                "Median",
                "Mode",
                "Standard deviation",
                "Variance",
                "Sum",
                "Min. value",
                "Max. value",
                "Count defined",
                "Count",
                "Concatenate",
                "Span",
                "First value",
                "Last value",
                "Random value",
                "Proportion defined",
            }
        )
        self.select_table_rows(table, [2])
        self.assert_enabled_cbs(
            {
                "Mean",
                "Median",
                "Mode",
                "Standard deviation",
                "Variance",
                "Sum",
                "Min. value",
                "Max. value",
                "Count defined",
                "Count",
                "Concatenate",
                "Span",
                "First value",
                "Last value",
                "Random value",
                "Proportion defined",
            }
        )
        self.select_table_rows(table, [3])  # discrete variable
        self.assert_enabled_cbs(
            {
                "Count defined",
                "Count",
                "Concatenate",
                "First value",
                "Last value",
                "Random value",
                "Proportion defined",
            }
        )
        self.select_table_rows(table, [4])  # string variable
        self.assert_enabled_cbs(
            {
                "Count defined",
                "Count",
                "Concatenate",
                "First value",
                "Last value",
                "Random value",
                "Proportion defined",
            }
        )
        self.select_table_rows(table, [3, 4])  # string variable
        self.assert_enabled_cbs(
            {
                "Count defined",
                "Count",
                "Concatenate",
                "First value",
                "Last value",
                "Random value",
                "Proportion defined",
            }
        )
        self.select_table_rows(table, [2, 3, 4])  # string variable
        self.assert_enabled_cbs(
            {
                "Mean",
                "Median",
                "Mode",
                "Standard deviation",
                "Variance",
                "Sum",
                "Min. value",
                "Max. value",
                "Count defined",
                "Count",
                "Concatenate",
                "Span",
                "First value",
                "Last value",
                "Random value",
                "Proportion defined",
            }
        )

    def assert_aggregations_equal(self, expected_text):
        model = self.widget.agg_table_model
        agg_text = [model.data(model.index(i, 1)) for i in range(model.rowCount())]
        self.assertListEqual(expected_text, agg_text)

    def test_aggregations_change(self):
        table = self.widget.agg_table_view
        d = self.data.domain

        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["Mean", "Mean", "Mean", "Concatenate", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0])
        self.widget.agg_checkboxes["Median"].click()
        self.assert_aggregations_equal(
            ["Mean, Median", "Mean", "Mean", "Concatenate", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1])
        self.widget.agg_checkboxes["Mode"].click()
        self.assert_aggregations_equal(
            ["Mean, Median, Mode", "Mean, Mode", "Mean", "Concatenate", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Concatenate"},
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
                "Concatenate",
                "Concatenate",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median", "Mode"},
                d["b"]: {"Mean", "Median", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["Median"].click()
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["Median"].checkState()
        )
        self.assert_aggregations_equal(
            ["Mean, Mode", "Mean, Mode", "Mean", "Concatenate", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Concatenate"},
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
            ["Mean, Median, Mode", "Mean, Mode", "Mean", "Concatenate", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.widget.agg_checkboxes["Median"].click()
        self.assertEqual(
            Qt.Unchecked, self.widget.agg_checkboxes["Median"].checkState()
        )
        self.assert_aggregations_equal(
            ["Mean, Mode", "Mean, Mode", "Mean", "Concatenate", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Concatenate"},
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
                "Concatenate, Count",
                "Concatenate",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode", "Count"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        # test the most complicated scenario: numeric with mode, numeric without
        # mode and discrete
        self.select_table_rows(table, [0])
        self.widget.agg_checkboxes["Mode"].click()
        self.assert_aggregations_equal(
            ["Mean, Count", "Mean, Mode", "Mean", "Concatenate, Count", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Count"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        self.select_table_rows(table, [0, 1, 3])
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["Mode"].checkState()
        )
        self.widget.agg_checkboxes["Mode"].click()
        # must stay partially checked since one Continuous can still have mode
        # as a aggregation and discrete cannot have it
        self.assertEqual(
            Qt.PartiallyChecked, self.widget.agg_checkboxes["Mode"].checkState()
        )
        self.assert_aggregations_equal(
            [
                "Mean, Mode, Count",
                "Mean, Mode",
                "Mean",
                "Concatenate, Count",
                "Concatenate",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode", "Count"},
                d["b"]: {"Mean", "Mode"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        # since now all that can have Mode have it as an aggregation it can be
        # unchecked on the next click
        self.widget.agg_checkboxes["Mode"].click()
        self.assertEqual(Qt.Unchecked, self.widget.agg_checkboxes["Mode"].checkState())
        self.assert_aggregations_equal(
            ["Mean, Count", "Mean", "Mean", "Concatenate, Count", "Concatenate"]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Count"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Concatenate"},
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
                "Concatenate, Count defined, Count",
                "Concatenate",
            ]
        )
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Mode", "Count", "Count defined"},
                d["b"]: {"Mean", "Mode", "Count defined"},
                d["cvar"]: {"Mean"},
                d["dvar"]: {"Count", "Count defined", "Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

    def test_aggregation(self):
        """Test aggregation results"""
        self.send_signal(self.widget.Inputs.data, self.data)
        output = self.get_output(self.widget.Outputs.data)

        np.testing.assert_array_almost_equal(
            output.X, [[1, 2.143, 0.317], [2, 2, 2]], decimal=3
        )
        np.testing.assert_array_equal(
            output.metas,
            np.array(
                [
                    [
                        "val1 val2 val2 val1 val2 val1",
                        "sval1 sval2 sval2 sval1 sval2 sval1",
                        1.0,
                    ],
                    [
                        "val2 val1 val2 val1 val2 val1",
                        "sval2 sval1 sval2 sval1 sval2 sval1",
                        2.0,
                    ],
                ],
                dtype=object,
            ),
        )

        # select all aggregations for all features except a and b
        self._set_selection(self.widget.gb_attrs_view, [1, 2])
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
            "cvar - Mode",
            "cvar - Standard deviation",
            "cvar - Variance",
            "cvar - Sum",
            "cvar - Min. value",
            "cvar - Max. value",
            "cvar - Span",
            "cvar - First value",
            "cvar - Last value",
            "cvar - Count defined",
            "cvar - Count",
            "cvar - Proportion defined",
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
            [.15, .15, .1, .07, .005, .3, .1, .2, .1, 0.1, 0.2, 2, 2, 1,
             "val1", "val2", 2, 2, 1,
             "sval1", "sval2", 2, 2, 1,
             "0.1 0.2", "val1 val2", "sval1 sval2",
             1, 1],
            [.3, .3, .3, np.nan, np.nan, .3, .3, .3, 0, .3, .3, 1, 2, 0.5,
             "val2", "val2", 1, 2, 0.5,
             "", "sval2", 2, 2, 1,
             "0.3", "val2", "sval2",
             1, 2],
            [.433, .4, .3, 0.153, 0.023, 1.3, .3, .6, .3, .3, .6, 3, 3, 1,
             "val1", "val1", 3, 3, 1,
             "sval1", "sval1", 3, 3, 1,
             "0.3 0.4 0.6", "val1 val2 val1", "sval1 sval2 sval1",
             1, 3],
            [1.5, 1.5, 1, 0.707, 0.5, 3, 1, 2, 1, 1, 2, 2, 2, 1,
             "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "1.0 2.0", "val2 val1", "sval2 sval1",
             2, 1],
            [-0.5, -0.5, -4, 4.95, 24.5, -1, -4, 3, 7, 3, -4, 2, 2, 1,
             "val2", "val1", 2, 2, 1,
             "sval2", "sval1", 2, 2, 1,
             "3.0 -4.0", "val2 val1", "sval2 sval1",
             2, 2],
            [5, 5, 5, 0, 0, 10, 5, 5, 0, 5, 5, 2, 2, 1,
             "val2", "val1", 2, 2, 1,
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
        self._set_selection(self.widget.gb_attrs_view, [0, 1])

        output = self.get_output(self.widget.Outputs.data)
        self.assertIn(self.data.domain["svar"], output.domain.metas)

    def test_context(self):
        d = self.data.domain
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["Mean", "Mean", "Mean", "Concatenate", "Concatenate"]
        )

        self.select_table_rows(self.widget.agg_table_view, [0, 2])
        self.widget.agg_checkboxes["Median"].click()
        self.assert_aggregations_equal(
            ["Mean, Median", "Mean", "Mean, Median", "Concatenate", "Concatenate"]
        )

        self._set_selection(self.widget.gb_attrs_view, [1, 2])
        self.assertListEqual([d["a"], d["b"]], self.widget.gb_attrs)
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean", "Median"},
                d["dvar"]: {"Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

        # send new data and previous data to check if context restored correctly
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assert_aggregations_equal(
            ["Mean, Median", "Mean", "Mean, Median", "Concatenate", "Concatenate"]
        )
        self._set_selection(self.widget.gb_attrs_view, [1, 2])
        self.assertListEqual([d["a"], d["b"]], self.widget.gb_attrs)
        self.assertDictEqual(
            {
                d["a"]: {"Mean", "Median"},
                d["b"]: {"Mean"},
                d["cvar"]: {"Mean", "Median"},
                d["dvar"]: {"Concatenate"},
                d["svar"]: {"Concatenate"},
            },
            self.widget.aggregations,
        )

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
        self._set_selection(self.widget.gb_attrs_view, [1])
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(output))

        # time variable as a grouped variable
        self.send_signal(self.widget.Inputs.data, data)
        self._set_selection(self.widget.gb_attrs_view, [5])
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(2, len(output))

    def test_only_nan_in_group(self):
        data = Table(
            Domain([ContinuousVariable("A"), ContinuousVariable("B")]),
            np.array([[1, np.nan], [2, 1], [1, np.nan], [2, 1]]),
        )
        self.send_signal(self.widget.Inputs.data, data)

        # select feature A as group-by
        self._set_selection(self.widget.gb_attrs_view, [0])
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
            "B - Mode",
            "B - Standard deviation",
            "B - Variance",
            "B - Sum",
            "B - Min. value",
            "B - Max. value",
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
                [n, n, n, n, n, 0, n, n, n, n, n, n, 0, 2, 0, "", 1],
                [1, 1, 1, 0, 0, 2, 1, 1, 0, 1, 1, 1, 2, 2, 1, "1.0 1.0", 2],
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


if __name__ == "__main__":
    unittest.main()
