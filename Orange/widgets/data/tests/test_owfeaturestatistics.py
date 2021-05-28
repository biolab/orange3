#pylint: disable=unsubscriptable-object
import datetime
import unittest
from collections import namedtuple
from functools import partial
from itertools import chain
from typing import List

import numpy as np
from AnyQt.QtCore import QItemSelection, QItemSelectionRange, \
    QItemSelectionModel, Qt

from orangewidget.settings import Context

from Orange.data import Table, Domain, StringVariable, ContinuousVariable, \
    DiscreteVariable, TimeVariable
from Orange.widgets.tests.base import WidgetTest, datasets
from Orange.widgets.tests.utils import simulate, table_dense_sparse
from Orange.widgets.data.owfeaturestatistics import \
    OWFeatureStatistics

VarDataPair = namedtuple('VarDataPair', ['variable', 'data'])

# Continuous variable variations
continuous_full = VarDataPair(
    ContinuousVariable('continuous_full'),
    np.array([0, 1, 2, 3, 4], dtype=float),
)
continuous_missing = VarDataPair(
    ContinuousVariable('continuous_missing'),
    np.array([0, 1, 2, np.nan, 4], dtype=float),
)
continuous_all_missing = VarDataPair(
    ContinuousVariable('continuous_all_missing'),
    np.array([np.nan] * 5, dtype=float),
)
continuous_same = VarDataPair(
    ContinuousVariable('continuous_same'),
    np.array([3] * 5, dtype=float),
)
continuous = [
    continuous_full, continuous_missing, continuous_all_missing,
    continuous_same
]

# Unordered discrete variable variations
rgb_full = VarDataPair(
    DiscreteVariable('rgb_full', values=('r', 'g', 'b')),
    np.array([0, 1, 1, 1, 2], dtype=float),
)
rgb_missing = VarDataPair(
    DiscreteVariable('rgb_missing', values=('r', 'g', 'b')),
    np.array([0, 1, 1, np.nan, 2], dtype=float),
)
rgb_all_missing = VarDataPair(
    DiscreteVariable('rgb_all_missing', values=('r', 'g', 'b')),
    np.array([np.nan] * 5, dtype=float),
)
rgb_bins_missing = VarDataPair(
    DiscreteVariable('rgb_bins_missing', values=('r', 'g', 'b')),
    np.array([np.nan, 1, 1, 1, np.nan], dtype=float),
)
rgb_same = VarDataPair(
    DiscreteVariable('rgb_same', values=('r', 'g', 'b')),
    np.array([2] * 5, dtype=float),
)
rgb = [
    rgb_full, rgb_missing, rgb_all_missing, rgb_bins_missing, rgb_same
]

# Ordered discrete variable variations
ints_full = VarDataPair(
    DiscreteVariable('ints_full', values=('2', '3', '4')),
    np.array([0, 1, 1, 1, 2], dtype=float),
)
ints_missing = VarDataPair(
    DiscreteVariable('ints_missing', values=('2', '3', '4')),
    np.array([0, 1, 1, np.nan, 2], dtype=float),
)
ints_all_missing = VarDataPair(
    DiscreteVariable('ints_all_missing', values=('2', '3', '4')),
    np.array([np.nan] * 5, dtype=float),
)
ints_bins_missing = VarDataPair(
    DiscreteVariable('ints_bins_missing', values=('2', '3', '4')),
    np.array([np.nan, 1, 1, 1, np.nan], dtype=float),
)
ints_same = VarDataPair(
    DiscreteVariable('ints_same', values=('2', '3', '4')),
    np.array([0] * 5, dtype=float),
)
ints = [
    ints_full, ints_missing, ints_all_missing, ints_bins_missing, ints_same
]

discrete = list(chain(rgb, ints))


def _to_timestamps(years):
    return [datetime.datetime(year, 1, 1).timestamp() if not np.isnan(year)
            else np.nan for year in years]


# Time variable variations, windows timestamps need to be valid timestamps so
# we'll just fill it in with arbitrary years
time_full = VarDataPair(
    TimeVariable('time_full', have_date=True, have_time=True),
    np.array(_to_timestamps([2000, 2001, 2002, 2003, 2004]), dtype=float),
)
time_missing = VarDataPair(
    TimeVariable('time_missing'),
    np.array(_to_timestamps([2000, np.nan, 2001, 2003, 2004]), dtype=float),
)
time_all_missing = VarDataPair(
    TimeVariable('time_all_missing'),
    np.array(_to_timestamps([np.nan] * 5), dtype=float),
)
time_same = VarDataPair(
    TimeVariable('time_same', have_date=True, have_time=True),
    np.array(_to_timestamps([2004] * 5), dtype=float),
)
time_negative = VarDataPair(
    TimeVariable('time_negative', have_date=True, have_time=True),
    np.array([0, -1, 24 * 60 * 60], dtype=float),
)
time = [
    time_full, time_missing, time_all_missing, time_same, time_negative
]

# String variable variations
string_full = VarDataPair(
    StringVariable('string_full'),
    np.array(['a', 'b', 'c', 'd', 'e'], dtype=object),
)
string_missing = VarDataPair(
    StringVariable('string_missing'),
    np.array(['a', 'b', 'c', StringVariable.Unknown, 'e'], dtype=object),
)
string_all_missing = VarDataPair(
    StringVariable('string_all_missing'),
    np.array([StringVariable.Unknown] * 5, dtype=object),
)
string_same = VarDataPair(
    StringVariable('string_same'),
    np.array(['a'] * 5, dtype=object),
)
string = [
    string_full, string_missing, string_all_missing, string_same
]


def make_table(attributes, target=None, metas=None):
    """Build an instance of a table given various variables.

    Parameters
    ----------
    attributes : Iterable[Tuple[Variable, np.array]
    target : Optional[Iterable[Tuple[Variable, np.array]]
    metas : Optional[Iterable[Tuple[Variable, np.array]]

    Returns
    -------
    Table

    """
    attribute_vars, attribute_vals = list(zip(*attributes))
    attribute_vals = np.array(attribute_vals).T

    target_vars, target_vals = None, None
    if target is not None:
        target_vars, target_vals = list(zip(*target))
        target_vals = np.array(target_vals).T

    meta_vars, meta_vals = None, None
    if metas is not None:
        meta_vars, meta_vals = list(zip(*metas))
        meta_vals = np.array(meta_vals).T

    return Table.from_numpy(
        Domain(attribute_vars, class_vars=target_vars, metas=meta_vars),
        X=attribute_vals, Y=target_vals, metas=meta_vals,
    )


class TestVariousDataSets(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWFeatureStatistics, stored_settings={'auto_commit': False}
        )

    def force_render_table(self):
        """Some fields e.g. histograms are only initialized when they actually
        need to be rendered"""
        model = self.widget.model
        for i in range(model.rowCount()):
            for j in range(model.columnCount()):
                model.data(model.index(i, j), Qt.DisplayRole)

    def run_through_variables(self):
        simulate.combobox_run_through_all(
            self.widget.cb_color_var, callback=self.force_render_table)

    @table_dense_sparse
    def test_runs_on_iris(self, prepare_table):
        self.send_signal(self.widget.Inputs.data, prepare_table(Table('iris')))

    def test_does_not_crash_on_data_removal(self):
        self.send_signal(self.widget.Inputs.data, make_table(discrete))
        self.send_signal(self.widget.Inputs.data, None)

    def test_does_not_crash_on_empty_domain(self):
        empty_data = Table('iris').transform(Domain([]))
        self.send_signal(self.widget.Inputs.data, empty_data)

    # No missing values
    @table_dense_sparse
    def test_on_data_with_no_missing_values(self, prepare_table):
        data = make_table([continuous_full, rgb_full, ints_full, time_full])
        self.send_signal(self.widget.Inputs.data, prepare_table(data))
        self.run_through_variables()

    @table_dense_sparse
    def test_on_data_with_no_missing_values_full_domain(self, prepare_table):
        data = make_table([continuous_full, time_full], [ints_full], [rgb_full])
        self.send_signal(self.widget.Inputs.data, prepare_table(data))
        self.run_through_variables()

    # With missing values
    @table_dense_sparse
    def test_on_data_with_missing_continuous_values(self, prepare_table):
        data = make_table([continuous_full, continuous_missing, rgb_full, ints_full, time_full])
        self.send_signal(self.widget.Inputs.data, prepare_table(data))
        self.run_through_variables()

    @table_dense_sparse
    def test_on_data_with_missing_discrete_values(self, prepare_table):
        data = make_table([continuous_full, rgb_full, rgb_missing, ints_full, time_full])
        self.send_signal(self.widget.Inputs.data, prepare_table(data))
        self.run_through_variables()

    @table_dense_sparse
    def test_on_data_with_discrete_values_all_the_same(self, prepare_table):
        data = make_table([continuous_full], [ints_same, rgb_same])
        self.send_signal(self.widget.Inputs.data, prepare_table(data))
        self.run_through_variables()

    @table_dense_sparse
    def test_on_data_with_continuous_values_all_the_same(self, prepare_table):
        data = make_table([ints_full, ints_same], [continuous_same, continuous_full])
        self.send_signal(self.widget.Inputs.data, prepare_table(data))
        self.run_through_variables()

    @table_dense_sparse
    def test_on_data_with_negative_timestamps(self, prepare_table):
        data = make_table([time_negative])
        self.send_signal(self.widget.Inputs.data, prepare_table(data))
        self.run_through_variables()

    def test_switching_to_dataset_with_no_target_var(self):
        """Switching from data set with target variable to a data set with
        no target variable should not result in crash."""
        data1 = make_table([continuous_full, ints_full], [ints_same, rgb_same])
        data2 = make_table([rgb_full, ints_full])

        self.send_signal(self.widget.Inputs.data, data1)
        self.force_render_table()

        self.send_signal(self.widget.Inputs.data, data2)
        self.force_render_table()

    def test_switching_to_dataset_with_target_var(self):
        """Switching from data set with no target variable to a data set with
        a target variable should not result in crash."""
        data1 = make_table([rgb_full, ints_full])
        data2 = make_table([continuous_full, ints_full], [ints_same, rgb_same])

        self.send_signal(self.widget.Inputs.data, data1)
        self.force_render_table()

        self.send_signal(self.widget.Inputs.data, data2)
        self.force_render_table()

    def test_on_edge_case_datasets(self):
        for data in datasets.datasets():
            try:
                self.send_signal(self.widget.Inputs.data, data)
                self.force_render_table()
            except Exception as e:
                raise AssertionError(f"Failed on `{data.name}`") from e


def select_rows(rows: List[int], widget: OWFeatureStatistics):
    """Since the widget sorts the rows, selecting rows isn't trivial."""
    indices = widget.model.mapToSourceRows(rows)

    selection = QItemSelection()
    for idx in indices:
        selection.append(QItemSelectionRange(
            widget.model.index(idx, 0),
            widget.model.index(idx, widget.model.columnCount() - 1)
        ))

    widget.table_view.selectionModel().select(
        selection, QItemSelectionModel.ClearAndSelect)


class TestFeatureStatisticsOutputs(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWFeatureStatistics, stored_settings={'auto_commit': False}
        )
        self.data = make_table(
            [continuous_full, continuous_missing],
            target=[rgb_full, rgb_missing], metas=[ints_full, ints_missing]
        )
        self.send_signal(self.widget.Inputs.data, self.data)
        self.select_rows = partial(select_rows, widget=self.widget)

    def test_changing_data_updates_ouput(self):
        # Test behaviour of widget when auto commit is OFF
        self.widget.auto_commit = False

        # We start of with some data and select some rows
        self.send_signal(self.widget.Inputs.data, self.data)
        self.select_rows([0])
        # By default, nothing should be sent since auto commit is off
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.statistics))
        # When we commit, the data should be on the output
        self.widget.unconditional_commit()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.statistics))

        # Send some new data
        iris = Table('iris')
        self.send_signal(self.widget.Inputs.data, iris)
        # By default, there should be nothing on the output
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.statistics))
        # Nothing should change after commit, since we haven't selected any rows
        self.widget.unconditional_commit()
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.statistics))

        # Now let's switch back to the original data, where we selected row 0
        self.send_signal(self.widget.Inputs.data, self.data)
        # Again, since auto commit is off, nothing should be on the output
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.statistics))
        # Since the row selection is saved into context settings, the appropriate
        # thing should be sent to output
        self.widget.unconditional_commit()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.statistics))

    def test_changing_data_updates_output_with_autocommit(self):
        # Test behaviour of widget when auto commit is ON
        self.widget.auto_commit = True

        # We start of with some data and select some rows
        self.send_signal(self.widget.Inputs.data, self.data)
        self.select_rows([0])
        # Selecting rows should send data to output
        self.assertIsNotNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.statistics))

        # Send some new data
        iris = Table('iris')
        self.send_signal(self.widget.Inputs.data, iris)
        # Don't select anything, so the outputs should be empty
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.statistics))

        # Now let's switch back to the original data, where we had selected row 0,
        # we expect that to be sent to output
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.statistics))

    def test_sends_single_attribute_table_to_output(self):
        # Check if selecting a single attribute row
        self.select_rows([0])
        self.widget.unconditional_commit()

        desired_domain = Domain(attributes=[continuous_full.variable])
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertEqual(output.domain, desired_domain)

    def test_sends_multiple_attribute_table_to_output(self):
        # Check if selecting a single attribute row
        self.select_rows([0, 1])
        self.widget.unconditional_commit()

        desired_domain = Domain(attributes=[
            continuous_full.variable, continuous_missing.variable,
        ])
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertEqual(output.domain, desired_domain)

    def test_sends_single_class_var_table_to_output(self):
        self.select_rows([2])
        self.widget.unconditional_commit()

        desired_domain = Domain(attributes=[], class_vars=[rgb_full.variable])
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertEqual(output.domain, desired_domain)

    def test_sends_single_meta_table_to_output(self):
        self.select_rows([4])
        self.widget.unconditional_commit()

        desired_domain = Domain(attributes=[], metas=[ints_full.variable])
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertEqual(output.domain, desired_domain)

    def test_sends_multiple_var_types_table_to_output(self):
        self.select_rows([0, 2, 4])
        self.widget.unconditional_commit()

        desired_domain = Domain(
            attributes=[continuous_full.variable],
            class_vars=[rgb_full.variable],
            metas=[ints_full.variable],
        )
        output = self.get_output(self.widget.Outputs.reduced_data)
        self.assertEqual(output.domain, desired_domain)

    def test_sends_all_samples_to_output(self):
        """All rows should be sent to output for selected column."""
        self.select_rows([0, 2])
        self.widget.unconditional_commit()

        selected_vars = Domain(
            attributes=[continuous_full.variable],
            class_vars=[rgb_full.variable],
        )

        output = self.get_output(self.widget.Outputs.reduced_data)
        np.testing.assert_equal(output.X, self.data[:, selected_vars.variables].X)
        np.testing.assert_equal(output.Y, self.data[:, selected_vars.variables].Y)

    def test_clearing_selection_sends_none_to_output(self):
        """Clearing all the selected rows should send `None` to output."""
        self.select_rows([0])
        self.widget.unconditional_commit()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.statistics))

        self.widget.table_view.clearSelection()
        self.widget.unconditional_commit()
        self.assertIsNone(self.get_output(self.widget.Outputs.reduced_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.statistics))


class TestFeatureStatisticsUI(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWFeatureStatistics, stored_settings={'auto_commit': False}
        )
        self.data1 = Table('iris')
        self.data2 = Table('zoo')
        self.select_rows = partial(select_rows, widget=self.widget)

    def test_restores_previous_selection(self):
        """Widget should remember selection with domain context handler."""
        # Send data and select rows
        domain1 = self.data1.domain
        self.send_signal(self.widget.Inputs.data, self.data1)
        self.select_rows([0, 2])
        self.assertEqual(set(self.widget.selected_vars),
                         {domain1[0], domain1[2]})

        # Sending new data clears selection
        self.send_signal(self.widget.Inputs.data, self.data2)
        self.assertEqual(len(self.widget.selected_vars), 0)

        # Sending back the old data restores the selection
        iris3 = self.data1.transform(
            Domain([domain1[2], domain1[0], domain1[1]], domain1.class_var))
        self.send_signal(self.widget.Inputs.data, iris3)
        self.assertEqual(set(self.widget.selected_vars),
                         {domain1[0], domain1[2]})

    def test_settings_migration_to_ver21(self):
        settings = {
            'controlAreaVisible': True, 'savedWidgetGeometry': '',
            '__version__': 1,
            'context_settings': [
                Context(
                    values={'auto_commit': (True, -2),
                            'color_var': ('iris', 101),
                            'selected_rows': [1, 4],
                            'sorting': ((1, 0), -2), '__version__': 1},
                    attributes={'petal length': 2, 'petal width': 2,
                                'sepal length': 2, 'sepal width': 2},
                    metas={'iris': 1})]
        }
        widget = self.create_widget(OWFeatureStatistics,
                                    stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.data1)
        domain = self.data1.domain
        self.assertEqual(widget.selected_vars, [domain["petal width"],
                                                domain["iris"]])

    def test_report(self):
        self.send_signal(self.widget.Inputs.data, self.data1)

        self.widget.report_button.click()
        report_text = self.widget.report_html

        self.assertIn("<table>", report_text)
        self.assertEqual(6, report_text.count("<tr>"))  # header + 5 rows


class TestSummary(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWFeatureStatistics)
        self.data = make_table(
            [continuous_full, continuous_missing],
            target=[rgb_full, rgb_missing], metas=[ints_full, ints_missing]
        )
        self.select_rows = partial(select_rows, widget=self.widget)


if __name__ == "__main__":
    unittest.main()
