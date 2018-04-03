# pylint: disable=protected-access
import unittest
from unittest.mock import patch, Mock

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QRadioButton

import Orange.clustering
from Orange.data import Table, Domain
from Orange.widgets import gui
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owkmeans import OWKMeans, ClusterTableModel


class TestClusterTableModel(unittest.TestCase):
    def test_model(self):
        model = ClusterTableModel()
        model.set_scores(["bad", 0.250, "another bad"], 3)

        self.assertEqual(model.start_k, 3)
        self.assertEqual(model.rowCount(), 3)
        ind0, ind1 = model.index(0, 0), model.index(1, 0)
        self.assertEqual(model.flags(ind0), Qt.NoItemFlags)
        self.assertEqual(model.flags(ind1),
                         Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        data = model.data
        self.assertEqual(data(ind0), "NA")
        self.assertEqual(data(ind1), "0.250")
        self.assertEqual(data(ind0, Qt.ToolTipRole), "bad")
        self.assertIsNone(data(ind1, Qt.ToolTipRole))
        self.assertIsNone(data(ind0, gui.BarRatioRole))
        self.assertAlmostEqual(data(ind1, gui.BarRatioRole), 0.250)

        self.assertAlmostEqual(data(ind1, Qt.TextAlignmentRole),
                               Qt.AlignVCenter | Qt.AlignLeft)

        self.assertEqual(model.headerData(1, Qt.Vertical), "4")


class TestOWKMeans(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWKMeans, stored_settings={"auto_commit": False, "version": 2}
        )  # type: OWKMeans
        self.iris = Table("iris")

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_migrate_version_1_settings(self):
        widget = self.create_widget(
            OWKMeans,
            stored_settings={'auto_apply': False}
        )
        self.assertFalse(widget.auto_commit)

    def test_optimization_report_display(self):
        """Check visibility of the table after selecting number of clusters"""
        self.widget.auto_commit = True
        self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)
        self.widget.optimize_k = True
        radio_buttons = self.widget.controls.optimize_k.findChildren(QRadioButton)

        radio_buttons[0].click()
        self.assertFalse(self.widget.optimize_k)
        self.assertTrue(self.widget.mainArea.isHidden())

        radio_buttons[1].click()
        self.assertTrue(self.widget.optimize_k)
        self.assertFalse(self.widget.mainArea.isHidden())
        self.widget.apply_button.button.click()

        self.wait_until_stop_blocking()
        self.assertEqual(self.widget.table_view.model().rowCount() > 0, True)

    def test_changing_k_changes_radio(self):
        widget = self.widget
        widget.auto_commit = True
        self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)

        widget.optimize_k = True

        widget.controls.k.setValue(5)
        self.assertFalse(widget.optimize_k)
        self.assertTrue(self.widget.mainArea.isHidden())

        widget.controls.k_from.setValue(5)
        self.assertTrue(widget.optimize_k)
        self.assertFalse(self.widget.mainArea.isHidden())

        widget.controls.k.setValue(3)
        self.assertFalse(widget.optimize_k)
        self.assertTrue(self.widget.mainArea.isHidden())

        widget.controls.k_to.setValue(9)
        self.assertTrue(widget.optimize_k)
        self.assertFalse(self.widget.mainArea.isHidden())

    def test_k_limits(self):
        widget = self.widget
        widget.controls.k_from.setValue(10)
        self.assertEqual(widget.k_from, 10)
        self.assertEqual(widget.k_to, 11)

        widget.controls.k_to.setValue(4)
        self.assertEqual(widget.k_from, 3)
        self.assertEqual(widget.k_to, 4)

    def test_no_data_hides_main_area(self):
        widget = self.widget
        widget.auto_commit = True
        widget.optimize_k = True
        widget.k_from, widget.k_to = 3, 4

        self.send_signal(self.widget.Inputs.data, None, wait=5000)
        self.assertTrue(self.widget.mainArea.isHidden())
        self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)
        self.assertFalse(self.widget.mainArea.isHidden())
        self.send_signal(self.widget.Inputs.data, None, wait=5000)
        self.assertTrue(self.widget.mainArea.isHidden())

    def test_data_limits(self):
        """Data error should be shown when `k` > n or when `k_from` > n. Data
        warning should be shown when `k_to` > n"""
        widget = self.widget
        widget.auto_commit = False

        self.send_signal(self.widget.Inputs.data, self.iris[:5])

        widget.k = 10
        self.commit_and_wait()
        self.assertTrue(widget.Error.not_enough_data.is_shown())

        widget.k = 3
        self.commit_and_wait()
        self.assertFalse(widget.Error.not_enough_data.is_shown())

        widget.k_from = 10
        widget.optimize_k = True
        self.commit_and_wait()
        self.assertTrue(widget.Error.not_enough_data.is_shown())

        widget.k_from = 3
        widget.k_to = 4
        self.commit_and_wait()
        self.assertFalse(widget.Error.not_enough_data.is_shown())

        widget.k_from = 3
        widget.k_to = 6
        self.commit_and_wait()
        self.assertFalse(widget.Error.not_enough_data.is_shown())
        self.assertTrue(widget.Warning.not_enough_data.is_shown())

    def test_use_cache(self):
        """Cache various clusterings for the dataset until data changes."""
        widget = self.widget
        widget.auto_commit = False
        self.send_signal(self.widget.Inputs.data, self.iris)

        with patch.object(widget, "_compute_clustering",
                          wraps=widget._compute_clustering) as compute:

            widget.k = 3
            widget.optimize_k = False

            self.commit_and_wait()
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(compute.call_args[1]['k'], 3)

            widget.k_from = 2
            widget.k_to = 3
            widget.optimize_k = True

            compute.reset_mock()
            self.commit_and_wait()
            # Since 3 was already computed before when we weren't optimizing,
            # we only need to compute for 3
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(compute.call_args[1]['k'], 2)

            # Commiting again should not recompute the clusterings
            compute.reset_mock()
            self.commit_and_wait()
            # compute.assert_not_called unfortunately didn't exist before 3.5
            self.assertFalse(compute.called)

    def test_data_on_output(self):
        """Check if data is on output after create widget and run"""
        self.widget.auto_commit = True
        self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)
        self.widget.apply_button.button.click()
        self.assertNotEqual(self.widget.data, None)
        # Disconnect the data
        self.send_signal(self.widget.Inputs.data, None)
        # removing data should have cleared the output
        self.assertEqual(self.widget.data, None)

    class KMeansFail(Orange.clustering.KMeans):
        fail_on = set()

        def fit(self, *args):
            # when not optimizing, params is empty?!
            k = self.params.get("n_clusters", 3)
            if k in self.fail_on:
                raise ValueError("k={} fails".format(k))
            return super().fit(*args)

    @patch("Orange.widgets.unsupervised.owkmeans.KMeans", new=KMeansFail)
    def test_optimization_fails(self):
        widget = self.widget
        widget.auto_commit = True
        widget.k_from, widget.k_to = 3, 8
        widget.optimize_k = True

        self.KMeansFail.fail_on = {3, 5, 7}
        model = widget.table_view.model()

        with patch.object(model, "set_scores", wraps=model.set_scores) as set_scores:
            self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)
            scores, start_k = set_scores.call_args[0]
            self.assertEqual(
                scores,
                [km if isinstance(km, str) else km.silhouette
                 for km in (widget.clusterings[k] for k in range(3, 9))]
            )
            self.assertEqual(start_k, 3)

        self.assertIsInstance(widget.clusterings[3], str)
        self.assertIsInstance(widget.clusterings[5], str)
        self.assertIsInstance(widget.clusterings[7], str)
        self.assertNotIsInstance(widget.clusterings[4], str)
        self.assertNotIsInstance(widget.clusterings[6], str)
        self.assertNotIsInstance(widget.clusterings[8], str)
        self.assertFalse(widget.Error.failed.is_shown())
        self.assertEqual(widget.selected_row(), 1)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

        self.KMeansFail.fail_on = set(range(3, 9))
        widget.invalidate()
        self.wait_until_stop_blocking()
        self.assertTrue(widget.Error.failed.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

        self.KMeansFail.fail_on = set()
        widget.invalidate()
        self.wait_until_stop_blocking()
        self.assertFalse(widget.Error.failed.is_shown())
        self.assertEqual(widget.selected_row(), 0)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    @patch("Orange.widgets.unsupervised.owkmeans.KMeans", new=KMeansFail)
    def test_run_fails(self):
        self.widget.k = 3
        self.widget.auto_commit = True
        self.widget.optimize_k = False
        self.KMeansFail.fail_on = {3}
        self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)
        self.assertTrue(self.widget.Error.failed.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

        self.KMeansFail.fail_on = set()
        self.widget.invalidate()
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Error.failed.is_shown())
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_select_best_row(self):
        class Cluster:
            def __init__(self, n):
                self.silhouette = n

        widget = self.widget
        widget.k_from, widget.k_to = 2, 6
        widget.clusterings = {k: Cluster(5 - (k - 4) ** 2) for k in range(2, 7)}
        widget.update_results()
        self.assertEqual(widget.selected_row(), 2)

        widget.clusterings = {k: "error" for k in range(2, 7)}
        widget.update_results()
        self.assertEqual(widget.selected_row(), None)

    def test_report(self):
        widget = self.widget
        widget.k = 4
        widget.optimize_k = False
        with patch.object(widget, "report_items") as report_items, \
                patch.object(widget, "report_data") as report_data, \
                patch.object(widget, "report_table") as report_table, \
                patch.object(widget, "selected_row", new=Mock(return_value=42)):
            widget.send_report()
            items = report_items.call_args[0][0]
            self.assertEqual(items[0], ("Number of clusters", 4))
            self.assertEqual(items[1][0], "Optimization")
            self.assertFalse(report_data.called)
            self.assertFalse(report_table.called)

            widget.data = data = Mock()
            widget.send_report()
            self.assertIs(report_data.call_args[0][1], data)
            self.assertFalse(report_table.called)

            report_data.reset_mock()
            report_items.reset_mock()
            widget.k_from, widget.k_to = 2, 3
            widget.optimize_k = True
            widget.send_report()
            items = report_items.call_args[0][0]
            self.assertEqual(items[0], ("Number of clusters", 44))
            self.assertIs(report_data.call_args[0][1], data)
            self.assertIs(report_table.call_args[0][1],
                          widget.table_view)

    def test_not_enough_rows(self):
        """
        Widget should not crash when there is less rows than k_from.
        GH-2172
        """
        table = self.iris[0:1, :]
        self.widget.controls.k_from.setValue(2)
        self.widget.controls.k_to.setValue(9)
        self.send_signal(self.widget.Inputs.data, table)

    def test_from_to_table(self):
        """
        From and To spins and number of rows in a scores table changes.
        GH-2172
        """
        k_from, k_to = 2, 9
        self.widget.controls.k_from.setValue(k_from)
        self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)
        check = lambda x: 2 if x - k_from + 1 < 2 else x - k_from + 1
        for i in range(k_from, k_to):
            self.widget.controls.k_to.setValue(i)
            self.commit_and_wait()
            self.assertEqual(len(self.widget.table_view.model().scores), check(i))
        for i in range(k_to, k_from, -1):
            self.widget.controls.k_to.setValue(i)
            self.commit_and_wait()
            self.assertEqual(len(self.widget.table_view.model().scores), check(i))

    def test_silhouette_column(self):
        widget = self.widget
        widget.auto_commit = True
        widget.k = 4
        widget.optimize_k = False

        random = np.random.RandomState(0)  # Avoid randomness in the test
        table = Table(random.rand(5010, 2))
        self.send_signal(self.widget.Inputs.data, table)
        outtable = self.get_output(widget.Outputs.annotated_data)
        outtable = outtable.get_column_view("Silhouette")[0]
        self.assertTrue(np.all(np.isnan(outtable)))
        self.assertTrue(widget.Warning.no_silhouettes.is_shown())

        self.send_signal(self.widget.Inputs.data, table[:100])
        outtable = self.get_output(widget.Outputs.annotated_data)
        outtable = outtable.get_column_view("Silhouette")[0]
        np.testing.assert_array_less(outtable, 1.01)
        np.testing.assert_array_less(-0.01, outtable)
        self.assertFalse(widget.Warning.no_silhouettes.is_shown())

    def test_invalidate_clusterings_cancels_jobs(self):
        widget = self.widget
        widget.auto_commit = False

        # Send the data without waiting
        self.send_signal(widget.Inputs.data, self.iris)
        widget.unconditional_commit()
        # Now, invalidate by changing max_iter
        widget.max_iterations = widget.max_iterations + 1
        widget.invalidate()
        self.wait_until_stop_blocking()

        self.assertEqual(widget.clusterings, {})

    def test_do_not_recluster_on_same_data(self):
        """Do not recluster data points when targets or metas change."""

        # Prepare some dummy data
        x = np.eye(5)
        y1, y2 = np.ones((5, 1)), np.ones((5, 2))
        meta1, meta2 = np.ones((5, 1)), np.ones((5, 2))

        table1 = Table.from_numpy(
            domain=Domain.from_numpy(X=x, Y=y1, metas=meta1),
            X=x, Y=y1, metas=meta1,
        )
        # X is same, should not cause update
        table2 = Table.from_numpy(
            domain=Domain.from_numpy(X=x, Y=y2, metas=meta2),
            X=x, Y=y2, metas=meta2,
        )
        # X is different, should cause update
        table3 = table1.copy()
        table3.X[:, 0] = 1

        with patch.object(self.widget, 'commit') as commit:
            self.send_signal(self.widget.Inputs.data, table1)
            self.commit_and_wait()
            call_count = commit.call_count

            # Sending data with same X should not recompute the clustering
            self.send_signal(self.widget.Inputs.data, table2)
            self.commit_and_wait()
            self.assertEqual(call_count, commit.call_count)

            # Sending data with different X should recompute the clustering
            self.send_signal(self.widget.Inputs.data, table3)
            self.commit_and_wait()
            self.assertEqual(call_count + 1, commit.call_count)


if __name__ == "__main__":
    unittest.main()
