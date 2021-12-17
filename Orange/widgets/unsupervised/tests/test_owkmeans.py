# pylint: disable=protected-access
import unittest
from unittest.mock import patch, Mock

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QRadioButton
from sklearn.metrics import silhouette_score

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
        self.data = Table("heart_disease")

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
        self.send_signal(self.widget.Inputs.data, self.data, wait=5000)
        self.widget.optimize_k = True
        radio_buttons = self.widget.controls.optimize_k.findChildren(QRadioButton)

        radio_buttons[0].click()
        self.assertFalse(self.widget.optimize_k)
        self.assertTrue(self.widget.mainArea.isHidden())

        radio_buttons[1].click()
        self.assertTrue(self.widget.optimize_k)
        self.assertFalse(self.widget.mainArea.isHidden())
        self.widget.apply_button.button.click()

        self.wait_until_finished()
        self.assertEqual(self.widget.table_view.model().rowCount() > 0, True)

    def test_changing_k_changes_radio(self):
        widget = self.widget
        widget.auto_commit = True
        self.send_signal(self.widget.Inputs.data, self.data, wait=5000)

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
        self.send_signal(self.widget.Inputs.data, self.data, wait=5000)
        self.assertFalse(self.widget.mainArea.isHidden())
        self.send_signal(self.widget.Inputs.data, None, wait=5000)
        self.assertTrue(self.widget.mainArea.isHidden())

    def test_data_limits(self):
        """Data error should be shown when `k` > n or when `k_from` > n. Data
        warning should be shown when `k_to` > n"""
        widget = self.widget
        widget.auto_commit = False

        self.send_signal(self.widget.Inputs.data, self.data[:5])

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
        self.send_signal(self.widget.Inputs.data, self.data)

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
        self.send_signal(self.widget.Inputs.data, self.data, wait=5000)
        self.widget.apply_button.button.click()
        self.assertNotEqual(self.widget.data, None)
        # Disconnect the data
        self.send_signal(self.widget.Inputs.data, None)
        # removing data should have cleared the output
        self.assertEqual(self.widget.data, None)

    def test_centroids_on_output(self):
        widget = self.widget
        widget.optimize_k = False
        widget.k = 4
        self.send_signal(widget.Inputs.data, self.data)
        self.commit_and_wait()
        widget.clusterings[widget.k].labels = np.array([0] * 100 + [1] * 203).flatten()
        widget.clusterings[widget.k].silhouette_samples = np.arange(303) / 303
        widget.send_data()
        out = self.get_output(widget.Outputs.centroids)
        np.testing.assert_array_almost_equal(
            np.array([[0, np.mean(np.arctan(np.arange(100) / 303)) / np.pi + 0.5],
                      [1, np.mean(np.arctan(np.arange(100, 303) / 303)) / np.pi + 0.5],
                      [2, 0], [3, 0]]), out.metas.astype(float))
        self.assertEqual(out.name, "heart_disease centroids")

    def test_centroids_domain_on_output(self):
        widget = self.widget
        widget.optimize_k = False
        widget.k = 4
        heart_disease = Table("heart_disease")
        heart_disease.name = Table.name  # untitled
        self.send_signal(widget.Inputs.data, heart_disease)
        self.commit_and_wait()

        in_attrs = heart_disease.domain.attributes
        out = self.get_output(widget.Outputs.centroids)
        out_attrs = out.domain.attributes
        out_names = {attr.name for attr in out_attrs}
        for attr in in_attrs:
            self.assertEqual(
                attr.name in out_names, attr.is_continuous,
                f"at attribute '{attr.name}'"
            )
        self.assertEqual(
            len(out_attrs),
            sum(attr.is_continuous or len(attr.values) for attr in in_attrs))
        self.assertEqual(out.name, "centroids")

    class KMeansFail(Orange.clustering.KMeans):
        fail_on = set()

        def fit(self, X, Y=None):
            # when not optimizing, params is empty?!
            k = self.params.get("n_clusters", 3)
            if k in self.fail_on:
                raise ValueError("k={} fails".format(k))
            return super().fit(X, Y)

    @patch("Orange.widgets.unsupervised.owkmeans.KMeans", new=KMeansFail)
    def test_optimization_fails(self):
        widget = self.widget
        widget.auto_commit = True
        widget.k_from, widget.k_to = 3, 8
        widget.optimize_k = True

        self.KMeansFail.fail_on = {3, 5, 7}
        model = widget.table_view.model()

        with patch.object(
                model, "set_scores", wraps=model.set_scores) as set_scores:
            self.send_signal(self.widget.Inputs.data, self.data, wait=5000)
            scores, start_k = set_scores.call_args[0]
            X = self.widget.preproces(self.data).X
            self.assertEqual(
                scores,
                [km if isinstance(km, str) else silhouette_score(
                    X, km(self.data))
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
        self.wait_until_finished()
        self.assertTrue(widget.Error.failed.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

        self.KMeansFail.fail_on = set()
        widget.invalidate()
        self.wait_until_finished()
        self.assertFalse(widget.Error.failed.is_shown())
        self.assertEqual(widget.selected_row(), 0)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    @patch("Orange.widgets.unsupervised.owkmeans.KMeans", new=KMeansFail)
    def test_run_fails(self):
        self.widget.k = 3
        self.widget.auto_commit = True
        self.widget.optimize_k = False
        self.KMeansFail.fail_on = {3}
        self.send_signal(self.widget.Inputs.data, self.data, wait=5000)
        self.assertTrue(self.widget.Error.failed.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

        self.KMeansFail.fail_on = set()
        self.widget.invalidate()
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.failed.is_shown())
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_select_best_row(self):
        widget = self.widget
        widget.k_from, widget.k_to = 2, 6
        widget.optimize_k = True
        widget.normalize = False
        self.send_signal(self.widget.Inputs.data, Table("housing"), wait=5000)
        self.commit_and_wait()
        widget.update_results()
        # for housing dataset without normalization,
        # the best selection is 3 clusters, so row no. 1
        self.assertEqual(widget.selected_row(), 1)

        self.widget.controls.normalize.toggle()
        self.send_signal(self.widget.Inputs.data, Table("housing"), wait=5000)
        self.commit_and_wait()
        widget.update_results()
        # for housing dataset with normalization,
        # the best selection is 2 clusters, so row no. 0
        self.assertEqual(widget.selected_row(), 0)

        widget.clusterings = {k: "error" for k in range(2, 7)}
        widget.update_results()
        self.assertEqual(widget.selected_row(), None)

    @patch("Orange.widgets.unsupervised.owkmeans.Normalize")
    def test_normalize_sparse(self, normalize):
        normalization = normalize.return_value = Mock(return_value=self.data)
        widget = self.widget
        widget.normalize = True
        norm_check = widget.controls.normalize

        x = sp.csr_matrix(np.random.randint(0, 2, (5, 10)))
        data = Table.from_numpy(None, x)

        self.send_signal(widget.Inputs.data, data)
        self.assertTrue(widget.Warning.no_sparse_normalization.is_shown())
        self.assertFalse(norm_check.isEnabled())
        normalization.assert_not_called()

        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Warning.no_sparse_normalization.is_shown())
        self.assertTrue(norm_check.isEnabled())
        normalization.assert_not_called()

        self.send_signal(widget.Inputs.data, data)
        self.assertTrue(widget.Warning.no_sparse_normalization.is_shown())
        self.assertFalse(norm_check.isEnabled())
        normalization.assert_not_called()

        self.send_signal(widget.Inputs.data, self.data)
        self.assertFalse(widget.Warning.no_sparse_normalization.is_shown())
        self.assertTrue(norm_check.isEnabled())
        normalization.assert_called()
        normalization.reset_mock()

        widget.controls.normalize.click()

        self.send_signal(widget.Inputs.data, data)
        self.assertFalse(widget.Warning.no_sparse_normalization.is_shown())
        self.assertFalse(norm_check.isEnabled())
        normalization.assert_not_called()

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
        table = self.data[0:1, :]
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
        self.send_signal(self.widget.Inputs.data, self.data, wait=5000)
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

        # Avoid randomness in the test
        random = np.random.RandomState(0)  # pylint: disable=no-member
        table = Table.from_numpy(None, random.rand(110, 2))
        with patch(
                "Orange.widgets.unsupervised.owkmeans.SILHOUETTE_MAX_SAMPLES",
                100):
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
        self.send_signal(widget.Inputs.data, self.data)
        widget.commit.now()
        # Now, invalidate by changing max_iter
        widget.max_iterations = widget.max_iterations + 1
        widget.invalidate()
        self.wait_until_finished()

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
        with table3.unlocked():
            table3.X[:, 0] = 1

        with patch.object(self.widget.commit, 'now') as commit:
            self.send_signal(self.widget.Inputs.data, table1)
            self.commit_and_wait()
            commit.reset_mock()

            # Sending data with same X should not recompute the clustering
            self.send_signal(self.widget.Inputs.data, table2)
            commit.assert_not_called()

            # Sending data with different X should recompute the clustering
            self.send_signal(self.widget.Inputs.data, table3)
            commit.assert_called_once()

    def test_correct_smart_init(self):
        # due to a bug where wrong init was passed to _compute_clustering
        self.send_signal(self.widget.Inputs.data, self.data[::10], wait=5000)
        self.widget.smart_init = 0
        self.widget.clusterings = {}
        with patch.object(self.widget, "_compute_clustering",
                          wraps=self.widget._compute_clustering) as compute:
            self.commit_and_wait()
            self.assertEqual(compute.call_args[1]['init'], "k-means++")
        self.widget.invalidate()  # reset caches
        self.widget.smart_init = 1
        with patch.object(self.widget, "_compute_clustering",
                          wraps=self.widget._compute_clustering) as compute:
            self.commit_and_wait()
            self.assertEqual(compute.call_args[1]['init'], "random")

    def test_always_same_cluster(self):
        """The same random state should always return the same clusters"""
        self.send_signal(self.widget.Inputs.data, self.data[::10], wait=5000)

        def cluster():
            self.widget.invalidate()  # reset caches
            self.commit_and_wait()
            return self.get_output(self.widget.Outputs.annotated_data).metas[:, 0]

        def assert_all_same(l):
            for a1, a2 in zip(l, l[1:]):
                np.testing.assert_equal(a1, a2)

        self.widget.smart_init = 0
        assert_all_same([cluster() for _ in range(5)])

        self.widget.smart_init = 1
        assert_all_same([cluster() for _ in range(5)])

    def test_error_no_attributes(self):
        domain = Domain([])
        table = Table.from_domain(domain, n_rows=10)
        self.widget.auto_commit = True
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.no_attributes.is_shown())

    def test_saved_selection(self):
        self.widget.send_data = Mock()
        self.widget.optimize_k = True
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()
        self.widget.table_view.selectRow(2)
        self.assertEqual(self.widget.selected_row(), 2)
        self.assertEqual(self.widget.send_data.call_count, 3)
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWKMeans, stored_settings=settings)
        w.send_data = Mock()
        self.send_signal(w.Inputs.data, self.data, widget=w)
        self.wait_until_finished(widget=w)
        self.assertEqual(w.send_data.call_count, 2)
        self.assertEqual(self.widget.selected_row(), w.selected_row())


if __name__ == "__main__":
    unittest.main()
