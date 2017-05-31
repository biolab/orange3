# pylint: disable=protected-access

import unittest
from unittest.mock import patch, Mock

from AnyQt.QtWidgets import QRadioButton
from AnyQt.QtCore import Qt

from Orange.widgets import gui
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owkmeans import OWKMeans, ClusterTableModel
import Orange.clustering

from Orange.data import Table, Domain, ContinuousVariable


class TestClusterTableModel(unittest.TestCase):
    # This test would belong to a separate class, but needs a widget
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
            OWKMeans, stored_settings={"auto_apply": False})  # type: OWKMeans
        self.iris = Table("iris")

    def test_optimization_report_display(self):
        """Check visibility of the table after selecting number of clusters"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.optimize_k = True
        radio_buttons = self.widget.controls.optimize_k.findChildren(QRadioButton)
        radio_buttons[0].click()
        self.assertFalse(self.widget.optimize_k)
        self.assertTrue(self.widget.mainArea.isHidden())
        radio_buttons[1].click()
        self.assertTrue(self.widget.optimize_k)
        self.assertFalse(self.widget.mainArea.isHidden())
        self.widget.apply_button.button.click()
        self.assertEqual(self.widget.table_view.model().rowCount() > 0, True)

    def test_changing_k_changes_radio(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        widget = self.widget
        widget.auto_run = True

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
        widget.optimize_k = True
        widget.k_from, widget.k_to = 3, 4

        self.send_signal(self.widget.Inputs.data, None)
        self.assertTrue(self.widget.mainArea.isHidden())
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertFalse(self.widget.mainArea.isHidden())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertTrue(self.widget.mainArea.isHidden())

    def test_data_limits(self):
        widget = self.widget

        self.send_signal(self.widget.Inputs.data, self.iris[:5])

        widget.k = 10
        widget.unconditional_apply()
        self.assertTrue(widget.Error.not_enough_data.is_shown())

        widget.k = 3
        widget.unconditional_apply()
        self.assertFalse(widget.Error.not_enough_data.is_shown())

        widget.k_from = 10
        widget.optimize_k = True
        widget.unconditional_apply()
        self.assertTrue(widget.Error.not_enough_data.is_shown())

        widget.k_from = 3
        widget.k_to = 4
        widget.unconditional_apply()
        self.assertFalse(widget.Error.not_enough_data.is_shown())

        widget.k_from = 3
        widget.k_to = 6
        widget.unconditional_apply()
        self.assertFalse(widget.Error.not_enough_data.is_shown())
        self.assertTrue(widget.Warning.not_enough_data.is_shown())

    def test_use_cache(self):
        widget = self.widget
        widget.k = 3
        widget.optimize_k = False

        self.send_signal(self.widget.Inputs.data, self.iris[:50])
        widget.unconditional_apply()

        widget.k_from = 2
        widget.k_to = 3
        widget.optimize_k = True
        with patch.object(widget, "_compute_clustering",
                          wraps=widget._compute_clustering) as compute, \
            patch.object(widget, "progressBar",
                         wraps=widget.progressBar) as progressBar:
            widget.unconditional_apply()
            self.assertEqual(compute.call_count, 1)
            compute.assert_called_with(2)
            self.assertEqual(progressBar.call_count, 1)
            progressBar.assert_called_with(1)

            compute.reset_mock()
            progressBar.reset_mock()

            widget.unconditional_apply()
            # compute.assert_not_called unfortunately didn't exist before 3.5
            self.assertFalse(compute.called)
            self.assertFalse(progressBar.called)


    def test_data_on_output(self):
        """Check if data is on output after create widget and run"""
        # Connect iris to widget
        self.send_signal(self.widget.Inputs.data, self.iris)
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
        widget.k_from = 3
        widget.k_to = 8
        widget.scoring = 0
        widget.optimize_k = True

        self.KMeansFail.fail_on = {3, 5, 7}
        model = widget.table_view.model()
        with patch.object(model, "set_scores",
                          wraps=model.set_scores) as set_scores:
            self.send_signal(self.widget.Inputs.data, self.iris)
            scores, start_k = set_scores.call_args[0]
            self.assertEqual(scores,
                             [km if isinstance(km, str) else km.silhouette
                              for km in (widget.clusterings[k]
                                         for k in range(3, 9))])
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
        self.assertTrue(widget.Error.failed.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

        self.KMeansFail.fail_on = set()
        widget.invalidate()
        self.assertFalse(widget.Error.failed.is_shown())
        self.assertEqual(widget.selected_row(), 0)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    @patch("Orange.widgets.unsupervised.owkmeans.KMeans", new=KMeansFail)
    def test_run_fails(self):
        self.widget.k = 3
        self.widget.optimize_k = False
        self.KMeansFail.fail_on = {3}
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Error.failed.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

        self.KMeansFail.fail_on = set()
        self.widget.invalidate()
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

    def test_get_var_name(self):
        widget = self.widget

        domain = Domain([ContinuousVariable(x) for x in "abc"])
        self.send_signal(self.widget.Inputs.data, Table(domain))
        self.assertEqual(widget._get_var_name(), "Cluster")

        domain = Domain([ContinuousVariable("Cluster"),
                         ContinuousVariable("Cluster (4)")])
        self.send_signal(self.widget.Inputs.data, Table(domain))
        self.assertEqual(widget._get_var_name(), "Cluster (5)")

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
        table = Table("iris")
        self.widget.controls.k_from.setValue(2)
        self.widget.controls.k_to.setValue(9)
        self.send_signal(self.widget.Inputs.data, table[0:1, :])

    def test_from_to_table(self):
        """
        From and To spins and number of rows in a scores table changes.
        GH-2172
        """
        table = Table("iris")
        k_from, k_to = 2, 9
        self.widget.controls.k_from.setValue(k_from)
        self.send_signal(self.widget.Inputs.data, table)
        check = lambda x: 2 if x - k_from + 1 < 2 else x - k_from + 1
        for i in range(k_from, k_to):
            self.widget.controls.k_to.setValue(i)
            self.assertEqual(len(self.widget.table_view.model().scores), check(i))
        for i in range(k_to, k_from, -1):
            self.widget.controls.k_to.setValue(i)
            self.assertEqual(len(self.widget.table_view.model().scores), check(i))


if __name__ == "__main__":
    unittest.main()
