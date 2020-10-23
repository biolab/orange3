# pylint: disable=protected-access
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from Orange.data import Table
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate, possible_duplicate_table
from Orange.widgets.unsupervised.owdbscan import OWDBSCAN, get_kth_distances
from Orange.widgets.utils.state_summary import format_summary_details


class TestOWDBSCAN(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDBSCAN)
        self.iris = Table("iris")

    def tearDown(self):
        self.widgets.remove(self.widget)
        self.widget.onDeleteWidget()
        self.widget = None

    def test_cluster(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)

        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNotNone(output)
        self.assertEqual(len(self.iris), len(output))
        self.assertTupleEqual(self.iris.X.shape, output.X.shape)
        self.assertTupleEqual(self.iris.Y.shape, output.Y.shape)
        self.assertEqual(2, output.metas.shape[1])

        self.assertEqual("Cluster", str(output.domain.metas[0]))
        self.assertEqual("DBSCAN Core", str(output.domain.metas[1]))

    def test_unique_domain(self):
        w = self.widget
        data = possible_duplicate_table("Cluster")
        self.send_signal(w.Inputs.data, data)
        output = self.get_output(w.Outputs.annotated_data)
        self.assertEqual(output.domain.metas[0].name, "Cluster (1)")

    def test_bad_input(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris[:1])
        self.assertTrue(w.Error.not_enough_instances.is_shown())

        self.send_signal(w.Inputs.data, self.iris[:2])
        self.assertFalse(w.Error.not_enough_instances.is_shown())

        self.send_signal(w.Inputs.data, self.iris)
        self.assertFalse(w.Error.not_enough_instances.is_shown())

    def test_data_none(self):
        w = self.widget

        self.send_signal(w.Inputs.data, None)

        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNone(output)

    def test_change_eps(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)

        # change parameters
        self.widget.controls.eps.setValue(0.5)
        output1 = self.get_output(w.Outputs.annotated_data)
        self.widget.controls.eps.setValue(1)
        output2 = self.get_output(w.Outputs.annotated_data)

        # on this data higher eps has greater sum of clusters - less nan
        # values
        self.assertGreater(np.nansum(output2.metas[:, 0]),
                           np.nansum(output1.metas[:, 0]))

        # try when no data
        self.send_signal(w.Inputs.data, None)
        self.widget.controls.eps.setValue(0.5)
        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNone(output)


    def test_change_min_samples(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)

        # change parameters
        self.widget.controls.min_samples.setValue(5)
        output1 = self.get_output(w.Outputs.annotated_data)
        self.widget.controls.min_samples.setValue(1)
        output2 = self.get_output(w.Outputs.annotated_data)

        # on this data lower min_samples has greater sum of clusters - less nan
        # values
        self.assertGreater(np.nansum(output2.metas[:, 0]),
                           np.nansum(output1.metas[:, 0]))

        # try when no data
        self.send_signal(w.Inputs.data, None)
        self.widget.controls.min_samples.setValue(3)
        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNone(output)

    def test_change_metric_idx(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)

        # change parameters
        cbox = self.widget.controls.metric_idx
        simulate.combobox_activate_index(cbox, 0)  # Euclidean
        output1 = self.get_output(w.Outputs.annotated_data)
        simulate.combobox_activate_index(cbox, 1)  # Manhattan
        output2 = self.get_output(w.Outputs.annotated_data)

        # Manhattan has more nan clusters
        self.assertGreater(np.nansum(output1.metas[:, 0]),
                           np.nansum(output2.metas[:, 0]))

        # try when no data
        self.send_signal(w.Inputs.data, None)
        cbox = self.widget.controls.metric_idx
        simulate.combobox_activate_index(cbox, 0)  # Euclidean

    def test_sparse_csr_data(self):
        self.iris.X = csr_matrix(self.iris.X)

        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)

        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNotNone(output)
        self.assertEqual(len(self.iris), len(output))
        self.assertTupleEqual(self.iris.X.shape, output.X.shape)
        self.assertTupleEqual(self.iris.Y.shape, output.Y.shape)
        self.assertEqual(2, output.metas.shape[1])

        self.assertEqual("Cluster", str(output.domain.metas[0]))
        self.assertEqual("DBSCAN Core", str(output.domain.metas[1]))

    def test_sparse_csc_data(self):
        self.iris.X = csc_matrix(self.iris.X)

        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)

        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNotNone(output)
        self.assertEqual(len(self.iris), len(output))
        self.assertTupleEqual(self.iris.X.shape, output.X.shape)
        self.assertTupleEqual(self.iris.Y.shape, output.Y.shape)
        self.assertEqual(2, output.metas.shape[1])

        self.assertEqual("Cluster", str(output.domain.metas[0]))
        self.assertEqual("DBSCAN Core", str(output.domain.metas[1]))

    def test_get_kth_distances(self):
        dists = get_kth_distances(self.iris, "euclidean", k=5)
        self.assertEqual(len(self.iris), len(dists))
        # dists must be sorted
        np.testing.assert_array_equal(dists, np.sort(dists)[::-1])

        # test with different distance - e.g. Orange distance
        dists = get_kth_distances(self.iris, Euclidean, k=5)
        self.assertEqual(len(self.iris), len(dists))
        # dists must be sorted
        np.testing.assert_array_equal(dists, np.sort(dists)[::-1])

    def test_metric_changed(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)
        cbox = w.controls.metric_idx
        simulate.combobox_activate_index(cbox, 2)

        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNotNone(output)
        self.assertEqual(len(self.iris), len(output))
        self.assertTupleEqual(self.iris.X.shape, output.X.shape)
        self.assertTupleEqual(self.iris.Y.shape, output.Y.shape)

    def test_large_data(self):
        """
        When data has less than 1000 instances they are subsampled in k-values
        computation.
        """
        w = self.widget

        data = Table(self.iris.domain,
            np.repeat(self.iris.X, 10, axis=0),
            np.repeat(self.iris.Y, 10, axis=0))

        self.send_signal(w.Inputs.data, data)
        output = self.get_output(w.Outputs.annotated_data)

        self.assertEqual(len(data), len(output))
        self.assertTupleEqual(data.X.shape, output.X.shape)
        self.assertTupleEqual(data.Y.shape, output.Y.shape)
        self.assertEqual(2, output.metas.shape[1])

    def test_titanic(self):
        """
        Titanic is a data-set with many 0 in k-nearest neighbours and thus some
        manipulation is required to set cut-point.
        This test checks whether widget works on those type of data.
        """
        w = self.widget
        data = Table("titanic")
        self.send_signal(w.Inputs.data, data)

    def test_missing_data(self):
        w = self.widget
        self.iris[1:5, 1] = np.nan
        self.send_signal(w.Inputs.data, self.iris)
        output = self.get_output(w.Outputs.annotated_data)
        self.assertTupleEqual((150, 1), output[:, "Cluster"].metas.shape)

    def test_summary(self):
        """Check if the status bar updates when data on input"""
        info = self.widget.info
        no_input, no_output = "No data on input", "No data on output"

        self.send_signal(self.widget.Inputs.data, self.iris)
        summary, details = f"{len(self.iris)}", format_summary_details(self.iris)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)
        output = self.get_output(self.widget.Outputs.annotated_data)
        summary, details = f"{len(output)}", format_summary_details(output)
        self.assertEqual(info._StateInfo__output_summary.brief, summary)
        self.assertEqual(info._StateInfo__output_summary.details, details)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)
