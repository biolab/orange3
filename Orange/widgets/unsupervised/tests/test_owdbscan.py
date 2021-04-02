# pylint: disable=protected-access
import unittest

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from Orange.data import Table
from Orange.clustering import DBSCAN
from Orange.distance import Euclidean
from Orange.preprocess import Normalize, Continuize, SklImpute
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate, possible_duplicate_table
from Orange.widgets.unsupervised.owdbscan import OWDBSCAN, get_kth_distances


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
        self.widget.controls.eps.valueChanged.emit(0.5)
        output1 = self.get_output(w.Outputs.annotated_data)
        self.widget.controls.eps.valueChanged.emit(1)
        output2 = self.get_output(w.Outputs.annotated_data)

        # on this data higher eps has greater sum of clusters - less nan
        # values
        self.assertGreater(np.nansum(output2.metas[:, 0]),
                           np.nansum(output1.metas[:, 0]))

        # try when no data
        self.send_signal(w.Inputs.data, None)
        self.widget.controls.eps.valueChanged.emit(0.5)
        output = self.get_output(w.Outputs.annotated_data)
        self.assertIsNone(output)


    def test_change_min_samples(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)

        # change parameters
        self.widget.controls.min_samples.valueChanged.emit(5)
        output1 = self.get_output(w.Outputs.annotated_data)
        self.widget.controls.min_samples.valueChanged.emit(1)
        output2 = self.get_output(w.Outputs.annotated_data)

        # on this data lower min_samples has greater sum of clusters - less nan
        # values
        self.assertGreater(np.nansum(output2.metas[:, 0]),
                           np.nansum(output1.metas[:, 0]))

        # try when no data
        self.send_signal(w.Inputs.data, None)
        self.widget.controls.min_samples.valueChanged.emit(3)
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
        with self.iris.unlocked():
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
        with self.iris.unlocked():
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

    def test_data_retain_ids(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        output = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_array_equal(self.iris.ids, output.ids)

    def test_missing_data(self):
        w = self.widget
        with self.iris.unlocked():
            self.iris[1:5, 1] = np.nan
        self.send_signal(w.Inputs.data, self.iris)
        output = self.get_output(w.Outputs.annotated_data)
        self.assertTupleEqual((150, 1), output[:, "Cluster"].metas.shape)

    def test_normalize_data(self):
        # not normalized
        self.widget.controls.normalize.setChecked(False)

        data = Table("heart_disease")
        self.send_signal(self.widget.Inputs.data, data)

        kwargs = {"eps": self.widget.eps,
                  "min_samples": self.widget.min_samples,
                  "metric": "euclidean"}
        clusters = DBSCAN(**kwargs)(data)

        output = self.get_output(self.widget.Outputs.annotated_data)
        output_clusters = output.metas[:, 0].copy()
        output_clusters[np.isnan(output_clusters)] = -1
        np.testing.assert_array_equal(output_clusters, clusters)

        # normalized
        self.widget.controls.normalize.setChecked(True)

        kwargs = {"eps": self.widget.eps,
                  "min_samples": self.widget.min_samples,
                  "metric": "euclidean"}
        for pp in (Continuize(), Normalize(), SklImpute()):
            data = pp(data)
        clusters = DBSCAN(**kwargs)(data)

        output = self.get_output(self.widget.Outputs.annotated_data)
        output_clusters = output.metas[:, 0].copy()
        output_clusters[np.isnan(output_clusters)] = -1
        np.testing.assert_array_equal(output_clusters, clusters)

    def test_normalize_changed(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_run_through_all(self.widget.controls.metric_idx)
        self.widget.controls.normalize.setChecked(False)
        simulate.combobox_run_through_all(self.widget.controls.metric_idx)


if __name__ == '__main__':
    unittest.main()
