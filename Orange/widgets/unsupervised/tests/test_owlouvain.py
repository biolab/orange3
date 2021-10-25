# pylint: disable=protected-access
from unittest.mock import patch

import numpy as np
from sklearn.utils import check_random_state

from orangewidget.settings import Context

from Orange.data import Table, Domain, ContinuousVariable
from Orange.preprocess import Normalize
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import table_dense_sparse
from Orange.widgets.unsupervised.owlouvainclustering import OWLouvainClustering

# Deterministic tests
np.random.seed(42)


class TestOWLouvain(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWLouvainClustering, stored_settings={'auto_commit': False}
        )
        self.iris = Table('iris')[::5].copy()

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def commit_and_wait(self, widget=None):
        widget = self.widget if widget is None else widget
        widget.commit()
        self.wait_until_stop_blocking(widget)

    def test_removing_data(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.commit_and_wait(self.widget)
        self.send_signal(self.widget.Inputs.data, None)
        self.commit_and_wait(self.widget)

    def test_clusters_ordered_by_size(self):
        """Cluster names should be sorted based on the number of instances."""
        x1 = np.array([[0, 0]] * 20)
        x2 = np.array([[1, 0]] * 15)
        x3 = np.array([[0, 1]] * 10)
        x4 = np.array([[1, 1]] * 5)
        data = np.vstack((x1, x2, x3, x4))
        # Remove any order depencence in data, not that this should affect it
        np.random.shuffle(data)

        table = Table.from_numpy(domain=Domain.from_numpy(X=data), X=data)

        self.send_signal(self.widget.Inputs.data, table)
        self.widget.k_neighbors = 4
        self.commit_and_wait()
        output = self.get_output(self.widget.Outputs.annotated_data)

        clustering = output.get_column_view('Cluster')[0].astype(int)
        counts = np.bincount(clustering)
        np.testing.assert_equal(counts, sorted(counts, reverse=True))

    def test_empty_dataset(self):
        # Prepare a table with 5 rows with only meta attributes
        meta = np.array([0] * 5)
        meta_var = ContinuousVariable(name='meta_var')
        table = Table.from_domain(domain=Domain([], metas=[meta_var]), n_rows=5)
        with table.unlocked():
            table.get_column_view(meta_var)[0][:] = meta

        self.send_signal(self.widget.Inputs.data, table)
        self.commit_and_wait()
        self.assertTrue(self.widget.Error.empty_dataset.is_shown())

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

        with patch.object(self.widget, '_invalidate_output') as commit:
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

    def test_only_recluster_when_necessary_pca_components_change(self):
        # Compute clustering on some data
        self.send_signal(self.widget.Inputs.data, self.iris)

        # When PCA checkbox is ticked, any update to slider should invalidate results
        self.widget.apply_pca_cbx.setChecked(True)
        self.widget.pca_components_slider.setValue(2)
        self.commit_and_wait()

        with patch.object(self.widget, '_invalidate_output') as invalidate:
            # Change slider value, this should invalidate output
            self.widget.pca_components_slider.setValue(4)
            self.commit_and_wait()
            self.assertEqual(invalidate.call_count, 1)

        with patch.object(self.widget, '_invalidate_output') as invalidate:
            # Don't change slider value, this shouldn't do anything
            self.widget.pca_components_slider.setValue(4)
            self.commit_and_wait()
            invalidate.assert_not_called()

        # When PCA checkbox is not ticked updating the slider should have no effect
        self.widget.apply_pca_cbx.setChecked(False)
        self.widget.pca_components_slider.setValue(2)
        self.commit_and_wait()

        with patch.object(self.widget, '_invalidate_output') as invalidate:
            # Change slider value, this should invalidate output
            self.widget.pca_components_slider.setValue(4)
            self.commit_and_wait()
            invalidate.assert_not_called()

    def test_invalidate(self):
        # pylint: disable=protected-access
        data = self.iris
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.commit()
        out = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNotNone(out)
        # invalidate the partitioning
        p = self.widget.partition
        g = self.widget.graph
        self.widget._invalidate_partition()
        # not in auto commit mode.
        self.assertTrue(self.widget.Information.modified.is_shown())
        self.widget.commit()
        out = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNotNone(out)
        self.assertIs(self.widget.graph, g)
        self.assertIsNot(self.widget.partition, p)

        self.assertFalse(self.widget.Information.modified.is_shown())
        self.widget._invalidate_graph()
        self.assertIsNone(self.widget.graph)
        self.assertIsNotNone(self.widget.pca_projection)
        self.assertTrue(self.widget.Information.modified.is_shown())
        self.widget._invalidate_pca_projection()
        self.assertIsNone(self.widget.pca_projection)
        self.widget.commit()
        self.get_output(self.widget.Outputs.annotated_data)
        self.assertFalse(self.widget.Information.modified.is_shown())

    def test_deterministic_clustering(self):
        # Compute clustering on iris
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.commit_and_wait()
        result1 = self.get_output(self.widget.Outputs.annotated_data)

        # Reset widget state
        self.send_signal(self.widget.Inputs.data, None)

        # Compute clustering on iris again
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.commit_and_wait()
        result2 = self.get_output(self.widget.Outputs.annotated_data)

        # Ensure that clustering was the same in both instances
        np.testing.assert_equal(result1.metas, result2.metas)

    @table_dense_sparse
    def test_normalize_data(self, prepare_table):
        """Check that normalization is called at the proper times."""
        data = prepare_table(self.iris)

        # Enable checkbox
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, data)
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_called_once()

        # Disable checkbox
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, data)
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_not_called()

    def test_dense_and_sparse_return_same_result(self):
        """Check that Louvain clustering returns identical results for both
        dense and sparse data."""
        random_state = check_random_state(42)

        # Randomly set some values to zero
        dense_data = self.iris
        mask = random_state.beta(1, 2, size=self.iris.X.shape) > 0.5
        with dense_data.unlocked():
            dense_data.X[mask] = 0
        sparse_data = dense_data.to_sparse()

        def _compute_clustering(data):
            self.send_signal(self.widget.Inputs.data, data)
            self.wait_until_stop_blocking()
            result = self.get_output(self.widget.Outputs.annotated_data)
            self.send_signal(self.widget.Inputs.data, None)
            return result

        # Disable normalization
        self.widget.controls.normalize.setChecked(False)
        dense_result = _compute_clustering(dense_data)
        sparse_result = _compute_clustering(sparse_data)
        np.testing.assert_equal(dense_result.metas, sparse_result.metas)

        # Enable normalization
        self.widget.controls.normalize.setChecked(True)
        dense_result = _compute_clustering(dense_data)
        sparse_result = _compute_clustering(sparse_data)
        np.testing.assert_equal(dense_result.metas, sparse_result.metas)

    def test_graph_output(self):
        w = self.widget

        # This test executes only if network add-on is installed
        if not hasattr(w.Outputs, "graph"):
            return

        self.send_signal(w.Inputs.data, self.iris)
        graph = self.get_output(w.Outputs.graph)
        self.assertEqual(len(graph.nodes), len(self.iris))

        self.send_signal(w.Inputs.data, None)
        graph = self.get_output(w.Outputs.graph)
        self.assertIsNone(graph)

    def test_migrate_settings(self):
        # any context settings are removed
        settings = {"context_settings": []}
        self.widget.migrate_settings(settings, 1)
        self.assertEqual(len(settings), 0)

        # context settings become ordinary settings
        settings = {"context_settings": [Context(values={'__version__': 1,
                                                         'apply_pca': (True, -2),
                                                         'k_neighbors': (29, -2),
                                                         'metric_idx': (1, -2),
                                                         'normalize': (False, -2),
                                                         'pca_components': (10, -2),
                                                         'resolution': (1.0, -2)})]}
        self.widget.migrate_settings(settings, 1)
        correct = {'apply_pca': True, 'k_neighbors': 29, 'metric_idx': 1,
                   'normalize': False, 'pca_components': 10, 'resolution': 1.0}
        self.assertEqual(sorted(settings.items()), sorted(correct.items()))
