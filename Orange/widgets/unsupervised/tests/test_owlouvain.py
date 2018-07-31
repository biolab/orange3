from unittest.mock import patch

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owlouvainclustering import OWLouvainClustering

# Deterministic tests
np.random.seed(42)


class TestOWLouvain(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(
            OWLouvainClustering, stored_settings={'auto_commit': False}
        )
        self.iris = Table('iris')

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_removing_data(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.commit_and_wait()
        self.send_signal(self.widget.Inputs.data, None)
        self.commit_and_wait()

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
        self.widget.unconditional_commit()
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
        table.get_column_view(meta_var)[0][:] = meta

        self.send_signal(self.widget.Inputs.data, table)
        self.widget.unconditional_commit()
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
