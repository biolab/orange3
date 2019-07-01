import numpy as np
from scipy.sparse import csr_matrix

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.unsupervised.owdbscan import OWDBSCAN


class TestOWCSVFileImport(WidgetTest):
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

        self.send_signal(w.Inputs.data, self.iris[:5])
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

    def test_sparse_data(self):
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
