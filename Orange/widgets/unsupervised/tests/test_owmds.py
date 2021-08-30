# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import os
from itertools import chain
import unittest
from unittest.mock import patch, Mock

import numpy as np

from Orange.data import Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.projection.manifold import torgerson
from Orange.widgets.settings import Context
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, datasets, ProjectionWidgetTestMixin
)
from Orange.widgets.tests.utils import simulate
from Orange.widgets.unsupervised.owmds import OWMDS, run_mds, Result


class TestOWMDS(WidgetTest, ProjectionWidgetTestMixin,
                WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Distances"
        cls.signal_data = Euclidean(cls.data)
        cls.same_input_output_domain = False

        my_dir = os.path.dirname(__file__)
        datasets_dir = os.path.join(my_dir, '..', '..', '..', 'datasets')
        cls.datasets_dir = os.path.realpath(datasets_dir)

    def setUp(self):
        self.widget = self.create_widget(
            OWMDS, stored_settings={
                "__version__": 2,
                "max_iter": 10,
                "initialization": OWMDS.PCA,
            }
        )  # type: OWMDS
        self.towns = DistMatrix.from_file(
            os.path.join(self.datasets_dir, "slovenian-towns.dst"))

    def tearDown(self):
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_plot_once(self):  # pylint: disable=arguments-differ
        """Test if data is plotted only once but committed on every input change"""
        table = Table("heart_disease")
        self.widget.setup_plot = Mock()
        self.widget.commit.deferred = self.widget.commit.now = Mock()
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.commit.deferred.reset_mock()
        self.wait_until_finished()
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()

        self.widget.commit.deferred.reset_mock()
        self.send_signal(self.widget.Inputs.data_subset, table[::10])
        self.wait_until_stop_blocking()
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()

    def test_pca_init(self):
        self.send_signal(self.signal_name, self.signal_data)
        output = self.get_output(self.widget.Outputs.annotated_data, wait=1000)
        expected = np.array(
            [[-2.69304803, 0.32676458],
             [-2.7246721, -0.20921726],
             [-2.90244761, -0.13630526],
             [-2.75281107, -0.33854819]]
        )
        np.testing.assert_array_almost_equal(output.metas[:4, :2], expected)

    def test_nan_plot(self):
        def combobox_run_through_all():
            cb = self.widget.controls
            simulate.combobox_run_through_all(cb.attr_color)
            # simulate.combobox_run_through_all(cb.attr_shape)
            simulate.combobox_run_through_all(cb.attr_size)
            # simulate.combobox_run_through_all(cb.attr_label)

        data = datasets.missing_data_1()
        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        combobox_run_through_all()

        self.send_signal(self.widget.Inputs.data, None)
        combobox_run_through_all()

        with data.unlocked():
            data.X[:, 0] = np.nan
            data.Y[:] = np.nan
            data.metas[:, 1] = np.nan

        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        combobox_run_through_all()

    @patch("Orange.projection.MDS.__call__", Mock(side_effect=MemoryError))
    def test_out_of_memory(self):
        with patch("sys.excepthook", Mock()) as hook:
            self.send_signal(self.widget.Inputs.data, self.data, wait=1000)
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.out_of_memory.is_shown())

    @patch("Orange.projection.MDS.__call__", Mock(side_effect=ValueError))
    def test_other_error(self):
        with patch("sys.excepthook", Mock()) as hook:
            self.send_signal(self.widget.Inputs.data, self.data, wait=1000)
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.optimization_error.is_shown())

    def test_distances_without_data_0(self):
        """
        Only distances and no data.
        GH-2335
        """
        signal_data = Euclidean(self.data, axis=0)
        signal_data.row_items = None
        self.send_signal("Distances", signal_data)

    def test_distances_without_data_1(self):
        """
        Only distances and no data.
        GH-2335
        """
        signal_data = Euclidean(self.data, axis=1)
        signal_data.row_items = None
        self.send_signal("Distances", signal_data)

    def test_small_data(self):
        data = self.data[:1]
        self.assertFalse(self.widget.Error.not_enough_rows.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        # self.assertTrue(self.widget.Error.not_enough_rows.is_shown())

    def test_run(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.run_button.click()
        self.widget.initialization = 0
        self.widget._OWMDS__invalidate_embedding()  # pylint: disable=protected-access

    def test_migrate_settings_from_version_1(self):
        context_settings = [
            Context(attributes={'iris': 1,
                                'petal length': 2, 'petal width': 2,
                                'sepal length': 2, 'sepal width': 2},
                    metas={},
                    ordered_domain=[('sepal length', 2),
                                    ('sepal width', 2),
                                    ('petal length', 2),
                                    ('petal width', 2),
                                    ('iris', 1)],
                    time=1500000000,
                    values={'__version__': 1,
                            'color_value': ('iris', 1),
                            'shape_value': ('iris', 1),
                            'size_value': ('Stress', -2),
                            'label_value': ('sepal length', 2)})]
        settings = {
            '__version__': 1,
            'autocommit': False,
            'connected_pairs': 5,
            'initialization': 0,
            'jitter': 0.5,
            'label_only_selected': True,
            'legend_anchor': ((1, 0), (1, 0)),
            'max_iter': 300,
            'refresh_rate': 3,
            'symbol_opacity': 230,
            'symbol_size': 8,
            'context_settings': context_settings,
            'savedWidgetGeometry': None
        }
        w = self.create_widget(OWMDS, stored_settings=settings)
        domain = self.data.domain
        self.send_signal(w.Inputs.data, self.data, widget=w)
        g = w.graph
        for a, value in ((w.attr_color, domain["iris"]),
                         (w.attr_shape, domain["iris"]),
                         (w.attr_size, "Stress"),
                         (w.attr_label, domain["sepal length"]),
                         (g.label_only_selected, True),
                         (g.alpha_value, 230),
                         (g.point_width, 8),
                         (g.jitter_size, 0.5)):
            self.assertEqual(a, value)
        self.assertFalse(w.auto_commit)

    def test_attr_label_from_dist_matrix_from_file(self):
        w = self.widget
        # Don't run the MDS optimization to save time and to prevent the
        # widget be in a blocking state when trying to send the next signal
        w.start = Mock()
        row_items = self.towns.row_items

        # Distance matrix with labels
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertIn(row_items.domain["label"], w.controls.attr_label.model())

        # Distances matrix without labels
        self.towns.row_items = None
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertEqual(list(w.controls.attr_label.model()), [None])

        # No data
        self.send_signal(w.Inputs.distances, None)
        self.assertEqual(list(w.controls.attr_label.model()), [None])

        # Distances matrix with labels again
        self.towns.row_items = row_items
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertIn(row_items.domain["label"], w.controls.attr_label.model())

        # Followed by no data
        self.towns.row_items = None
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertEqual(list(w.controls.attr_label.model()), [None])

    def test_attr_label_from_dist_matrix_from_data(self):
        w = self.widget
        # Don't run the MDS optimization to save time and to prevent the
        # widget be in a blocking state when trying to send the next signal
        w.start = Mock()

        data = Table("zoo")
        dist = Euclidean(data)
        self.send_signal(w.Inputs.distances, dist)
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas))
                        < set(w.controls.attr_label.model()))

    def test_attr_label_from_data(self):
        w = self.widget
        # Don't run the MDS optimization to save time and to prevent the
        # widget be in a blocking state when trying to send the next signal
        w.start = Mock()

        data = Table("zoo")
        dist = Euclidean(data)
        self.send_signal(w.Inputs.distances, dist)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas))
                        < set(w.controls.attr_label.model()))

    def test_attr_label_matrix_and_data(self):
        w = self.widget
        # Don't run the MDS optimization to save time and to prevent the
        # widget be in a blocking state when trying to send the next signal
        w.start = Mock()

        # Data and matrix
        data = Table("zoo")
        dist = Euclidean(data)
        self.send_signal(w.Inputs.distances, dist)
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas))
                        < set(w.controls.attr_label.model()))

        # Has data, but receives a signal without data: has to keep the label
        self.send_signal(w.Inputs.distances, None)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas))
                        < set(w.controls.attr_label.model()))

        # Has matrix without data, and loses the data: remove the label
        self.send_signal(w.Inputs.data, None)
        self.assertEqual(list(w.controls.attr_label.model()), [None])

        # Has matrix without data, receives data: add attrs to combo, select
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas))
                        < set(w.controls.attr_label.model()))

    def test_saved_matrix_and_data(self):
        towns_data = self.towns.row_items
        attr_label = self.widget.controls.attr_label
        self.widget.start = Mock()
        self.towns.row_items = None

        # Matrix without data
        self.send_signal(self.widget.Inputs.distances, self.towns)
        self.assertIsNotNone(self.widget.graph.scatterplot_item)
        self.assertEqual(list(attr_label.model()), [None])

        # Data
        self.send_signal(self.widget.Inputs.data, towns_data)
        self.assertIn(towns_data.domain["label"], attr_label.model())

    def test_matrix_columns_tooltip(self):
        dist = Euclidean(self.data, axis=0)
        self.send_signal(self.widget.Inputs.distances, dist)
        self.assertIn("sepal length", self.widget.get_tooltip([0]))

    def test_matrix_columns_labels(self):
        dist = Euclidean(self.data, axis=0)
        self.send_signal(self.widget.Inputs.distances, dist)
        simulate.combobox_activate_index(self.widget.controls.attr_label, 2)

    def test_matrix_columns_default_label(self):
        dist = Euclidean(self.data, axis=0)
        self.send_signal(self.widget.Inputs.distances, dist)
        label_text = self.widget.controls.attr_label.currentText()
        self.assertEqual(label_text, "labels")


class TestOWMDSRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("iris")
        cls.distances = Euclidean(cls.data)
        cls.init = torgerson(cls.distances)
        cls.args = (cls.distances, 300, 25, 0, cls.init)

    def test_Result(self):
        result = Result(embedding=self.init)
        self.assertIsInstance(result.embedding, np.ndarray)

    def test_run_mds(self):
        state = Mock()
        state.is_interruption_requested.return_value = False
        result = run_mds(*(self.args + (state,)))
        array = np.array([[-2.69280967, 0.32544313],
                          [-2.72409383, -0.21287617],
                          [-2.9022707, -0.13465859],
                          [-2.75267253, -0.33899134],
                          [-2.74108069, 0.35393209]])
        np.testing.assert_almost_equal(array, result.embedding[:5])
        state.set_status.assert_called_once_with("Running...")
        self.assertGreater(state.set_partial_result.call_count, 2)
        self.assertGreater(state.set_progress_value.call_count, 2)

    def test_run_do_not_modify_model_inplace(self):
        state = Mock()
        state.is_interruption_requested.return_value = True
        result = run_mds(*(self.args + (state,)))
        state.set_partial_result.assert_called_once()
        self.assertIsNot(self.init, result.embedding)
        self.assertTrue((self.init != result.embedding).any())


if __name__ == "__main__":
    unittest.main()
