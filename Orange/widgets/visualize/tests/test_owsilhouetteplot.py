# pylint: disable=protected-access
# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import random
import unittest
from unittest.mock import Mock

import numpy as np

from orangewidget.settings import Context

import Orange.distance
from Orange.data import (
    Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable)
from Orange.misc import DistMatrix
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.visualize.owsilhouetteplot import OWSilhouettePlot
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.tests.utils import possible_duplicate_table


class TestOWSilhouettePlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.same_input_output_domain = False

        cls.signal_name = OWSilhouettePlot.Inputs.data
        cls.signal_data = cls.data
        cls.scorename = "Silhouette ({})".format(cls.data.domain.class_var.name)

    def setUp(self):
        self.widget = self.create_widget(OWSilhouettePlot,
                                         stored_settings={"auto_commit": True})
        self.widget = self.widget  # type: OWSilhouettePlot

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data, self.data[:0])

    def test_outputs_add_scores(self):
        # check output when appending scores
        self.send_signal(self.widget.Inputs.data, self.data)
        selected_indices = self._select_data()
        selected = self.get_output(self.widget.Outputs.selected_data)
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(self.scorename, selected.domain.metas[0].name)
        self.assertEqual(self.scorename, annotated.domain.metas[0].name)
        np.testing.assert_array_equal(selected.X, self.data.X[selected_indices])

    def _select_data(self):
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget._silplot.setSelection(points)
        return sorted(points)

    def test_insufficient_clusters(self):
        iris = self.data
        data_one_cluster = iris[:3]  # three instances Iris-setosa only
        self.send_signal(self.widget.Inputs.data, data_one_cluster)
        self.assertTrue(self.widget.Error.need_two_clusters.is_shown())

        data_singletons = iris[[0, 50, 100]]
        assert len(np.unique(data_singletons.Y)) == 3  # 3 instances 3 labels
        self.send_signal(self.widget.Inputs.data, data_singletons)
        self.assertTrue(self.widget.Error.singleton_clusters_all.is_shown())

    def test_not_symmetric(self):
        w = self.widget
        self.send_signal(w.Inputs.data, DistMatrix([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(w.Error.input_validation_error.is_shown())
        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Error.input_validation_error.is_shown())

    def test_unknowns_in_labels(self):
        data = self.data[[0, 1, 2, 50, 51, 52, 100, 101, 102]]
        with data.unlocked(data.Y):
            data.Y[::3] = np.nan
        valid = ~np.isnan(data.Y.flatten())
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        scores = output[:, self.scorename].metas.flatten()
        self.assertTrue(np.all(np.isnan(scores[::3])))
        self.assertTrue(np.all(np.isfinite(scores[valid])))

        # Run again on subset with known labels
        data_1 = data[np.flatnonzero(valid)]
        self.send_signal(self.widget.Inputs.data, data_1)
        output_1 = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        scores_1 = output_1[:, self.scorename].metas.flatten()
        self.assertTrue(np.all(np.isfinite(scores_1)))
        # the scores must match
        np.testing.assert_almost_equal(scores_1, scores[valid], decimal=12)

    def test_nan_distances(self):
        self.widget.distance_idx = 2
        self.assertEqual(self.widget.Distances[self.widget.distance_idx][0],
                         'Cosine')
        data = self.data[[0, 1, 2, 50, 51, 52, 100, 101, 102]]
        with data.unlocked(data.X):
            data.X[::3] = 0
        valid = np.any(data.X != 0, axis=1)
        self.assertFalse(self.widget.Warning.nan_distances.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(np.isnan(self.widget._matrix).any())
        self.assertTrue(self.widget.Warning.nan_distances.is_shown())
        output = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        scores = output[:, self.scorename].metas.flatten()
        self.assertTrue(np.all(np.isnan(scores[::3])))
        self.assertTrue(np.all(np.isfinite(scores[valid])))

    def test_ignore_categorical(self):
        data = Table('heart_disease')
        self.widget.distance_idx = 2
        self.assertEqual(self.widget.Distances[self.widget.distance_idx][0],
                         'Cosine')
        self.assertFalse(self.widget.Warning.ignoring_categorical.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.ignoring_categorical.is_shown())
        output = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(len(output.domain.variables), len(data.domain.variables))
        self.widget.distance_idx = 0
        self.widget._update()
        self.assertFalse(self.widget.Warning.ignoring_categorical.is_shown())

    def test_meta_object_dtype(self):
        # gh-1875: Test on mixed string/discrete metas
        data = self.data[::5]
        domain = Domain(data.domain.attributes,
                        [],
                        [data.domain["iris"], StringVariable("S")])
        data = data.from_table(domain, data)
        self.send_signal(self.widget.Inputs.data, data)

    def test_memory_error(self):
        """
        Handling memory error.
        GH-2336
        Handling value error as well. This value error is in a relation with memory error.
        GH-2521
        """
        for i, side_effect in enumerate([MemoryError, ValueError]):
            data = Table("iris")[::3]
            self.send_signal(self.widget.Inputs.data, data)
            self.assertFalse(self.widget.Error.memory_error.is_shown())
            self.assertFalse(self.widget.Error.value_error.is_shown())
            with unittest.mock.patch("numpy.asarray", side_effect=side_effect):
                self.widget._matrix = None
                self.widget.data = data
                self.widget._effective_data = data
                self.widget._update()
                self.assertTrue(self.widget.Error.memory_error.is_shown() != i)
                self.assertTrue(self.widget.Error.value_error.is_shown() == i)

    def test_bad_data_range(self):
        """
        Silhouette Plot now sets axis range properly.
        GH-2377
        """
        nan = np.NaN
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a"), ContinuousVariable("b"), ContinuousVariable("c")],
                [DiscreteVariable("d", values=("y", "n"))]),
            list(zip([4, nan, nan],
                     [15, nan, nan],
                     [16, nan, nan],
                     "nyy"))
        )
        self.send_signal(self.widget.Inputs.data, table)

    def test_saved_selection(self):
        self.widget.settingsHandler.pack_data(self.widget)
        # no assert here, just test is doesn't crash on empty

        iris = Table("iris")

        self.send_signal(self.widget.Inputs.data, iris)
        random.seed(42)
        points = random.sample(range(0, len(self.data)), 20)
        self.widget._silplot.setSelection(points)
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWSilhouettePlot, stored_settings=settings)
        self.send_signal(w.Inputs.data, iris, widget=w)
        self.assertEqual(len(self.get_output(w.Outputs.selected_data)), 20)

    def test_distance_input(self):
        widget = self.widget
        data = Table("heart_disease")[::4]
        matrix = Orange.distance.Euclidean(data)
        self.send_signal(widget.Inputs.data, matrix, widget=widget)
        self.assertIsNotNone(widget.distances)
        self.assertIsNotNone(widget.data)
        self.assertFalse(widget._distances_gui_box.isEnabled())

        self.send_signal(widget.Inputs.data, data, widget=widget)
        self.assertIsNone(widget.distances)
        self.assertIsNotNone(widget.data)
        self.assertTrue(widget._distances_gui_box.isEnabled())

    def test_input_distance_no_data(self):
        widget = self.widget
        matrix = DistMatrix(
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            row_items=None
        )
        self.send_signal(widget.Inputs.data, matrix, widget=widget)
        self.assertTrue(widget.Error.input_validation_error.is_shown())
        self.assertIsNone(widget.data)
        self.assertIsNone(widget.distances)
        self.send_signal(widget.Inputs.data, None, widget=widget)
        self.assertFalse(widget.Error.input_validation_error.is_shown())

    def test_no_group_var(self):
        widget = self.widget
        data = Table("iris")[::4]
        data = data[:, data.domain.attributes]
        matrix = Orange.distance.Euclidean(data)
        self.send_signal(widget.Inputs.data, matrix, widget=widget)

        self.assertTrue(widget.Error.input_validation_error.is_shown())
        self.assertIsNone(widget.data)
        self.assertIsNone(widget.distances)

        self.send_signal(widget.Inputs.data, None, widget=widget)
        self.assertFalse(widget.Error.input_validation_error.is_shown())

    def test_unique_output_domain(self):
        widget = self.widget
        data = possible_duplicate_table('Silhouette (iris)')
        matrix = Orange.distance.Euclidean(data)
        self.send_signal(widget.Inputs.data, matrix, widget=widget)

        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(output.domain.metas[0].name, 'Silhouette (iris) (1)')

    def test_report(self):
        widget = self.widget
        widget.report_plot = Mock()
        widget.report_caption = Mock()

        widget.send_report()
        widget.report_plot.assert_not_called()
        widget.report_caption.assert_not_called()

        data = Table("zoo")
        self.send_signal(widget.Inputs.data, data)

        widget.annotation_var = None
        widget.send_report()
        widget.report_plot.assert_called()
        widget.report_caption.assert_called()
        text = widget.report_caption.call_args[0][0]
        self.assertIn(data.domain.class_var.name, text)
        self.assertNotIn("nnotated", text)

        widget.annotation_var = data.domain.metas[0]
        widget._silplot.rowNamesVisible = lambda: True
        widget.send_report()
        text = widget.report_caption.call_args[0][0]
        self.assertIn(data.domain.class_var.name, text)
        self.assertIn("nnotated", text)
        self.assertIn(data.domain.metas[0].name, text)

    def test_migration(self):
        enc_domain = dict(
            attributes=(('foo', 1), ('bar', 1), ('baz', 1), ('bax', 1),
                        ('cfoo', 1), ('mbaz', 1)))

        # No annotation
        context = Context(
            values=dict(cluster_var_idx=(0, -2), annotation_var_idx=(0, -2)),
            **enc_domain
        )
        OWSilhouettePlot.migrate_context(context, 1)
        values = context.values
        self.assertNotIn("cluster_var_idx", values)
        self.assertNotIn("annotation_var_idx", values)
        self.assertEqual(values["cluster_var"], ("foo", 101))
        self.assertEqual(values["annotation_var"], None)

        # We have annotation
        context = Context(
            values=dict(cluster_var_idx=(2, -2), annotation_var_idx=(4, -2)),
            **enc_domain
        )
        OWSilhouettePlot.migrate_context(context, 1)
        self.assertNotIn("cluster_var_idx", values)
        self.assertNotIn("annotation_var_idx", values)
        self.assertEqual(context.values["cluster_var"], ("baz", 101))
        self.assertEqual(context.values["annotation_var"], ("bax", 101))

        # We thought was had annotation, but the index is wrong due to
        # incorrect domain
        context = Context(
            values=dict(cluster_var_idx=(4, -2), annotation_var_idx=(7, -2)),
            **enc_domain
        )
        OWSilhouettePlot.migrate_context(context, 1)
        self.assertEqual(context.values["cluster_var"], ("cfoo", 101))
        self.assertNotIn("annotation_var_idx", values)


if __name__ == "__main__":
    unittest.main()
