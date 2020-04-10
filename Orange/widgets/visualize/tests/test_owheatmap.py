# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import warnings
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from AnyQt.QtCore import Qt, QModelIndex

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.preprocess import Continuize
from Orange.widgets.utils import colorpalettes
from Orange.widgets.visualize.owheatmap import OWHeatMap, Clustering
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, datasets


def image_row_colors(image):
    colors = np.full((image.height(), 3), np.nan)
    for r in range(image.height()):
        c = image.pixelColor(0, r)
        colors[r] = c.red(), c.green(), c.blue()
    return colors


class TestOWHeatMap(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.housing = Table("housing")
        cls.titanic = Table("titanic")

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWHeatMap)  # type: OWHeatMap

    def test_input_data(self):
        """Check widget's data with data on the input"""
        for data in (self.data, self.housing):
            self.assertEqual(self.widget.data, None)
            self.send_signal(self.widget.Inputs.data, data)
            self.assertEqual(self.widget.data, data)
            self.assertFalse(self.widget.Error.active)
            self.assertFalse(self.widget.Warning.active)
            self.assertFalse(self.widget.Information.active)
            self.send_signal(self.widget.Inputs.data, None)

        self.send_signal(self.widget.Inputs.data, self.data[:0])

    def test_error_message(self):
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertTrue(self.widget.Error.active)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.Error.active)

    def test_information_message(self):
        self.widget.set_row_clustering(Clustering.OrderedClustering)
        continuizer = Continuize()
        cont_titanic = continuizer(self.titanic)
        self.widget.MaxClustering = 1000
        self.send_signal(self.widget.Inputs.data, cont_titanic)
        self.assertTrue(self.widget.Information.active)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.Information.active)

    def test_settings_changed(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        # check output when "Sorting Column" setting changes
        self._select_data()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.widget.set_col_clustering(Clustering.OrderedClustering)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        # check output when "Sorting Row" setting changes
        self._select_data()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.widget.set_row_clustering(Clustering.OrderedClustering)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        # check output when "Merge by k-means" setting changes
        self._select_data()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.widget.controls.merge_kmeans.setChecked(True)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def _select_data(self):
        selected_indices = list(range(10, 31))
        self.widget.scene.widget.selectRows(selected_indices)
        self.widget.on_selection_finished()
        return selected_indices

    def test_not_enough_data_settings_changed(self):
        """Check widget for dataset with one feature or for one instance"""
        # Test tests that widget handles this exact condition
        warnings.filterwarnings("ignore", "Number of distinct clusters",
                                ConvergenceWarning)

        msg = self.widget.Error
        clusterings = (Clustering.None_, Clustering.Clustering,
                       Clustering.OrderedClustering)
        for kmeans_checked in (False, True):
            self.widget.controls.merge_kmeans.setChecked(kmeans_checked)
            for col_clust in clusterings:
                self.widget.set_col_clustering(col_clust)
                self.send_signal(self.widget.Inputs.data, None)
                self.send_signal(self.widget.Inputs.data, self.data[:, 0])
                if col_clust != Clustering.None_:
                    self.assertTrue(msg.not_enough_features.is_shown())
                for row_clust in clusterings:
                    self.widget.set_row_clustering(row_clust)
                    self.send_signal(self.widget.Inputs.data, None)
                    self.send_signal(self.widget.Inputs.data, self.data[0:1])
                    if row_clust != Clustering.None_:
                        self.assertTrue(msg.not_enough_instances.is_shown())
                    elif kmeans_checked and row_clust != Clustering.None_:
                        self.assertTrue(msg.not_enough_instances_k_means.is_shown())
            self.send_signal(self.widget.Inputs.data, None)
            self.assertFalse(msg.not_enough_features.is_shown())
            self.assertFalse(msg.not_enough_instances.is_shown())
            self.assertFalse(msg.not_enough_instances_k_means.is_shown())

    def test_color_low_high(self):
        """
        Prevent horizontal sliders to set Low >= High.
        GH-2025
        """
        self.widget.controls.threshold_low.setValue(4)
        self.widget.controls.threshold_high.setValue(2)
        self.assertGreater(self.widget.threshold_high, self.widget.threshold_low)

    def test_data_column_nans(self):
        """
        Send data with one column with all values set to NaN.
        ValueError should not be thrown (Invalid number of variable columns)
        That column is now suppose to be removed in a table array and
        in a domain as well.
        GH-2057
        """
        table = datasets.data_one_column_nans()
        self.widget.controls.merge_kmeans.setChecked(True)
        self.send_signal(self.widget.Inputs.data, table)

    def test_cluster_column_on_all_zero_column(self):
        # Pearson distance used for clustering of columns does not
        # handle all zero columns well
        iris = Table("iris")
        iris[:, 0] = 0

        self.widget.col_clustering = True
        self.widget.set_dataset(iris)

    def test_empty_clusters(self):
        """Test if empty clusters are not displayed and warning is shown"""
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

        table = Table.from_numpy(Domain([ContinuousVariable("y")]),
                                 data.reshape((9, 1)))
        self.widget.controls.merge_kmeans.setChecked(True)

        # Test tests that widget handles this exact condition
        warnings.filterwarnings("ignore", "Number of distinct clusters",
                                ConvergenceWarning)
        self.send_signal(self.widget.Inputs.data, table)

        self.assertTrue(self.widget.Warning.empty_clusters.is_shown())
        self.assertEqual(len(self.widget.merge_indices), 3)

    def test_use_enough_colors(self):
        # Before 201906 thresholds modified the palette and decreased
        # the number of colors used.
        data = np.arange(1000).reshape(-1, 1)
        table = Table.from_numpy(Domain([ContinuousVariable("y")]), data)
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.threshold_high = 0.05
        self.widget.update_color_schema()
        heatmap_widget = self.widget.scene.widget.heatmap_widget_grid[0][0]
        image = heatmap_widget.pixmap().toImage()
        colors = image_row_colors(image)
        unique_colors = len(np.unique(colors, axis=0))
        self.assertLessEqual(len(data)*self.widget.threshold_low, unique_colors)

    def test_cls_with_single_instance(self):
        table = Table(Domain([ContinuousVariable("c1")],
                             [DiscreteVariable("c2", values=("a", "b"))]),
                      np.array([[1], [2], [3]]), np.array([[0], [0], [1]]))
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.set_row_clustering(Clustering.Clustering)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget, 'unconditional_commit') as commit:
            self.widget.auto_commit = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.titanic)
            commit.assert_called()

    def test_saved_selection(self):
        iris = Table("iris")

        self.send_signal(self.widget.Inputs.data, iris)
        selected_indices = list(range(10, 31))
        self.widget.scene.widget.selectRows(selected_indices)
        self.widget.on_selection_finished()
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWHeatMap, stored_settings=settings)
        self.send_signal(w.Inputs.data, iris, widget=w)
        self.assertEqual(len(self.get_output(w.Outputs.selected_data)), 21)

    def test_set_split_var(self):
        data = Table("brown-selected")
        w = self.widget
        self.send_signal(self.widget.Inputs.data, data, widget=w)
        self.assertIs(w.split_by_var, data.domain.class_var)
        self.assertEqual(len(w.parts.rows),
                         len(data.domain.class_var.values))
        w.set_split_variable(None)
        self.assertIs(w.split_by_var, None)
        self.assertEqual(len(w.parts.rows), 1)

    def test_palette_centering(self):
        data = np.arange(2).reshape(-1, 1)
        table = Table.from_numpy(Domain([ContinuousVariable("y")]), data)
        self.send_signal(self.widget.Inputs.data, table)

        self.widget.color_palette = lambda: \
            colorpalettes.ContinuousPalette.from_colors(
                (0, 255, 0), (255, 0, 0), (0, 0, 0)).lookup_table()

        desired_uncentered = [[0, 255, 0],
                              [255, 0, 0]]

        desired_centered = [[0, 0, 0],
                            [255, 0, 0]]

        for center, desired in [(False, desired_uncentered), (True, desired_centered)]:
            with patch.object(OWHeatMap, "center_palette", center):
                self.widget.update_color_schema()
                heatmap_widget = self.widget.scene.widget.heatmap_widget_grid[0][0]
                image = heatmap_widget.pixmap().toImage()
                colors = image_row_colors(image)
                np.testing.assert_almost_equal(colors, desired)

    def test_palette_center(self):
        widget = self.widget
        model = widget.color_cb.model()
        for idx in range(model.rowCount(QModelIndex())):
            palette = model.data(model.index(idx, 0), Qt.UserRole)
            if palette is None:
                continue
            widget.color_cb.setCurrentIndex(idx)
            self.assertEqual(widget.center_palette,
                             bool(palette.flags & palette.Diverging))

    def test_migrate_settings_v3(self):
        w = self.create_widget(
            OWHeatMap, stored_settings={
                "row_clustering": False,
                "col_clustering": True,
            }
        )
        self.assertEqual(w.row_clustering, Clustering.None_)
        self.assertEqual(w.col_clustering, Clustering.OrderedClustering)

    def test_row_color_annotations(self):
        widget = self.widget
        data = Table("brown-selected")[::5]
        self.send_signal(widget.Inputs.data, data, widget=widget)
        widget.set_annotation_color_var(data.domain["function"])
        self.assertTrue(widget.scene.widget.right_side_colors[0].isVisible())
        widget.set_annotation_color_var(None)
        self.assertFalse(widget.scene.widget.right_side_colors[0].isVisible())


if __name__ == "__main__":
    unittest.main()
