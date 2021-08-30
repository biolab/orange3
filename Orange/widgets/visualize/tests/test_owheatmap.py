# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import warnings
import unittest
from unittest.mock import patch, Mock

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from AnyQt.QtCore import Qt, QModelIndex

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
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
        cls.brown_selected = Table("brown-selected")

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
        self.widget.MaxClustering = 20
        self.widget.MaxOrderedClustering = 15
        data = self.brown_selected[:, :10]
        self.send_signal(self.widget.Inputs.data, data[:15])
        self.assertFalse(self.widget.Information.active)
        self.send_signal(self.widget.Inputs.data, data[:16])
        self.assertTrue(self.widget.Information.active)
        self.assertEqual(self.widget.row_clustering, Clustering.Clustering)
        self.send_signal(self.widget.Inputs.data, data[:20])
        self.assertFalse(self.widget.Information.active)
        self.send_signal(self.widget.Inputs.data, data[:21])
        self.assertTrue(self.widget.Information.active)

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
        with iris.unlocked():
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
        with patch.object(self.widget.commit, 'now') as commit:
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

    def test_saved_selection_when_not_possible(self):
        # Has stored selection but ot enough columns for clustering.
        iris = Table("iris")[:, ["petal width"]]
        w = self.create_widget(
            OWHeatMap, stored_settings={
                "__version__": 3,
                "col_clustering_method": "Clustering",
                "selected_rows": [1, 2, 3],
            }
        )
        self.send_signal(w.Inputs.data, iris)
        out = self.get_output(w.Outputs.selected_data)
        self.assertSequenceEqual(list(out.ids), list(iris.ids[[1, 2, 3]]))

    def test_set_split_var(self):
        data = self.brown_selected[::3]
        w = self.widget
        self.send_signal(self.widget.Inputs.data, data, widget=w)
        self.assertIs(w.split_by_var, data.domain.class_var)
        self.assertEqual(len(w.parts.rows),
                         len(data.domain.class_var.values))
        w.set_split_variable(None)
        self.assertIs(w.split_by_var, None)
        self.assertEqual(len(w.parts.rows), 1)

    def test_set_split_var_missing(self):
        data = self.brown_selected[::3].copy()
        with data.unlocked():
            data.Y[::5] = np.nan
        w = self.widget
        self.send_signal(self.widget.Inputs.data, data, widget=w)
        self.assertIs(w.split_by_var, data.domain.class_var)
        self.assertEqual(len(w.parts.rows),
                         len(data.domain.class_var.values) + 1)

    def _brown_selected_10(self):
        data = self.brown_selected[::5]
        data = data.transform(
            Domain(data.domain.attributes[:10], data.domain.class_vars,
                   data.domain.metas + (data.domain["diau g"],)))
        data.ensure_copy()
        return data

    def test_set_split_column_key(self):
        data = self._brown_selected_10()
        function = data.domain["function"]
        data_t = data.transpose(data)
        w = self.widget
        self.send_signal(self.widget.Inputs.data, data_t, widget=w)
        w.set_column_split_var(function)
        self.assertEqual(len(w.parts.columns), len(function.values))
        w.set_column_split_var(None)
        self.assertEqual(len(w.parts.columns), 1)

    def test_set_split_column_key_missing(self):
        data = self._brown_selected_10()
        with data.unlocked():
            data.Y[:5] = np.nan
        data_t = data.transpose(data)
        function = data.domain["function"]
        w = self.widget
        self.send_signal(self.widget.Inputs.data, data_t, widget=w)
        w.set_column_split_var(function)
        self.assertEqual(len(w.parts.columns), len(function.values) + 1)
        ncols = sum(len(p.indices) for p in w.parts.columns)
        self.assertEqual(ncols, len(data_t.domain.attributes))
        w.set_column_split_var(None)
        self.assertEqual(len(w.parts.columns), 1)

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

    def test_centering_threshold_change(self):
        data = np.arange(2).reshape(-1, 1)
        table = Table.from_numpy(Domain([ContinuousVariable("y")]), data)
        self.send_signal(self.widget.Inputs.data, table)

        cmw = self.widget.color_map_widget
        palette_index = cmw.findData(
            colorpalettes.ContinuousPalettes["diverging_bwr_40_95_c42"],
            Qt.UserRole)
        cmw.setCurrentIndex(palette_index)

        self.widget.update_color_schema = Mock()
        cmw.centerChanged.emit(42)
        self.widget.update_color_schema.assert_called()

    def test_palette_center(self):
        widget = self.widget
        model = widget.color_map_widget.model()
        for idx in range(model.rowCount(QModelIndex())):
            palette = model.data(model.index(idx, 0), Qt.UserRole)
            if palette is None:
                continue
            widget.color_map_widget.setCurrentIndex(idx)
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
        data = self.brown_selected[::5]
        self.send_signal(widget.Inputs.data, data, widget=widget)
        widget.set_annotation_color_var(data.domain["function"])
        self.assertTrue(widget.scene.widget.right_side_colors[0].isVisible())
        widget.set_annotation_color_var(None)
        self.assertFalse(widget.scene.widget.right_side_colors[0].isVisible())

    def test_row_color_annotations_with_na(self):
        widget = self.widget
        data = self._brown_selected_10()
        with data.unlocked():
            data.Y[:3] = np.nan
            data.metas[:3, -1] = np.nan
        self.send_signal(widget.Inputs.data, data, widget=widget)
        widget.set_annotation_color_var(data.domain["function"])
        self.assertTrue(widget.scene.widget.right_side_colors[0].isVisible())
        widget.set_annotation_color_var(data.domain["diau g"])
        with data.unlocked():
            data.Y[:] = np.nan
            data.metas[:, -1] = np.nan
        self.send_signal(widget.Inputs.data, data, widget=widget)
        widget.set_annotation_color_var(data.domain["function"])
        widget.set_annotation_color_var(data.domain["diau g"])
        widget.set_annotation_color_var(None)
        self.assertFalse(widget.scene.widget.right_side_colors[0].isVisible())

    def test_col_color_annotations(self):
        widget = self.widget
        data = self._brown_selected_10()
        data_t = data.transpose(data)
        self.send_signal(widget.Inputs.data, data_t, widget=widget)
        # discrete
        widget.set_column_annotation_color_var(data.domain["function"])
        self.assertTrue(widget.scene.widget.top_side_colors[0].isVisible())
        # continuous
        widget.set_column_annotation_color_var(data.domain["diau g"])
        widget.set_column_annotation_color_var(None)
        self.assertFalse(widget.scene.widget.top_side_colors[0].isVisible())

    def test_col_color_annotations_with_na(self):
        widget = self.widget
        data = self._brown_selected_10()
        with data.unlocked():
            data.Y[:3] = np.nan
            data.metas[:3, -1] = np.nan
        data_t = data.transpose(data)
        self.send_signal(widget.Inputs.data, data_t, widget=widget)
        widget.set_column_annotation_color_var(data.domain["function"])
        self.assertTrue(widget.scene.widget.top_side_colors[0].isVisible())
        widget.set_column_annotation_color_var(data.domain["diau g"])
        with data.unlocked():
            data.Y[:] = np.nan
            data.metas[:, -1] = np.nan
        data_t = data.transpose(data)
        self.send_signal(widget.Inputs.data, data_t, widget=widget)
        widget.set_column_annotation_color_var(data.domain["function"])
        widget.set_column_annotation_color_var(data.domain["diau g"])
        widget.set_column_annotation_color_var(None)
        self.assertFalse(widget.scene.widget.top_side_colors[0].isVisible())


if __name__ == "__main__":
    unittest.main()
