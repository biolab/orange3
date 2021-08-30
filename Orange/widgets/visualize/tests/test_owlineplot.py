# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,protected-access
import random
import unittest
from unittest.mock import patch, Mock

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QPointF
from AnyQt.QtGui import QFont

import pyqtgraph
from pyqtgraph import PlotCurveItem
from pyqtgraph.Point import Point

from orangewidget.tests.base import DEFAULT_TIMEOUT

from Orange.data import Table
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, datasets
)
from Orange.widgets.visualize.owlineplot import (
    OWLinePlot, ccw, intersects, line_intersects_profiles
)


class TestOWLinePLot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWLinePlot)
        self.titanic = Table("titanic")
        self.housing = Table("housing")

    def _select_data(self):
        random.seed(42)
        indices = random.sample(range(0, len(self.data)), 20)
        self.widget.selection_changed(indices)
        return self.widget.selection

    def test_input_data(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(self.widget.group_view.model().rowCount(), 2)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.group_view.model().rowCount(), 1)

    def test_input_continuous_class(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(self.widget.group_view.model().rowCount(), 1)

    def test_input_discrete_features(self):
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertEqual(self.widget.group_view.model().rowCount(), 1)
        self.assertTrue(self.widget.Error.not_enough_attrs.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_attrs.is_shown())

    def test_input_subset_data(self):
        self.send_signal(self.widget.Inputs.data, self.data[:70])
        self.send_signal(self.widget.Inputs.data_subset, self.data[::10])
        self.assertEqual(len(self.widget.subset_indices), 7)
        self.widget.controls.show_profiles.click()
        self.widget.selection_changed(range(20))
        self.assertEqual(len(self.widget.selection), 20)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.selection)

    def test_select(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # Select 0:5
        sel_indices = list(range(5))
        self.widget.selection_changed(sel_indices)
        selected = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected.X, self.data[sel_indices].X)

        # Shift-select 10:15 (add 10:15 to selection)
        indices = list(range(10, 15))
        sel_indices.extend(indices)
        with self.modifiers(Qt.ShiftModifier):
            self.widget.selection_changed(indices)
        selected = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected.X, self.data[sel_indices].X)

        # Select 15:20
        sel_indices = list(range(15, 20))
        self.widget.selection_changed(sel_indices)
        selected = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected.X, self.data[sel_indices].X)

        # Control-select 10:17 (add 10:15, remove 15:17)
        indices = list(range(10, 17))
        sel_indices.extend(indices[:5])
        sel_indices.remove(15)
        sel_indices.remove(16)
        sel_indices = sorted(sel_indices)
        with self.modifiers(Qt.ControlModifier):
            self.widget.selection_changed(indices)
        selected = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected.X, self.data[sel_indices].X)

        # Alt-select 15:30 (remove 17:20)
        indices = list(range(15, 30))
        sel_indices.remove(17)
        sel_indices.remove(18)
        sel_indices.remove(19)
        sel_indices = sorted(sel_indices)
        with self.modifiers(Qt.AltModifier):
            self.widget.selection_changed(indices)
        selected = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected.X, self.data[sel_indices].X)

    def test_saved_selection(self):
        data = self.data.copy()
        with data.unlocked():
            data[0, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        mask = np.zeros(len(data) - 1, dtype=bool)
        mask[::10] = True
        self.widget.selection_changed(mask)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWLinePlot, stored_settings=settings)
        self.send_signal(w.Inputs.data, data, widget=w)
        self.assertEqual(len(w.graph.selection), np.sum(mask))
        np.testing.assert_equal(self.widget.graph.selection, w.graph.selection)
        output = self.get_output(w.Outputs.selected_data, widget=w)
        self.assertIsNotNone(output)
        self.send_signal(w.Inputs.data, data, widget=w)
        self.assertEqual(len(w.graph.selection), 0)
        self.assertIsNone(self.get_output(w.Outputs.selected_data, widget=w))

    def test_selection_line(self):
        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = Point(0, 0)
        event.pos.return_value = Point(1, 1)
        event.isFinish.return_value = True

        # drag a line before data is sent
        self.widget.graph.view_box.mouseDragEvent(event)
        self.assertIsNone(self.widget.selection)

        # drag a line after data is sent
        self.send_signal(self.widget.Inputs.data, self.data)

        # set view-dependent click coordinates
        pos_down, pos_up = QPointF(2.38, 4.84), QPointF(3.58, 4.76)
        mapToParent = self.widget.graph.view_box.childGroup.mapToParent

        event.buttonDownPos.return_value = mapToParent(pos_down)
        event.pos.return_value = mapToParent(pos_up)

        self.widget.graph.view_box.mouseDragEvent(event)
        line = self.widget.graph.view_box.selection_line
        self.assertFalse(line.line().isNull())

        # click on the plot resets selection
        self.assertEqual(len(self.widget.selection), 55)
        self.widget.graph.view_box.mouseClickEvent(event)
        self.assertListEqual(self.widget.selection, [])

    @patch("Orange.widgets.visualize.owlineplot.SEL_MAX_INSTANCES", 100)
    def test_select_lines_enabled(self):
        self.send_signal(self.widget.Inputs.data, self.data[::2])
        self.assertTrue(self.widget.graph.view_box._can_select)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.graph.view_box._can_select)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertTrue(self.widget.graph.view_box._can_select)

    @patch("Orange.widgets.visualize.owlineplot.MAX_FEATURES", 2)
    def test_max_features(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(len(self.widget.graph_variables), 2)
        self.assertTrue(self.widget.Information.too_many_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.too_many_features.is_shown())

    def test_data_with_missing_values(self):
        data = self.data.copy()
        with data.unlocked():
            data[0, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Information.hidden_instances.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.hidden_instances.is_shown())

    def test_display_options(self):
        self.send_signal(self.widget.Inputs.data, self.data[::10])
        # Plot lines, range, mean and error bars
        self.widget.controls.show_profiles.click()
        self.widget.controls.show_error.click()
        # Plot range, mean and error bars
        self.widget.controls.show_profiles.click()
        # Plot mean and error bars
        self.widget.controls.show_range.click()
        # Only show error bars is selected
        self.widget.controls.show_mean.click()
        self.assertTrue(self.widget.Warning.no_display_option.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.no_display_option.is_shown())

    def test_group_view(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        model = self.widget.group_view.model()
        self.assertEqual(len(model), 2)
        index = self.widget.group_view.selectedIndexes()[0]
        self.assertEqual(model.data(index), self.data.domain.class_var.name)
        self.send_signal(self.widget.Inputs.data, self.housing)
        index = self.widget.group_view.selectedIndexes()[0]
        self.assertEqual(model.data(index), "None")

    def test_group_var_none(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        index = self.widget.group_view.model().index(0)
        self.widget.group_view.setCurrentIndex(index)
        m, n, p = self.widget.graph.view_box._profile_items.shape
        self.assertEqual(m, len(self.data.domain.attributes))
        self.assertEqual(n, len(self.data))
        self.assertEqual(p, 2)
        self.assertFalse(self.widget.graph.legend.isVisible())

    def test_group_var_none_single_instance(self):
        self.send_signal(self.widget.Inputs.data, self.housing[:1])
        m, n, p = self.widget.graph.view_box._profile_items.shape
        self.assertEqual(m, len(self.housing.domain.attributes))
        self.assertEqual(n, 1)
        self.assertEqual(p, 2)
        self.assertFalse(self.widget.graph.legend.isVisible())

    def test_datasets(self):
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_valid_data.is_shown())

    def test_none_data(self):
        self.send_signal(self.widget.Inputs.data, self.data[:0])
        self.widget.controls.show_profiles.click()

    def test_plot_subset(self):
        settings = self.widget.settingsHandler.pack_data(self.widget)
        settings["show_range"] = False
        w = self.create_widget(OWLinePlot, stored_settings=settings)
        w.graph.addItem = Mock()

        self.send_signal(w.Inputs.data, self.data, widget=w)
        w.graph.addItem.assert_called()
        w.graph.addItem.reset_mock()

        # don't show subset instances
        self.send_signal(w.Inputs.data_subset, self.data[::10], widget=w)
        w.graph.addItem.assert_not_called()

        # show subset instances
        w.controls.show_profiles.setChecked(True)
        w.graph.addItem.assert_called()

    def test_plot_only_mean(self):
        settings = self.widget.settingsHandler.pack_data(self.widget)
        settings["show_range"] = False
        w = self.create_widget(OWLinePlot, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.data, widget=w)
        curves = [i for i in w.graph.items() if isinstance(i, PlotCurveItem)]
        self.assertEqual(len(curves), 3)

    def test_sparse_data(self):
        table = Table("iris").to_sparse()
        self.assertTrue(sp.issparse(table.X))
        self.send_signal(self.widget.Inputs.data, table)
        self.send_signal(self.widget.Inputs.data_subset, table[::30])
        self.assertEqual(len(self.widget.subset_indices), 5)

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as commit:
            self.widget.auto_commit = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.titanic)
            commit.assert_called()

    def test_visual_settings(self, timeout=DEFAULT_TIMEOUT):
        graph = self.widget.graph
        font = QFont()
        font.setItalic(True)
        font.setFamily("Helvetica")

        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished(timeout=timeout)

        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(20)
        self.assertFontEqual(graph.parameter_setter.title_item.item.font(), font)

        key, value = ("Fonts", "Axis title", "Font size"), 14
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(14)
        for ax in ["bottom", "left"]:
            axis = graph.parameter_setter.getAxis(ax)
            self.assertFontEqual(axis.label.font(), font)

        key, value = ('Fonts', 'Axis ticks', 'Font size'), 15
        self.widget.set_visual_settings(key, value)
        key, value = ('Fonts', 'Axis ticks', 'Italic'), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(15)
        for ax in ["bottom", "left"]:
            axis = graph.parameter_setter.getAxis(ax)
            self.assertFontEqual(axis.style["tickFont"], font)

        key, value = ("Fonts", "Legend", "Font size"), 16
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Legend", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(16)
        legend_item = list(graph.parameter_setter.legend_items)[0]
        self.assertFontEqual(legend_item[1].item.font(), font)

        key, value = ("Annotations", "Title", "Title"), "Foo"
        self.widget.set_visual_settings(key, value)
        self.assertEqual(graph.parameter_setter.title_item.item.toPlainText(), "Foo")
        self.assertEqual(graph.parameter_setter.title_item.text, "Foo")

        key, value = ("Annotations", "x-axis title", "Title"), "Foo2"
        self.widget.set_visual_settings(key, value)
        axis = graph.parameter_setter.getAxis("bottom")
        self.assertEqual(axis.label.toPlainText().strip(), "Foo2")
        self.assertEqual(axis.labelText, "Foo2")

        key, value = ("Annotations", "y-axis title", "Title"), "Foo3"
        self.widget.set_visual_settings(key, value)
        axis = graph.parameter_setter.getAxis("left")
        self.assertEqual(axis.label.toPlainText().strip(), "Foo3")
        self.assertEqual(axis.labelText, "Foo3")

    def assertFontEqual(self, font1, font2):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())


class TestSegmentsIntersection(unittest.TestCase):
    def test_ccw(self):
        a = np.array([[4, 1], [1, 1]])
        b = np.array([[3, 2], [2, 2]])
        c = np.array([[2, 1], [3, 3]])
        np.testing.assert_array_equal(np.array([True, False]), ccw(a, b, c))

    def test_intersects(self):
        a = np.array([1, 0])
        b = np.array([2, -1])
        c = np.array([[2, -2], [1, -1], [0, 0], [1, -0.01]])
        d = np.array([[4, 0], [2, 0], [-1, -1], [2, 0]])
        np.testing.assert_array_equal(np.array([False, True, False, True]),
                                      intersects(a, b, c, d))

    def test_lines_intersection(self):
        a = np.array([1, 0])
        b = np.array([2, -1])
        # create data with four instances and three features
        x = np.array([[2, 4, 4], [1, 2, 2], [0, -1, 2], [1, 2, 2]])
        y = np.array([[-2, 0, 1], [-1, 0, 1], [0, -1, 0], [-0.01, 0, 3]])
        table = np.array([np.vstack((x[:, i], y[:, i])).T
                          for i in range(y.shape[1])])
        i = line_intersects_profiles(a, b, table)
        np.testing.assert_array_equal(np.array([False, True, True, True]), i)


if __name__ == "__main__":
    unittest.main()
