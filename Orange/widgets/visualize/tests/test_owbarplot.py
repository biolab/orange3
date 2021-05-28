# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,protected-access
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFont

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, simulate, \
    WidgetOutputsTestMixin, datasets
from Orange.widgets.visualize.owbarplot import OWBarPlot


class TestOWBarPlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.titanic = Table("titanic")
        cls.housing = Table("housing")
        cls.heart = Table("heart_disease")

    def setUp(self):
        self.widget = self.create_widget(OWBarPlot)

    def _select_data(self):
        self.widget.graph.select_by_indices(list(range(0, len(self.data), 5)))
        return self.widget.selection

    def test_input_no_cont_features(self):
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertTrue(self.widget.Error.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_cont_features.is_shown())

    @patch("Orange.widgets.visualize.owbarplot.MAX_INSTANCES", 10)
    def test_input_to_many_instances(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTrue(self.widget.Information.too_many_instances.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.too_many_instances.is_shown())

    def test_init_attr_values(self):
        controls = self.widget.controls
        self.assertEqual(controls.selected_var.currentText(), "")
        self.assertEqual(controls.group_var.currentText(), "None")
        self.assertEqual(controls.annot_var.currentText(), "None")
        self.assertEqual(controls.color_var.currentText(), "(Same color)")

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.assertEqual(controls.selected_var.currentText(), "age")
        self.assertEqual(controls.selected_var.model().rowCount(), 6)
        self.assertEqual(controls.group_var.currentText(), "None")
        self.assertEqual(controls.group_var.model().rowCount(), 11)
        self.assertEqual(controls.annot_var.currentText(), "None")
        self.assertEqual(controls.annot_var.model().rowCount(), 12)
        self.assertEqual(controls.color_var.currentText(),
                         "diameter narrowing")
        self.assertEqual(controls.color_var.model().rowCount(), 11)

        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(controls.selected_var.currentText(), "sepal length")
        self.assertEqual(controls.selected_var.model().rowCount(), 4)
        self.assertEqual(controls.group_var.currentText(), "None")
        self.assertEqual(controls.group_var.model().rowCount(), 3)
        self.assertEqual(controls.annot_var.currentText(), "None")
        self.assertEqual(controls.annot_var.model().rowCount(), 4)
        self.assertEqual(controls.color_var.currentText(), "iris")
        self.assertEqual(controls.color_var.model().rowCount(), 3)

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(controls.selected_var.currentText(), "MEDV")
        self.assertEqual(controls.selected_var.model().rowCount(), 15)
        self.assertEqual(controls.group_var.currentText(), "None")
        self.assertEqual(controls.group_var.model().rowCount(), 1)
        self.assertEqual(controls.annot_var.currentText(), "None")
        self.assertEqual(controls.annot_var.model().rowCount(), 2)
        self.assertEqual(controls.color_var.currentText(), "(Same color)")
        self.assertEqual(controls.color_var.model().rowCount(), 1)

        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertEqual(controls.selected_var.currentText(), "")
        self.assertEqual(controls.selected_var.model().rowCount(), 0)
        self.assertEqual(controls.group_var.currentText(), "None")
        self.assertEqual(controls.group_var.model().rowCount(), 1)
        self.assertEqual(controls.annot_var.currentText(), "None")
        self.assertEqual(controls.annot_var.model().rowCount(), 2)
        self.assertEqual(controls.color_var.currentText(), "(Same color)")
        self.assertEqual(controls.color_var.model().rowCount(), 1)

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.assertEqual(controls.selected_var.currentText(), "age")
        self.assertEqual(controls.selected_var.model().rowCount(), 6)
        self.assertEqual(controls.group_var.currentText(), "None")
        self.assertEqual(controls.group_var.model().rowCount(), 11)
        self.assertEqual(controls.annot_var.currentText(), "None")
        self.assertEqual(controls.annot_var.model().rowCount(), 12)
        self.assertEqual(controls.color_var.currentText(),
                         "diameter narrowing")
        self.assertEqual(controls.color_var.model().rowCount(), 11)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(controls.selected_var.currentText(), "")
        self.assertEqual(controls.selected_var.model().rowCount(), 0)
        self.assertEqual(controls.group_var.currentText(), "None")
        self.assertEqual(controls.group_var.model().rowCount(), 1)
        self.assertEqual(controls.annot_var.currentText(), "None")
        self.assertEqual(controls.annot_var.model().rowCount(), 2)
        self.assertEqual(controls.color_var.currentText(), "(Same color)")
        self.assertEqual(controls.color_var.model().rowCount(), 1)

    def test_group_axis(self):
        group_axis = self.widget.graph.group_axis
        annot_axis = self.widget.graph.getAxis('bottom')
        controls = self.widget.controls

        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(group_axis.isVisible())
        self.assertTrue(annot_axis.isVisible())

        simulate.combobox_activate_item(controls.group_var, "iris")
        self.assertTrue(group_axis.isVisible())
        self.assertFalse(annot_axis.isVisible())

        simulate.combobox_activate_item(controls.annot_var, "iris")
        self.assertTrue(group_axis.isVisible())
        self.assertTrue(annot_axis.isVisible())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(group_axis.isVisible())
        self.assertFalse(annot_axis.isVisible())

    def test_datasets(self):
        controls = self.widget.controls
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)
            simulate.combobox_run_through_all(controls.selected_var)
            simulate.combobox_run_through_all(controls.group_var)
            simulate.combobox_run_through_all(controls.annot_var)
            simulate.combobox_run_through_all(controls.color_var)

    def test_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # Select 0:5
        sel_indices = list(range(5))
        self.widget.graph.select_by_indices(sel_indices)
        self.assertSelectedIndices(sel_indices)

        # Shift-select 10:15 (add 10:15 to selection)
        indices = list(range(10, 15))
        sel_indices.extend(indices)
        with self.modifiers(Qt.ShiftModifier):
            self.widget.graph.select_by_indices(indices)
        self.assertSelectedIndices(sel_indices)

        # Select 15:20
        sel_indices = list(range(15, 20))
        self.widget.graph.select_by_indices(sel_indices)
        self.assertSelectedIndices(sel_indices)

        # Control-select 10:17 (add 10:15, remove 15:17)
        indices = list(range(10, 17))
        sel_indices.extend(indices[:5])
        sel_indices.remove(15)
        sel_indices.remove(16)
        sel_indices = sorted(sel_indices)
        with self.modifiers(Qt.ControlModifier):
            self.widget.graph.select_by_indices(indices)
        self.assertSelectedIndices(sel_indices)

        # Alt-select 15:30 (remove 17:20)
        indices = list(range(15, 30))
        sel_indices.remove(17)
        sel_indices.remove(18)
        sel_indices.remove(19)
        sel_indices = sorted(sel_indices)
        with self.modifiers(Qt.AltModifier):
            self.widget.graph.select_by_indices(indices)
        self.assertSelectedIndices(sel_indices)

    def test_retain_selection_on_param_change(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        indices = self._select_data()
        self.assertSelectedIndices(indices)

        controls = self.widget.controls
        simulate.combobox_run_through_all(controls.selected_var)
        simulate.combobox_run_through_all(controls.group_var)
        simulate.combobox_run_through_all(controls.annot_var)
        simulate.combobox_run_through_all(controls.color_var)
        self.assertSelectedIndices(indices)

    def test_data_subset(self):
        subset = list(range(0, 150, 6))
        self.send_signal(self.widget.Inputs.data, self.data[::2])
        self.assertListEqual(self.widget.subset_indices, [])
        self.send_signal(self.widget.Inputs.data_subset, self.data[::3])
        self.assertListEqual(self.widget.subset_indices,
                             list(self.data[subset].ids))
        self.send_signal(self.widget.Inputs.data_subset, None)
        self.assertListEqual(self.widget.subset_indices, [])
        self.send_signal(self.widget.Inputs.data_subset, self.data[::3])
        self.assertListEqual(self.widget.subset_indices,
                             list(self.data[subset].ids))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertListEqual(self.widget.subset_indices, [])
        self.send_signal(self.widget.Inputs.data, self.data[::2])
        self.assertListEqual(self.widget.subset_indices,
                             list(self.data[subset].ids))

    def test_plot_data_subset(self):
        self.send_signal(self.widget.Inputs.data, self.data[::2])
        brushes = self.widget.graph.bar_item.opts["brushes"]
        self.assertTrue(all([color.alpha() == 255 for i, color
                             in enumerate(brushes)]))

        self.send_signal(self.widget.Inputs.data_subset, self.data[::3])
        indices = list(range(0, 75, 3))
        brushes = self.widget.graph.bar_item.opts["brushes"]
        self.assertTrue(all([color.alpha() == 255 for i, color
                             in enumerate(brushes) if i in indices]))
        self.assertTrue(all([color.alpha() == 50 for i, color
                             in enumerate(brushes) if i not in indices]))

        self.send_signal(self.widget.Inputs.data_subset, None)
        brushes = self.widget.graph.bar_item.opts["brushes"]
        self.assertTrue(all([color.alpha() == 255 for i, color
                             in enumerate(brushes)]))

    def test_plot(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNotNone(self.widget.graph.bar_item)
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertIsNotNone(self.widget.graph.bar_item)
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertIsNone(self.widget.graph.bar_item)
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.assertIsNotNone(self.widget.graph.bar_item)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.graph.bar_item)

    def test_saved_workflow(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        indices = self._select_data()
        self.assertSelectedIndices(indices, data=self.heart)

        chol = self.heart.domain["cholesterol"]
        chest = self.heart.domain["chest pain"]
        gender = self.heart.domain["gender"]
        thal = self.heart.domain["thal"]
        controls = self.widget.controls
        simulate.combobox_activate_item(controls.selected_var, chol.name)
        simulate.combobox_activate_item(controls.group_var, chest.name)
        simulate.combobox_activate_item(controls.annot_var, gender.name)
        simulate.combobox_activate_item(controls.color_var, thal.name)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWBarPlot, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.heart, widget=w)
        self.assertEqual(w.controls.selected_var.currentText(), chol.name)
        self.assertEqual(w.controls.group_var.currentText(), chest.name)
        self.assertEqual(w.controls.annot_var.currentText(), gender.name)
        self.assertEqual(w.controls.color_var.currentText(), thal.name)
        self.assertSelectedIndices(indices, data=self.heart, widget=w)

    def test_sparse_data(self):
        table = Table("iris").to_sparse()
        self.assertTrue(sp.issparse(table.X))
        self.send_signal(self.widget.Inputs.data, table)
        self.send_signal(self.widget.Inputs.data_subset, table[::30])
        self.assertEqual(len(self.widget.subset_indices), 5)

    def test_hidden_vars(self):
        data = Table("iris")
        data.domain.attributes[0].attributes["hidden"] = True
        self.send_signal(self.widget.Inputs.data, data)
        controls = self.widget.controls
        self.assertEqual(controls.selected_var.currentText(), "sepal width")
        self.assertEqual(controls.selected_var.model().rowCount(), 3)

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()

    def test_visual_settings(self):
        graph = self.widget.graph
        font = QFont()
        font.setItalic(True)
        font.setFamily("Helvetica")

        self.send_signal(self.widget.Inputs.data, self.data)
        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(20)
        self.assertFontEqual(
            graph.parameter_setter.title_item.item.font(), font
        )

        key, value = ("Fonts", "Axis title", "Font size"), 16
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(16)
        for item in graph.parameter_setter.axis_items:
            self.assertFontEqual(item.label.font(), font)

        key, value = ("Fonts", "Axis ticks", "Font size"), 15
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis ticks", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(15)
        for item in graph.parameter_setter.axis_items:
            self.assertFontEqual(item.style["tickFont"], font)

        key, value = ("Fonts", "Legend", "Font size"), 14
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Legend", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(14)
        legend_item = list(graph.parameter_setter.legend_items)[0]
        self.assertFontEqual(legend_item[1].item.font(), font)

        key, value = ("Annotations", "Title", "Title"), "Foo"
        self.widget.set_visual_settings(key, value)
        self.assertEqual(
            graph.parameter_setter.title_item.item.toPlainText(), "Foo"
        )
        self.assertEqual(graph.parameter_setter.title_item.text, "Foo")

        key, value = ("Figure", "Gridlines", "Show"), False
        self.assertTrue(graph.getAxis("left").grid)
        self.widget.set_visual_settings(key, value)
        self.assertFalse(graph.getAxis("left").grid)

        key, value = ("Figure", "Bottom axis", "Vertical ticks"), False
        self.assertTrue(graph.getAxis("bottom").style["rotateTicks"])
        self.widget.set_visual_settings(key, value)
        self.assertFalse(graph.getAxis("bottom").style["rotateTicks"])

        key, value = ("Figure", "Group axis", "Vertical ticks"), True
        self.assertFalse(graph.group_axis.style["rotateTicks"])
        self.widget.set_visual_settings(key, value)
        self.assertTrue(graph.group_axis.style["rotateTicks"])

    def assertFontEqual(self, font1, font2):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())

    def assertSelectedIndices(self, indices, data=None, widget=None):
        if data is None:
            data = self.data
        if widget is None:
            widget = self.widget
        selected = self.get_output(widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected.X, data[indices].X)

        indices = self.widget.grouped_indices_inverted
        self.assertSetEqual(set(widget.graph.selection), set(indices))
        pens = widget.graph.bar_item.opts["pens"]
        self.assertTrue(all([pen.style() == 2 for i, pen
                             in enumerate(pens) if i in indices]))
        self.assertTrue(all([pen.style() == 1 for i, pen
                             in enumerate(pens) if i not in indices]))


if __name__ == "__main__":
    unittest.main()
