# pylint: disable=protected-access
import unittest
from unittest.mock import patch, Mock

import numpy as np

from AnyQt.QtCore import QItemSelectionModel, QPointF, Qt
from AnyQt.QtGui import QFont

from pyqtgraph import ViewBox

from Orange.data import Table
from Orange.widgets.tests.base import datasets, simulate, \
    WidgetOutputsTestMixin, WidgetTest
from Orange.widgets.visualize.owviolinplot import OWViolinPlot, \
    ViolinPlotViewBox, scale_density, WIDTH


class TestUtils(unittest.TestCase):
    # pylint: disable=no-self-use
    def test_scale_density_retain_original_data(self):
        array = np.arange(10)
        scaled = scale_density(WIDTH, array, 15, 20)
        np.testing.assert_array_equal(array, np.arange(10))
        np.testing.assert_array_equal(scaled, np.arange(10) / 20)


class TestOWViolinPlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.housing = Table("housing")

    def setUp(self):
        self.widget = self.create_widget(OWViolinPlot)

    def _select_data(self):
        self.widget.graph._update_selection(QPointF(0, 5), QPointF(0, 6), True)
        assert len(self.widget.selection) == 30
        return self.widget.selection

    def test_kernels(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        kernel_combo = self.widget.controls.kernel_index
        for kernel in self.widget.KERNEL_LABELS[1:]:
            simulate.combobox_activate_item(kernel_combo, kernel)

    def test_no_cont_features(self):
        data = Table("zoo")
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_cont_features.is_shown())

    def test_not_enough_instances(self):
        self.send_signal(self.widget.Inputs.data, self.data[:1])
        self.assertTrue(self.widget.Error.not_enough_instances.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

    def test_controls(self):
        self.widget.controls.show_strip_plot.setChecked(True)
        self.widget.controls.show_rug_plot.setChecked(True)
        self.widget.controls.order_violins.setChecked(True)
        self.widget.controls.orientation_index.buttons[0].click()
        self.widget.controls.kernel_index.setCurrentIndex(1)
        self.widget.controls.scale_index.setCurrentIndex(1)

        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.controls.show_box_plot.setChecked(False)
        self.widget.controls.show_strip_plot.setChecked(False)
        self.widget.controls.show_rug_plot.setChecked(False)
        self.widget.controls.order_violins.setChecked(False)
        self.widget.controls.orientation_index.buttons[1].click()
        self.widget.controls.kernel_index.setCurrentIndex(0)
        self.widget.controls.scale_index.setCurrentIndex(2)

        self.send_signal(self.widget.Inputs.data, None)
        self.widget.controls.show_box_plot.setChecked(True)
        self.widget.controls.show_strip_plot.setChecked(True)
        self.widget.controls.show_rug_plot.setChecked(True)
        self.widget.controls.order_violins.setChecked(True)
        self.widget.controls.orientation_index.buttons[0].click()
        self.widget.controls.kernel_index.setCurrentIndex(1)
        self.widget.controls.scale_index.setCurrentIndex(1)

    def test_enable_controls(self):
        self.assertTrue(self.widget.controls.order_violins.isEnabled())
        self.assertTrue(self.widget.controls.scale_index.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertFalse(self.widget.controls.order_violins.isEnabled())
        self.assertFalse(self.widget.controls.scale_index.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTrue(self.widget.controls.order_violins.isEnabled())
        self.assertTrue(self.widget.controls.scale_index.isEnabled())

        self.__select_value(self.widget._group_var_view, "None")
        self.assertFalse(self.widget.controls.order_violins.isEnabled())
        self.assertFalse(self.widget.controls.scale_index.isEnabled())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertTrue(self.widget.controls.order_violins.isEnabled())
        self.assertTrue(self.widget.controls.scale_index.isEnabled())

    def test_datasets(self):
        self.widget.controls.show_strip_plot.setChecked(True)
        self.widget.controls.show_rug_plot.setChecked(True)
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)
            for i in range(3):
                cb = self.widget.controls.scale_index
                simulate.combobox_activate_index(cb, i)

    def test_unique_values(self):
        self.send_signal(self.widget.Inputs.data, self.data[:5])
        self.__select_value(self.widget._value_var_view, "petal width")

    def test_paint(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.controls.show_strip_plot.setChecked(True)
        self.widget.controls.show_rug_plot.setChecked(True)

        painter = Mock()
        painter.save = Mock()
        painter.drawLine = Mock()
        painter.drawPath = Mock()
        painter.drawRect = Mock()
        painter.restore = Mock()

        item = self.widget.graph._ViolinPlot__violin_items[0]
        item.paint(painter, Mock())
        painter.drawPath.assert_called_once()

        painter.drawLine.reset_mock()
        item = self.widget.graph._ViolinPlot__box_items[0]
        item.paint(painter, Mock(), None)
        self.assertEqual(painter.drawLine.call_count, 2)

        self._select_data()
        item = self.widget.graph._ViolinPlot__selection_rects[0]
        item.paint(painter, Mock())
        painter.drawRect.assert_called_once()

        self.widget.controls.orientation_index.buttons[0].click()

        painter.drawPath.reset_mock()
        item = self.widget.graph._ViolinPlot__violin_items[0]
        item.paint(painter, Mock())
        painter.drawPath.assert_called_once()

        painter.drawLine.reset_mock()
        item = self.widget.graph._ViolinPlot__box_items[0]
        item.paint(painter, Mock(), None)
        self.assertEqual(painter.drawLine.call_count, 2)

        painter.drawRect.reset_mock()
        item = self.widget.graph._ViolinPlot__selection_rects[0]
        item.paint(painter, Mock())
        painter.drawRect.assert_called_once()

        self.assertEqual(painter.save.call_count, 6)
        self.assertEqual(painter.restore.call_count, 6)

    @patch.object(ViolinPlotViewBox, "mapToView")
    def test_select(self, mocked_mapToView: Mock):
        mocked_mapToView.side_effect = lambda x: x

        view_box: ViewBox = self.widget.graph.getViewBox()

        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = QPointF(0, 5)
        event.pos.return_value = QPointF(0, 6)
        event.isFinish.return_value = True

        view_box.mouseDragEvent(event)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected)

        self.send_signal(self.widget.Inputs.data, self.data)
        view_box.mouseDragEvent(event)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 30)

        view_box.mouseDragEvent(event, 1)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 30)

        view_box.mouseClickEvent(Mock())
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected)

        view_box.mouseDragEvent(event)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 30)

    def test_set_selection_not_data(self):
        self.widget.graph.set_selection([1, 2])

    @patch.object(ViolinPlotViewBox, "mapToView")
    def test_selection_rect(self, mocked_mapToView: Mock):
        mocked_mapToView.side_effect = lambda x: x

        view_box: ViewBox = self.widget.graph.getViewBox()

        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = QPointF(0, 5)
        event.pos.return_value = QPointF(0, 6)
        event.isFinish.return_value = True

        self.send_signal(self.widget.Inputs.data, self.data)
        view_box.mouseDragEvent(event)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 30)

        sel_rect = self.widget.graph._ViolinPlot__selection_rects[0]
        self.assertEqual(sel_rect.selection_rect.height(), 1)

        self.widget.controls.orientation_index.buttons[0].click()
        sel_rect = self.widget.graph._ViolinPlot__selection_rects[0]
        self.assertEqual(sel_rect.selection_rect.width(), 1)

    def test_selection_no_data(self):
        self.widget.graph._update_selection(QPointF(0, 5), QPointF(0, 6), 1)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNone(selected)

    def test_selection_no_group(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.widget.graph._update_selection(QPointF(0, 30), QPointF(0, 40), 1)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 53)

    def test_selection_sort_violins(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__select_value(self.widget._value_var_view, "sepal width")
        self.widget.controls.show_strip_plot.setChecked(True)

        self.widget.graph._update_selection(QPointF(0, 4), QPointF(0, 5), 1)
        selected1 = self.get_output(self.widget.Outputs.selected_data)

        self.widget.controls.order_violins.setChecked(True)
        selected2 = self.get_output(self.widget.Outputs.selected_data)

        self.assert_table_equal(selected1, selected2)

    def test_selection_orientation(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.widget.graph._update_selection(QPointF(0, 30), QPointF(0, 40), 1)
        self.widget.controls.orientation_index.buttons[0].click()
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 53)

        self.widget.graph._update_selection(QPointF(30, 0), QPointF(40, 0), 1)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected), 53)

    def test_saved_selection(self):
        graph = self.widget.graph
        self.send_signal(self.widget.Inputs.data, self.data)
        graph._update_selection(QPointF(0, 6), QPointF(0, 5.5), 1)
        selected1 = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected1), 5)

        with patch("AnyQt.QtWidgets.QApplication.keyboardModifiers",
                   lambda: Qt.ShiftModifier):
            graph._update_selection(QPointF(6, 6), QPointF(6, 5.5), 1)
        selected2 = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(selected2), 13)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWViolinPlot, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.data, widget=widget)
        selected3 = self.get_output(widget.Outputs.selected_data,
                                    widget=widget)
        self.assert_table_equal(selected2, selected3)

    def test_visual_settings(self):
        graph = self.widget.graph

        def test_settings():
            font = QFont("Helvetica", italic=True, pointSize=20)
            self.assertFontEqual(
                graph.parameter_setter.title_item.item.font(), font
            )

            font.setPointSize(16)
            for item in graph.parameter_setter.axis_items:
                self.assertFontEqual(item.label.font(), font)

            font.setPointSize(15)
            for item in graph.parameter_setter.axis_items:
                self.assertFontEqual(item.style["tickFont"], font)

            self.assertEqual(
                graph.parameter_setter.title_item.item.toPlainText(), "Foo"
            )
            self.assertEqual(graph.parameter_setter.title_item.text, "Foo")

            self.assertTrue(
                graph.parameter_setter.bottom_axis.style["rotateTicks"]
            )

        self.send_signal(self.widget.Inputs.data, self.data)
        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Title", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis title", "Font size"), 16
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis ticks", "Font size"), 15
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis ticks", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Annotations", "Title", "Title"), "Foo"
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Bottom axis", "Vertical tick text"), True
        self.widget.set_visual_settings(key, value)

        self.send_signal(self.widget.Inputs.data, self.data)
        test_settings()

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.data, self.data)
        test_settings()

    def assertFontEqual(self, font1, font2):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())

    @staticmethod
    def __select_value(list_, value):
        model = list_.model()
        for i in range(model.rowCount()):
            idx = model.index(i, 0)
            if model.data(idx) == value:
                list_.selectionModel().setCurrentIndex(
                    idx, QItemSelectionModel.ClearAndSelect)


if __name__ == "__main__":
    unittest.main()
