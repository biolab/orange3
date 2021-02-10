# pylint: disable=protected-access
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import seaborn as sns

from AnyQt.QtCore import QItemSelectionModel, QPointF, Qt
from AnyQt.QtGui import QFont

from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame
from Orange.widgets.tests.base import datasets, simulate, \
    WidgetOutputsTestMixin, WidgetTest
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.visualize.owviolinplot import OWViolinPlot


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

    def test_summary(self):
        info = self.widget.info
        no_input, no_output = "No data on input", "No data on output"

        self.send_signal(self.widget.Inputs.data, self.data)
        details = format_summary_details(self.data)
        self.assertEqual(info._StateInfo__input_summary.brief, "150")
        self.assertEqual(info._StateInfo__input_summary.details, details)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

        self._select_data()
        output = self.get_output(self.widget.Outputs.selected_data)
        details = format_summary_details(output)
        self.assertEqual(info._StateInfo__output_summary.brief, "30")
        self.assertEqual(info._StateInfo__output_summary.details, details)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

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

        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.controls.show_box_plot.setChecked(False)
        self.widget.controls.show_strip_plot.setChecked(False)
        self.widget.controls.show_rug_plot.setChecked(False)
        self.widget.controls.order_violins.setChecked(False)
        self.widget.controls.orientation_index.buttons[1].click()
        self.widget.controls.kernel_index.setCurrentIndex(0)

        self.send_signal(self.widget.Inputs.data, None)
        self.widget.controls.show_box_plot.setChecked(True)
        self.widget.controls.show_strip_plot.setChecked(True)
        self.widget.controls.show_rug_plot.setChecked(True)
        self.widget.controls.order_violins.setChecked(True)
        self.widget.controls.orientation_index.buttons[0].click()
        self.widget.controls.kernel_index.setCurrentIndex(1)

    def test_datasets(self):
        self.widget.controls.show_strip_plot.setChecked(True)
        self.widget.controls.show_rug_plot.setChecked(True)
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)

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

    def test_seaborn(self):
        # inner{“box”, “quartile”, “point”, “stick”, None}, optional
        self.assertEqual(True, False)
        data = Table("heart_disease")
        print(data.domain)
        df = table_to_frame(data)
        x = df["diameter narrowing"]
        y = df["ST by exercise"]
        y = df["major vessels colored"]

        data = Table("iris")
        print(data.domain)
        df = table_to_frame(data)
        x = df["iris"]
        y = df["sepal length"]
        # hue = df["chest pain"]
        print(y.min(), y.max())
        sns.violinplot(
            x=x,
            y=y,
            # inner="stick",
            # orient="h",
            #   hue=hue,
            scale="count",
            # data=df,
            # split=True,
        )
        plt.show()

        sns.kdeplot()


if __name__ == "__main__":
    unittest.main()
