# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access,unsubscriptable-object
import unittest
from unittest.mock import patch, Mock

import numpy as np
from AnyQt.QtCore import Qt, QSize, QRect
from AnyQt.QtGui import QBrush

from Orange.data import Table, ContinuousVariable, DiscreteVariable, Domain
from Orange.util import color_to_hex
from Orange.widgets.utils import colorpalettes
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.data import owcolor
from Orange.widgets.data.owcolor import ColorRole
from Orange.widgets.tests.base import WidgetTest
from orangewidget.tests.base import GuiTest


class AttrDescTest(unittest.TestCase):
    def test_name(self):
        x = ContinuousVariable("x")
        desc = owcolor.AttrDesc(x)
        self.assertEqual(desc.name, "x")
        desc.name = "y"
        self.assertEqual(desc.name, "y")
        desc.name = None
        self.assertEqual(desc.name, "x")


class DiscAttrTest(unittest.TestCase):
    def setUp(self):
        x = DiscreteVariable("x", ["a", "b", "c"])
        self.desc = owcolor.DiscAttrDesc(x)

    def test_colors(self):
        desc = self.desc
        colors = desc.colors.copy()
        desc.set_color(2, (0, 0, 0))
        colors[2] = 0
        np.testing.assert_equal(desc.colors, colors)

    def test_values(self):
        desc = self.desc
        self.assertEqual(desc.values, ("a", "b", "c"))
        desc.set_value(1, "d")
        self.assertEqual(desc.values, ("a", "d", "c"))

    def test_create_variable(self):
        desc = self.desc
        desc.set_color(0, [1, 2, 3])
        desc.set_color(1, [4, 5, 6])
        desc.set_color(2, [7, 8, 9])
        desc.name = "z"
        desc.set_value(1, "d")
        var = desc.create_variable()
        self.assertIsInstance(var, DiscreteVariable)
        self.assertEqual(var.name, "z")
        self.assertEqual(var.values, ("a", "d", "c"))
        np.testing.assert_equal(var.colors, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        palette = desc.var.attributes["palette"] = object()
        var = desc.create_variable()
        self.assertIs(desc.var.attributes["palette"], palette)
        self.assertFalse(hasattr(var.attributes, "palette"))


class ContAttrDesc(unittest.TestCase):
    def setUp(self):
        x = ContinuousVariable("x")
        self.desc = owcolor.ContAttrDesc(x)

    def test_palette(self):
        desc = self.desc
        palette = desc.palette_name
        self.assertIsInstance(palette, str)
        desc.palette_name = "foo"
        self.assertEqual(desc.palette_name, "foo")
        desc.palette_name = None
        self.assertEqual(desc.palette_name, palette)

    def test_create_variable(self):
        desc = self.desc
        desc.name = "z"
        palette_name = _find_other_palette(
            colorpalettes.ContinuousPalettes[desc.palette_name]).name
        desc.palette_name = palette_name
        var = desc.create_variable()
        self.assertIsInstance(var, ContinuousVariable)
        self.assertEqual(var.name, "z")
        self.assertEqual(var.palette.name, palette_name)

        colors = desc.var.attributes["colors"] = object()
        var = desc.create_variable()
        self.assertIs(desc.var.attributes["colors"], colors)
        self.assertFalse(hasattr(var.attributes, "colors"))


class BaseTestColorTableModel:
    def test_row_count(self):
        model = self.model

        self.assertEqual(model.rowCount(), 0)
        model.set_data(self.descs)
        self.assertEqual(model.rowCount(), len(self.descs))
        self.assertEqual(model.rowCount(self.model.index(0, 0)), 0)

    def test_data(self):
        self.model.set_data(self.descs)
        data = self.model.data

        index = self.model.index(1, 0)
        self.assertEqual(data(index, Qt.DisplayRole), self.descs[1].name)
        self.assertEqual(data(index, Qt.EditRole), self.descs[1].name)
        self.assertTrue(data(index, Qt.FontRole).bold())
        self.assertTrue(data(index, Qt.TextAlignmentRole) & Qt.AlignRight)

        self.descs[1].name = "bar"
        self.assertEqual(data(index), "bar")

        index = self.model.index(2, 0)
        self.assertEqual(data(index, Qt.DisplayRole), self.descs[2].name)


    def test_set_data(self):
        emit = Mock()
        try:
            self.model.dataChanged.connect(emit)
            self.model.set_data(self.descs)
            data = self.model.data
            setData = self.model.setData

            index = self.model.index(1, 0)
            assert self.descs[1].name != "foo"
            self.assertFalse(setData(index, "foo", Qt.DisplayRole))
            emit.assert_not_called()
            self.assertEqual(data(index, Qt.DisplayRole), self.descs[1].name)
            self.assertTrue(setData(index, "foo", Qt.EditRole))
            emit.assert_called()
            self.assertEqual(data(index, Qt.DisplayRole), "foo")
            self.assertEqual(self.descs[1].name, "foo")
        finally:
            self.model.dataChanged.disconnect(emit)


class TestDiscColorTableModel(GuiTest, BaseTestColorTableModel):
    def setUp(self):
        x = DiscreteVariable("x", list("abc"))
        y = DiscreteVariable("y", list("def"))
        z = DiscreteVariable("z", list("ghijk"))
        self.descs = [owcolor.DiscAttrDesc(v) for v in (x, y, z)]
        self.model = owcolor.DiscColorTableModel()

    def test_column_count(self):
        model = self.model

        self.assertEqual(model.columnCount(), 1)
        model.set_data(self.descs[:2])
        self.assertEqual(model.columnCount(), 4)
        model.set_data(self.descs)
        self.assertEqual(model.columnCount(), 6)

        self.assertEqual(model.columnCount(model.index(0, 0)), 0)

    def test_data(self):
        super().test_data()

        model = self.model

        self.assertIsNone(model.data(model.index(0, 4)))

        index = model.index(1, 2)
        self.assertEqual(model.data(index, Qt.DisplayRole), "e")
        self.assertEqual(model.data(index, Qt.EditRole), "e")
        font = model.data(index, Qt.FontRole)
        self.assertTrue(font is None or not font.bold())

        var_colors = self.descs[1].var.colors[1]
        color = model.data(index, Qt.DecorationRole)
        np.testing.assert_equal(color.getRgb()[:3], var_colors)

        color = model.data(index, owcolor.ColorRole)
        np.testing.assert_equal(color, var_colors)

        self.assertEqual(
            model.data(index, Qt.ToolTipRole), color_to_hex(var_colors))

        self.assertIsNone(model.data(model.index(0, 4)))

        index = model.index(2, 5)
        self.assertEqual(model.data(index, Qt.DisplayRole), "k")
        self.assertEqual(model.data(index, Qt.EditRole), "k")
        font = model.data(index, Qt.FontRole)
        self.assertTrue(font is None or not font.bold())

        var_colors = self.descs[2].var.colors[4]
        color = model.data(index, Qt.DecorationRole)
        np.testing.assert_equal(color.getRgb()[:3], var_colors)

        color = model.data(index, owcolor.ColorRole)
        np.testing.assert_equal(color, var_colors)

        self.assertEqual(
            model.data(index, Qt.ToolTipRole), color_to_hex(var_colors))

        self.descs[2].set_value(4, "foo")
        self.assertEqual(model.data(index, Qt.DisplayRole), "foo")

    def test_set_data(self):
        super().test_set_data()

        model = self.model
        emit = Mock()
        try:
            model.dataChanged.connect(emit)

            index = model.index(2, 5)

            self.assertEqual(model.data(index, Qt.DisplayRole), "k")
            self.assertEqual(model.data(index, Qt.EditRole), "k")
            self.assertFalse(model.setData(index, "foo", Qt.DisplayRole))
            emit.assert_not_called()
            self.assertEqual(model.data(index, Qt.DisplayRole), "k")
            self.assertTrue(model.setData(index, "foo", Qt.EditRole))
            emit.assert_called()
            emit.reset_mock()
            self.assertEqual(model.data(index, Qt.DisplayRole), "foo")
            self.assertEqual(self.descs[2].values, ("g", "h", "i", "j", "foo"))

            new_color = [0, 1, 2]
            self.assertTrue(model.setData(index, new_color + [255], ColorRole))
            emit.assert_called()
            emit.reset_mock()
            color = model.data(index, Qt.DecorationRole)
            rgb = [color.red(), color.green(), color.blue()]
            self.assertEqual(rgb, new_color)

            color = model.data(index, owcolor.ColorRole)
            self.assertEqual(list(color), new_color)

            self.assertEqual(
                model.data(index, Qt.ToolTipRole), color_to_hex(new_color))

            np.testing.assert_equal(self.descs[2].colors[4], rgb)
        finally:
            model.dataChanged.disconnect(emit)


def _find_other_palette(initial):
    for palette in colorpalettes.ContinuousPalettes.values():
        if palette.name != initial.name:
            return palette
    return None  # pragma: no cover


class TestContColorTableModel(GuiTest, BaseTestColorTableModel):
    def setUp(self):
        z = ContinuousVariable("z")
        w = ContinuousVariable("w")
        u = ContinuousVariable("u")
        self.descs = [owcolor.ContAttrDesc(v) for v in (z, w, u)]
        self.model = owcolor.ContColorTableModel()

    def test_column_count(self):
        model = self.model

        model.set_data(self.descs)
        self.assertEqual(model.columnCount(), 3)
        self.assertEqual(model.columnCount(model.index(0, 0)), 0)

    def test_data(self):
        super().test_data()

        model = self.model
        index = model.index(1, 1)
        palette = colorpalettes.ContinuousPalettes[self.descs[1].palette_name]
        self.assertEqual(model.data(index, Qt.ToolTipRole),
                         palette.friendly_name)
        self.assertEqual(model.data(index, ColorRole), palette)
        with patch.object(palette, "color_strip") as color_strip:
            strip = model.data(index, owcolor.StripRole)
            self.assertIs(strip, color_strip.return_value)
            color_strip.assert_called_with(128, 16)
        self.assertIsInstance(model.data(index, Qt.SizeHintRole), QSize)
        self.assertIsNone(model.data(index, Qt.FontRole))

        palette = _find_other_palette(self.descs[1])
        self.descs[1].palette_name = palette.name
        self.assertIs(model.data(index, ColorRole), palette)

        index = self.model.index(1, 2)
        self.assertIsNone(model.data(index, Qt.ToolTipRole))
        self.assertIsInstance(model.data(index, Qt.SizeHintRole), QSize)
        self.assertIsInstance(model.data(index, Qt.ForegroundRole), QBrush)
        self.assertIsNone(model.data(index, Qt.DisplayRole))
        model.set_mouse_row(0)
        self.assertIsNone(model.data(index, Qt.DisplayRole))
        model.set_mouse_row(1)
        self.assertEqual(model.data(index, Qt.DisplayRole), "Copy to all")

    def test_set_data(self):
        super().test_set_data()

        model = self.model
        index = model.index(1, 1)
        index2 = model.index(2, 1)
        initial = model.data(index, ColorRole)
        initial2 = model.data(index, ColorRole)
        assert initial.name == initial2.name
        palette = _find_other_palette(initial)

        emit = Mock()
        try:
            model.dataChanged.connect(emit)

            self.assertFalse(model.setData(index, None, Qt.DisplayRole))
            emit.assert_not_called()

            self.assertTrue(model.setData(index, palette, ColorRole))
            emit.assert_called()
            self.assertIs(model.data(index2, ColorRole), initial2)

            self.assertEqual(model.data(index, Qt.ToolTipRole),
                             palette.friendly_name)
            self.assertEqual(model.data(index, ColorRole), palette)
            self.assertEqual(self.descs[1].palette_name, palette.name)
            with patch.object(palette, "color_strip") as color_strip:
                strip = model.data(index, owcolor.StripRole)
                self.assertIs(strip, color_strip.return_value)
                color_strip.assert_called_with(128, 16)
        finally:
            model.dataChanged.disconnect(emit)

    def test_copy_to_all(self):
        super().test_set_data()

        model = self.model
        index = model.index(1, 1)
        initial = model.data(index, ColorRole)
        palette = _find_other_palette(initial)

        emit = Mock()
        try:
            model.dataChanged.connect(emit)
            model.setData(index, palette, ColorRole)
            emit.assert_called()
            emit.reset_mock()

            model.copy_to_all(index)
            emit.assert_called_once()
            for row, desc in enumerate(self.descs):
                self.assertEqual(
                    model.data(model.index(row, 1), ColorRole).name,
                    palette.name)
                self.assertEqual(desc.palette_name, palette.name)
        finally:
            model.dataChanged.disconnect(emit)


class TestColorStripDelegate(GuiTest):
    def setUp(self):
        z = ContinuousVariable("z")
        w = ContinuousVariable("w")
        u = ContinuousVariable("u")
        self.descs = [owcolor.ContAttrDesc(v) for v in (z, w, u)]
        self.model = owcolor.ContColorTableModel()
        self.model.set_data(self.descs)
        self.table = owcolor.ContinuousTable(self.model)

    def test_color_combo(self):
        model = self.model
        index = model.index(1, 1)
        initial = model.data(index, ColorRole)
        palette = _find_other_palette(initial)
        model.setData(index, palette, ColorRole)
        self.assertEqual(self.descs[1].palette_name, palette.name)

        combo = self.table.color_delegate.createEditor(None, Mock(), index)
        self.assertEqual(combo.currentText(), palette.friendly_name)
        palette = _find_other_palette(palette)
        combo.setCurrentIndex(combo.findText(palette.friendly_name))
        self.assertEqual(self.descs[1].palette_name, palette.name)

        with patch.object(self.table, "closeEditor") as closeEditor:
            combo.hidePopup()
            closeEditor.assert_called()

    @patch.object(owcolor.HorizontalGridDelegate, "paint")
    def test_paint(self, _):
        model = self.model
        index = model.index(1, 1)
        painter = Mock()
        option = Mock()
        option.rect = QRect(10, 20, 30, 40)
        index.data = Mock()
        index.data.return_value = Mock()
        index.data.return_value.height = Mock(return_value=42)
        self.table.color_delegate.paint(painter, option, index)
        self.assertIs(painter.drawPixmap.call_args[0][2],
                      index.data.return_value)


class TestOWColor(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(owcolor.OWColor)
        self.iris = Table("iris")

    def test_reuse_old_settings(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        assert isinstance(self.widget, owcolor.OWColor)
        self.widget.saveSettings()

        w = self.create_widget(owcolor.OWColor, reset_default_settings=False)
        self.send_signal(self.widget.Inputs.data, self.iris, widget=w)

    def test_invalid_input_colors(self):
        a = ContinuousVariable("a")
        a.attributes["colors"] = "invalid"
        t = Table.from_domain(Domain([a]))

        self.send_signal(self.widget.Inputs.data, t)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget, 'unconditional_commit') as commit:
            self.widget.auto_apply = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.iris)
            commit.assert_called()

    def test_commit_on_data_changed(self):
        widget = self.widget
        model = widget.cont_model
        self.send_signal(widget.Inputs.data, self.iris)
        with patch.object(widget, 'commit') as commit:
            commit.reset_mock()
            model.setData(model.index(0, 0), "y", Qt.EditRole)
            commit.assert_called()

    def test_lose_data(self):
        widget = self.widget
        send = widget.Outputs.data.send = Mock()

        self.send_signal(widget.Inputs.data, self.iris)
        send.assert_called_once()
        self.assertEqual(widget.disc_model.rowCount(), 1)
        self.assertEqual(widget.cont_model.rowCount(), 4)

        send.reset_mock()
        self.send_signal(widget.Inputs.data, None)
        send.assert_called_with(None)
        self.assertEqual(widget.disc_model.rowCount(), 0)
        self.assertEqual(widget.cont_model.rowCount(), 0)

    def test_model_content(self):
        widget = self.widget
        data = Table("heart_disease")
        self.send_signal(widget.Inputs.data, data)

        dm = widget.disc_model
        self.assertEqual(
            [dm.data(dm.index(i, 0)) for i in range(dm.rowCount())],
            [var.name for var in data.domain.variables if var.is_discrete]
        )

        cm = widget.disc_model
        self.assertEqual(
            [dm.data(cm.index(i, 0)) for i in range(cm.rowCount())],
            [var.name for var in data.domain.variables if var.is_discrete]
        )

    def test_report(self):
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.data, Table("zoo"))
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()

    def test_string_variables(self):
        self.send_signal(self.widget.Inputs.data, Table("zoo"))

    def test_summary(self):
        """Check if the status bar is updated when data is received"""
        data = self.iris
        input_sum = self.widget.info.set_input_summary = Mock()
        output_sum = self.widget.info.set_output_summary = Mock()

        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(len(data), format_summary_details(data))
        output = self.get_output(self.widget.Outputs.data)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))
        input_sum.reset_mock()
        output_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_once()
        self.assertEqual(input_sum.call_args[0][0].brief, "")
        output_sum.assert_called_once()
        self.assertEqual(output_sum.call_args[0][0].brief, "")


if __name__ == "__main__":
    unittest.main()
