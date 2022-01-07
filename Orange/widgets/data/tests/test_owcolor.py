# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access,unsubscriptable-object
import json
import os
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
from AnyQt.QtCore import Qt, QSize, QRect
from AnyQt.QtGui import QBrush

from orangewidget.tests.base import GuiTest

from Orange.data import Table, ContinuousVariable, DiscreteVariable, Domain
from Orange.preprocess.transformation import Identity
from Orange.util import color_to_hex
from Orange.widgets.utils import colorpalettes
from Orange.widgets.data import owcolor
from Orange.widgets.data.owcolor import ColorRole
from Orange.widgets.tests.base import WidgetTest


class AttrDescTest(unittest.TestCase):
    def test_name(self):
        x = ContinuousVariable("x")
        desc = owcolor.AttrDesc(x)
        self.assertEqual(desc.name, "x")
        desc.name = "y"
        self.assertEqual(desc.name, "y")
        desc.name = None
        self.assertEqual(desc.name, "x")

    def test_reset(self):
        x = ContinuousVariable("x")
        desc = owcolor.AttrDesc(x)
        desc.reset()
        self.assertEqual(desc.name, "x")
        desc.name = "y"
        desc.reset()
        self.assertEqual(desc.name, "x")

    def test_to_dict(self):
        x = ContinuousVariable("x")
        desc = owcolor.AttrDesc(x)
        self.assertEqual(desc.to_dict(), {})
        desc2, warns = owcolor.AttrDesc.from_dict(x, desc.to_dict())
        self.assertEqual(warns, [])
        self.assertIsNone(desc2.new_name)

        desc.name = "y"
        self.assertEqual(desc.to_dict(), {"rename": "y"})

        desc2, warns = owcolor.AttrDesc.from_dict(x, desc.to_dict())
        self.assertEqual(warns, [])
        self.assertEqual(desc2.new_name, "y")

        self.assertRaises(owcolor.InvalidFileFormat,
                          owcolor.AttrDesc.from_dict, x, {"rename": 42})
        self.assertRaises(owcolor.InvalidFileFormat,
                          owcolor.AttrDesc.from_dict, x, [])

        # Additional keys shouldn't cause exceptions
        owcolor.AttrDesc.from_dict(x, {"foo": 42})


class DiscAttrTest(unittest.TestCase):
    def setUp(self):
        self.var = DiscreteVariable("x", ["a", "b", "c"])
        self.desc = owcolor.DiscAttrDesc(self.var)

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
        self.assertIsInstance(var.compute_value, Identity)
        self.assertIs(var.compute_value.variable, desc.var)

        palette = desc.var.attributes["palette"] = object()
        var = desc.create_variable()
        self.assertIs(desc.var.attributes["palette"], palette)
        self.assertFalse(hasattr(var.attributes, "palette"))
        self.assertIsInstance(var.compute_value, Identity)
        self.assertIs(var.compute_value.variable, desc.var)

    def test_reset(self):
        desc = self.desc
        desc.set_color(0, [1, 2, 3])
        desc.set_color(1, [4, 5, 6])
        desc.set_color(2, [7, 8, 9])
        desc.set_value(1, "d")
        desc.reset()
        np.testing.assert_equal(desc.colors, self.var.colors)
        self.assertEqual(desc.values, self.var.values)

    def test_to_dict(self):
        desc = owcolor.DiscAttrDesc(self.var)
        self.assertEqual(desc.to_dict(), {})
        desc2, warns = owcolor.DiscAttrDesc.from_dict(self.var, desc.to_dict())
        self.assertEqual(warns, [])
        self.assertIsNone(desc2.new_name)
        self.assertIsNone(desc2.new_values)
        self.assertIsNone(desc2.new_colors)

        desc.name = "y"
        self.assertEqual(desc.to_dict(), {"rename": "y"})

        desc2, warns = owcolor.DiscAttrDesc.from_dict(self.var, desc.to_dict())
        self.assertEqual(warns, [])
        self.assertEqual(desc2.new_name, "y")
        self.assertIsNone(desc2.new_values)
        self.assertIsNone(desc2.new_colors)

        desc.set_value(1, "b2")
        desc.set_color(1, [1, 2, 3])
        desc.set_color(2, [2, 3, 4])
        self.assertEqual(desc.to_dict(), {
            "rename": "y",
            "renamed_values": {"b": "b2"},
            "colors": {"a": color_to_hex(desc.colors[0]),
                       "b": "#010203",
                       "c": "#020304"}
        })

        desc2, warns = owcolor.DiscAttrDesc.from_dict(self.var, desc.to_dict())
        self.assertEqual(warns, [])
        cols = list(desc.colors)
        cols[1:] = [[1, 2, 3], [2, 3, 4]]
        np.testing.assert_equal(desc2.colors, cols)
        self.assertEqual(desc2.values, ("a", "b2", "c"))

        desc2, warns = owcolor.DiscAttrDesc.from_dict(
            self.var,
            {"rename": "y",
             "renamed_values": {"b": "b2", "d": "x"},  # d is redundant
             "colors": {"b": "#010203",
                        "c": "#020304",
                        "d": "#123456"}  # d is redundant and must be ignored
            })
        self.assertEqual(warns, [])
        cols = list(desc.colors)
        cols[1:] = [[1, 2, 3], [2, 3, 4]]
        np.testing.assert_equal(desc2.colors, cols)
        self.assertEqual(desc2.values, ("a", "b2", "c"))

    def test_from_dict_coliding_values(self):
        desc2, warns = owcolor.DiscAttrDesc.from_dict(
            self.var, {"renamed_values": {"a": "b"}})
        self.assertEqual(len(warns), 1)
        self.assertTrue("duplicate names" in warns[0])
        self.assertIsNone(desc2.new_values)
        self.assertEqual(desc2.values, ("a", "b", "c"))

        desc2, warns = owcolor.DiscAttrDesc.from_dict(
            self.var, {"renamed_values": {"a": "e", "b": "e"}})
        self.assertEqual(len(warns), 1)
        self.assertTrue("duplicate names" in warns[0])
        self.assertIsNone(desc2.new_values)
        self.assertEqual(desc2.values, ("a", "b", "c"))

        desc2, warns = owcolor.DiscAttrDesc.from_dict(
            self.var, {"renamed_values": {"a": "b", "b": "a"}})
        self.assertEqual(warns, [])
        self.assertEqual(desc2.values, ("b", "a", "c"))

    def test_from_dict_exceptions(self):
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.AttrDesc.from_dict, self.var, [])
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"rename": 42}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"colors": []}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"colors": {"a": 42}}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"colors": {4: "#000000"}}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"colors": {"a": "#00"}}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"colors": {"a": "#qwerty"}}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"renamed_values": []}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"renamed_values": {"a": 42}}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.DiscAttrDesc.from_dict, self.var, {"renamed_values": {4: "#000000"}}
        )


class ContAttrDescTest(unittest.TestCase):
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
        self.assertIsInstance(var.compute_value, Identity)
        self.assertIs(var.compute_value.variable, desc.var)

        colors = desc.var.attributes["colors"] = object()
        var = desc.create_variable()
        self.assertIs(desc.var.attributes["colors"], colors)
        self.assertFalse(hasattr(var.attributes, "colors"))
        self.assertIsInstance(var.compute_value, Identity)
        self.assertIs(var.compute_value.variable, desc.var)

    def test_reset(self):
        desc = self.desc
        palette_name = desc.palette_name
        desc.palette_name = _find_other_palette(
            colorpalettes.ContinuousPalettes[palette_name]).name
        desc.reset()
        np.testing.assert_equal(desc.palette_name, palette_name)

    def test_to_dict(self):
        x = ContinuousVariable("x")
        desc = owcolor.ContAttrDesc(x)
        self.assertEqual(desc.to_dict(), {})
        desc2, warns = owcolor.ContAttrDesc.from_dict(x, desc.to_dict())
        self.assertEqual(warns, [])
        self.assertIsNone(desc2.new_name)
        self.assertIsNone(desc2.new_palette_name)

        desc.name = "y"
        self.assertEqual(desc.to_dict(), {"rename": "y"})

        desc2, warns = owcolor.ContAttrDesc.from_dict(x, desc.to_dict())
        self.assertEqual(warns, [])
        self.assertEqual(desc2.new_name, "y")
        self.assertIsNone(desc2.new_palette_name)

        desc = owcolor.ContAttrDesc(x)
        desc.palette_name = "linear_viridis"
        self.assertEqual(desc.to_dict(), {"colors": "linear_viridis"})

    def test_from_dict_exceptions(self):
        x = ContinuousVariable("x")
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.ContAttrDesc.from_dict, x, []
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.ContAttrDesc.from_dict, x, {"colors": 42}
        )
        self.assertRaises(
            owcolor.InvalidFileFormat,
            owcolor.ContAttrDesc.from_dict, x, {"colors": "no such palette"}
        )


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

    def test_reset(self):
        for desc in self.descs:
            desc.reset = Mock()
        self.model.set_data(self.descs)
        self.model.reset()
        for desc in self.descs:
            desc.reset.assert_called()


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
        # Handling of invalid value in deprecated 'colors': warn, but don't fail
        with self.assertWarns(DeprecationWarning):
            a = ContinuousVariable("a")
            a.attributes["colors"] = "invalid"
            t = Table.from_domain(Domain([a]))
            self.send_signal(self.widget.Inputs.data, t)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as commit:
            self.widget.auto_apply = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.iris)
            commit.assert_called()

    def test_commit_on_data_changed(self):
        widget = self.widget
        model = widget.cont_model
        self.send_signal(widget.Inputs.data, self.iris)
        with patch.object(widget.commit, 'deferred') as commit:
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

    def test_reset(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        cont_model = self.widget.cont_model
        disc_model = self.widget.disc_model
        cont_model.setData(cont_model.index(0, 0), "a", Qt.EditRole)
        disc_model.setData(disc_model.index(0, 0), "b", Qt.EditRole)
        outp = self.get_output(self.widget.Outputs.data)
        self.assertEqual(outp.domain[0].name, "a")
        self.assertEqual(outp.domain.class_var.name, "b")
        self.widget.reset()
        outp = self.get_output(self.widget.Outputs.data)
        self.assertEqual(
            outp.domain[0].name, self.iris.domain[0].name)
        self.assertEqual(
            outp.domain.class_var.name, self.iris.domain.class_var.name)

    def test_save(self):
        self.widget._save_var_defs = Mock()
        with patch.object(owcolor.QFileDialog, "getSaveFileName",
                          return_value=("", "")):
            self.widget.save()
            self.widget._save_var_defs.assert_not_called()
        with patch.object(owcolor.QFileDialog, "getSaveFileName",
                          return_value=("foo", "bar")):
            self.widget.save()
            self.widget._save_var_defs.assert_called_with("foo")

    def test_save_low(self):
        descA = owcolor.DiscAttrDesc(
            DiscreteVariable("varA", values=tuple("abc")))
        descA.name = "a2"
        descB = owcolor.DiscAttrDesc(
            DiscreteVariable("varB", values=tuple("abc")))
        descB.set_value(1, "X")
        descC = owcolor.ContAttrDesc(ContinuousVariable("varC"))
        descC.name = "c2"
        descD = owcolor.ContAttrDesc(ContinuousVariable("varD"))
        descD.palette_name = "linear_viridis"
        descE = owcolor.ContAttrDesc(ContinuousVariable("varE"))
        self.widget.disc_descs = [descA, descB]
        self.widget.cont_descs = [descC, descD, descE]

        with tempfile.TemporaryDirectory() as path:
            fname = os.path.join(path, "foo.colors")
            self.widget._save_var_defs(fname)
            with open(fname) as f:
                js = json.load(f)
            self.assertEqual(js["categorical"],
                             {"varA": {"rename": "a2"},
                              "varB": {"renamed_values": {"b": "X"}}})
            self.assertEqual(js["numeric"],
                             {"varC": {"rename": "c2"},
                              "varD": {"colors": "linear_viridis"}})

    @patch("Orange.widgets.data.owcolor.QMessageBox.critical")
    def test_load(self, msg_box):
        self.widget._parse_var_defs = Mock()
        self.widget._save_var_defs = Mock()
        with patch.object(owcolor.QFileDialog, "getOpenFileName",
                          return_value=("", "")):
            with patch("builtins.open"):
                self.widget.load()
                open.assert_not_called()
                self.widget._parse_var_defs.assert_not_called()

        with patch.object(owcolor.QFileDialog, "getOpenFileName",
                          return_value=("foo.colors", "*.colors")):
            with patch("json.load") as json_load, \
                    patch("builtins.open", side_effect=IOError):
                self.widget.load()
                msg_box.assert_called()
                msg_box.reset_mock()
                json_load.assert_not_called()
                self.widget._parse_var_defs.assert_not_called()

        with patch.object(owcolor.QFileDialog, "getOpenFileName",
                          return_value=("foo.colors", "*.colors")):
            with patch("json.load", side_effect=json.JSONDecodeError("err", "d", 42)), \
                    patch("builtins.open"):
                self.widget.load()
                msg_box.assert_called()
                msg_box.reset_mock()
                self.widget._parse_var_defs.assert_not_called()

        with patch.object(owcolor.QFileDialog, "getOpenFileName",
                          return_value=("foo.colors", "*.colors")):
            with patch("json.load"), patch("builtins.open"):
                self.widget.load()
                msg_box.assert_not_called()
                msg_box.reset_mock()
                self.widget._parse_var_defs.assert_called_with(json.load.return_value)

    @patch("Orange.widgets.data.owcolor.QMessageBox.warning")
    def test_load_ignore_warning(self, msg_box):
        self.widget._parse_var_defs(dict(categorical={}, numeric={}))
        msg_box.assert_not_called()

        no_change = dict(renamed_values={}, colors={})
        for names, message in (
                (("foo",),
                 "'foo'"),
                (("foo", "bar"),
                 "'foo' and 'bar'"),
                (("foo", "bar", "baz"),
                 "'foo', 'bar' and 'baz'"),
                (("foo", "bar", "baz", "qux"),
                 "'foo', 'bar', 'baz' and 'qux'"),
                (("foo", "bar", "baz", "qux", "quux"),
                 "'foo', 'bar', 'baz', 'qux' and 'quux'"),
                (("foo", "bar", "baz", "qux", "quux", "corge"),
                 "'foo', 'bar', 'baz', 'qux' and 2 other"),
                (("foo", "bar", "baz", "qux", "quux", "corge", "grault"),
                 "'foo', 'bar', 'baz', 'qux' and 3 other")):
            self.widget._parse_var_defs(dict(
                categorical=dict.fromkeys(names, no_change),
                numeric={}))
            self.assertIn(message, msg_box.call_args[0][2])

    def _create_descs(self):
        disc_vars = [DiscreteVariable(f"var{c}", values=("a", "b", "c"))
                     for c in "AB"]
        cont_vars = [ContinuousVariable(f"var{c}") for c in "CDE"]
        self.widget.disc_descs = [owcolor.DiscAttrDesc(v) for v in disc_vars]
        self.widget.cont_descs = [owcolor.ContAttrDesc(v) for v in cont_vars]
        return disc_vars, cont_vars

    def test_parse_var_defs(self):
        js = {"categorical": {"varA": {"rename": "a2"},
                              "varB": {"renamed_values": {"b": "X"}}},
              "numeric": {"varC": {"rename": "c2"},
                          "varD": {"colors": "linear_viridis"}}}

        self._create_descs()
        descE = self.widget.cont_descs[-1]

        self.widget._parse_var_defs(js)
        self.assertEqual(len(self.widget.disc_descs), 2)
        descA = self.widget.disc_descs[0]
        self.assertEqual(descA.name, "a2")
        self.assertIsNone(descA.new_values)
        self.assertIsNone(descA.new_colors)

        descB = self.widget.disc_descs[1]
        self.assertIsNone(descB.new_name)
        self.assertEqual(descB.new_values, ["a", "X", "c"])
        self.assertIsNone(descB.new_colors)

        self.assertEqual(len(self.widget.cont_descs), 3)
        descC = self.widget.cont_descs[0]
        self.assertEqual(descC.name, "c2")
        self.assertIsNone(descC.new_palette_name)

        descD = self.widget.cont_descs[1]
        self.assertIsNone(descD.new_name)
        self.assertEqual(descD.new_palette_name, "linear_viridis")

        self.assertIs(self.widget.cont_descs[2], descE)

    def test_parse_var_defs_invalid(self):
        self.assertRaises(
            owcolor.InvalidFileFormat,
            self.widget._parse_var_defs, 42)
        self.assertRaises(
            owcolor.InvalidFileFormat,
            self.widget._parse_var_defs,
            {"categorical": {"a": 42}, "numeric": {}})
        self.assertRaises(
            owcolor.InvalidFileFormat,
            self.widget._parse_var_defs,
            {"categorical": {"a": {"rename": 4}}, "numeric": {}})
        self.assertRaises(
            owcolor.InvalidFileFormat,
            self.widget._parse_var_defs,
            {"categorical": {42: {"rename": "b"}}, "numeric": {}})

    @patch("Orange.widgets.data.owcolor.QMessageBox.warning")
    def test_parse_var_defs_shows_warnings(self, msg_box):
        self._create_descs()
        self.widget._parse_var_defs(
            {"categorical": {"varA": {"renamed_values": {"a": "b"}}},
             "numeric": {}})
        msg_box.assert_called()
        self.assertTrue("duplicate names" in msg_box.call_args[0][2])

    @patch("Orange.widgets.data.owcolor.QMessageBox.warning")
    def test_parse_var_defs_no_rename(self, msg_box):
        self._create_descs()

        self.widget._parse_var_defs(
            {"categorical": {"varA": {"rename": "varB"}},
             "numeric": {}})
        msg_box.assert_called()
        self.assertTrue("duplicated names" in msg_box.call_args[0][2])
        msg_box.reset_mock()

        self.widget._parse_var_defs(
            {"categorical": {"varA": {"rename": "X"}},
             "numeric": {"varD": {"rename": "X"}}})
        msg_box.assert_called()
        self.assertTrue("duplicated names" in msg_box.call_args[0][2])
        msg_box.reset_mock()

        self.widget._parse_var_defs(
            {"categorical": {"varA": {"rename": "varD"}},
             "numeric": {"varD": {"rename": "varA"}}})
        msg_box.assert_not_called()


if __name__ == "__main__":
    unittest.main()
