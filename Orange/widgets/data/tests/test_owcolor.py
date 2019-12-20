# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, protected-access
import unittest
from unittest.mock import patch, Mock

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor

from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.data.owcolor import OWColor, ColorRole, DiscColorTableModel
from Orange.widgets.tests.base import WidgetTest


class TestOWColor(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWColor)
        self.iris = Table("iris")

    def test_reuse_old_settings(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        assert isinstance(self.widget, OWColor)
        self.widget.saveSettings()

        w = self.create_widget(OWColor, reset_default_settings=False)
        self.send_signal(self.widget.Inputs.data, self.iris, widget=w)

    def test_invalid_input_colors(self):
        a = ContinuousVariable("a")
        a.attributes["colors"] = "invalid"
        _ = a.colors
        t = Table.from_domain(Domain([a]))

        self.send_signal(self.widget.Inputs.data, t)

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget, 'unconditional_commit') as commit:
            self.widget.auto_apply = False
            commit.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.iris)
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

    def test_base_model(self):
        widget = self.widget
        dm = widget.disc_model
        cm = widget.disc_model

        data = Table("heart_disease")
        self.send_signal(widget.Inputs.data, data)
        self.assertEqual(
            [dm.data(dm.index(i, 0)) for i in range(dm.rowCount())],
            [var.name for var in data.domain.variables if var.is_discrete]
        )
        self.assertEqual(
            [dm.data(cm.index(i, 0)) for i in range(cm.rowCount())],
            [var.name for var in data.domain.variables if var.is_discrete]
        )

        dm.setData(dm.index(1, 0), "foo", Qt.EditRole)
        self.assertEqual(dm.data(dm.index(1, 0)), "foo")
        self.assertEqual(widget.disc_colors[1].name, "foo")

        widget.disc_colors[1].name = "bar"
        self.assertEqual(dm.data(dm.index(1, 0)), "bar")

    def test_disc_model(self):
        widget = self.widget
        dm = widget.disc_model

        data = Table("heart_disease")
        self.send_signal(widget.Inputs.data, data)

        # Consider these two as sanity checks
        self.assertEqual(dm.data(dm.index(0, 0)), "gender")
        self.assertEqual(dm.data(dm.index(1, 0)), "chest pain")

        self.assertEqual(dm.columnCount(), 5)  # 1 + four types of chest pain
        self.assertEqual(dm.data(dm.index(0, 1)), "female")
        self.assertEqual(dm.data(dm.index(0, 2)), "male")
        self.assertIsNone(dm.data(dm.index(0, 3)))
        self.assertIsNone(dm.data(dm.index(0, 4)))
        self.assertIsNone(dm.data(dm.index(0, 3), ColorRole))
        self.assertIsNone(dm.data(dm.index(0, 3), Qt.DecorationRole))
        self.assertIsNone(dm.data(dm.index(0, 3), Qt.ToolTipRole))

        chest_pain = data.domain["chest pain"]
        self.assertEqual(
            [dm.data(dm.index(1, i)) for i in range(1, 5)],
            list(chest_pain.values))
        np.testing.assert_equal(
            [dm.data(dm.index(1, i), ColorRole) for i in range(1, 5)],
            list(chest_pain.colors))
        self.assertEqual(
            [dm.data(dm.index(1, i), Qt.DecorationRole) for i in range(1, 5)],
            [QColor(*color) for color in chest_pain.colors])
        self.assertEqual(
            [dm.data(dm.index(1, i), Qt.ToolTipRole) for i in range(1, 5)],
            list(map(DiscColorTableModel._encode_color, chest_pain.colors)))

        dm.setData(dm.index(0, 1), "F", Qt.EditRole)
        self.assertEqual(dm.data(dm.index(0, 1)), "F")
        self.assertEqual(widget.disc_colors[0].values, ["F", "male"])

        widget.disc_colors[0].values[1] = "M"
        self.assertEqual(dm.data(dm.index(0, 2)), "M")

        dm.setData(dm.index(0, 1), (1, 2, 3, 4), ColorRole)
        self.assertEqual(list(dm.data(dm.index(0, 1), ColorRole)), [1, 2, 3])
        self.assertEqual(list(widget.disc_colors[0].colors[0]), [1, 2, 3])

        widget.disc_colors[0].colors[1] = (4, 5, 6)
        self.assertEqual(list(dm.data(dm.index(0, 2), ColorRole)), [4, 5, 6])

    def test_report(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()  # don't crash


if __name__ == "__main__":
    unittest.main()
