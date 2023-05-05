import unittest

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor

from orangewidget.tests.base import GuiTest

from Orange.misc import DistMatrix
from Orange.widgets.utils.itemdelegates import FixedFormatNumericColumnDelegate
from Orange.widgets.utils.distmatrixmodel import DistMatrixModel


class TestModel(GuiTest):
    def assert_brush_value(self, brush, value):
        self.assert_brush_color(brush, QColor.fromHsv(120, int(value), 255))

    def assert_brush_color(self, brush, color):
        self.assertEqual(brush.color().getRgb()[:3], color.getRgb()[:3])

    def test_data(self):
        model = DistMatrixModel()

        dist = DistMatrix(np.array([[1.0, 2, 3], [0, 10, 5]]))
        model.set_data(dist)

        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(model.columnCount(), 3)
        index = model.index(0, 1)
        self.assertEqual(
            index.data(FixedFormatNumericColumnDelegate.ColumnDataSpanRole),
            (0, 10))
        self.assertEqual(index.data(Qt.DisplayRole), 2)
        self.assert_brush_value(index.data(Qt.BackgroundRole), 2 / 10 * 170)

    def test_header_data(self):
        model = DistMatrixModel()

        dist = DistMatrix(np.array([[1.0, 2, 3], [0, 10, 5]]))
        model.set_data(dist)

        self.assertIsNone(model.headerData(1, Qt.Horizontal, Qt.DisplayRole))
        self.assertIsNone(model.headerData(1, Qt.Horizontal, Qt.BackgroundRole))
        self.assertIsNone(model.headerData(1, Qt.Horizontal, Qt.ForegroundRole))
        self.assertIsNone(model.headerData(1, Qt.Vertical, Qt.DisplayRole))
        self.assertIsNone(model.headerData(1, Qt.Vertical, Qt.BackgroundRole))
        self.assertIsNone(model.headerData(1, Qt.Vertical, Qt.ForegroundRole))

        model.set_labels(Qt.Horizontal, list("abc"))
        self.assertEqual(model.headerData(1, Qt.Horizontal, Qt.DisplayRole), "b")
        self.assertIsNone(model.headerData(1, Qt.Vertical, Qt.DisplayRole))
        # These shouldn't fail; what they return ... we don't care here
        model.headerData(1, Qt.Horizontal, Qt.BackgroundRole)
        model.headerData(1, Qt.Horizontal, Qt.ForegroundRole)
        model.headerData(1, Qt.Vertical, Qt.BackgroundRole)
        model.headerData(1, Qt.Vertical, Qt.ForegroundRole)

        model.set_labels(Qt.Vertical, list("de"))
        self.assertEqual(model.headerData(1, Qt.Horizontal, Qt.DisplayRole), "b")
        self.assertEqual(model.headerData(1, Qt.Vertical, Qt.DisplayRole), "e")

        model.set_labels(Qt.Horizontal, None)
        self.assertIsNone(model.headerData(1, Qt.Horizontal, Qt.DisplayRole))
        self.assertEqual(model.headerData(1, Qt.Vertical, Qt.DisplayRole), "e")

        colors = np.array([QColor(1, 2, 3), QColor(4, 5, 6), QColor(7, 8, 9)])
        model.set_labels(Qt.Horizontal, list("abc"), colors)

        self.assert_brush_color(
            model.headerData(1, Qt.Horizontal, Qt.BackgroundRole),
            colors[1].lighter(150))
        self.assertIsNone(model.headerData(1, Qt.Vertical, Qt.BackgroundRole))

        vcolors = np.array([QColor(12, 13, 14), QColor(9, 10, 11)])
        model.set_labels(Qt.Vertical, list("de"), vcolors)
        self.assert_brush_color(
            model.headerData(1, Qt.Horizontal, Qt.BackgroundRole),
            colors[1].lighter(150))
        self.assert_brush_color(
            model.headerData(1, Qt.Vertical, Qt.BackgroundRole),
            vcolors[1].lighter(150))


if __name__ == "__main__":
    unittest.main()
