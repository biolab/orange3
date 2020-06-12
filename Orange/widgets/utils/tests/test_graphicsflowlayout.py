from AnyQt.QtCore import Qt, QSizeF, QRectF
from AnyQt.QtWidgets import QGraphicsWidget

from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.graphicsflowlayout import GraphicsFlowLayout


class TestGraphicsFlowLayout(GuiTest):
    def test_layout(self):
        layout = GraphicsFlowLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setHorizontalSpacing(3)
        self.assertEqual(layout.horizontalSpacing(), 3)
        layout.setVerticalSpacing(3)
        self.assertEqual(layout.verticalSpacing(), 3)

        def widget():
            w = QGraphicsWidget()
            w.setMinimumSize(QSizeF(10, 10))
            w.setMaximumSize(QSizeF(10, 10))
            return w

        layout.addItem(widget())
        layout.addItem(widget())
        layout.addItem(widget())
        self.assertEqual(layout.count(), 3)
        sh = layout.effectiveSizeHint(Qt.PreferredSize)
        self.assertEqual(sh, QSizeF(30 + 6 + 2, 12))
        sh = layout.effectiveSizeHint(Qt.PreferredSize, QSizeF(12, -1))
        self.assertEqual(sh, QSizeF(12, 30 + 6 + 2))
        layout.setGeometry(QRectF(0, 0, sh.width(), sh.height()))
        w1 = layout.itemAt(0)
        self.assertEqual(w1.geometry(), QRectF(1, 1, 10, 10))
        w3 = layout.itemAt(2)
        self.assertEqual(w3.geometry(), QRectF(1, 1 + 2 * 10 + 2 * 3, 10, 10))

    def test_add_remove(self):
        layout = GraphicsFlowLayout()
        layout.addItem(GraphicsFlowLayout())
        layout.removeAt(0)
        self.assertEqual(layout.count(), 0)
        layout.addItem(GraphicsFlowLayout())
        item = layout.itemAt(0)
        self.assertIs(item.parentLayoutItem(), layout)
        layout.removeItem(item)
        self.assertIs(item.parentLayoutItem(), None)
