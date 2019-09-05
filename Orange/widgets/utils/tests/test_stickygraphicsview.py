from PyQt5.QtCore import Qt, QRectF, QPoint, QPointF
from PyQt5.QtGui import QBrush, QWheelEvent
from PyQt5.QtWidgets import QGraphicsScene, QWidget, QApplication

from Orange.widgets.tests.base import GuiTest

from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView


class TestStickyGraphicsView(GuiTest):
    def test(self):
        view = StickyGraphicsView()
        scene = QGraphicsScene(view)
        scene.setBackgroundBrush(QBrush(Qt.lightGray, Qt.CrossPattern))
        view.setScene(scene)
        scene.addRect(
            QRectF(0, 0, 300, 20), Qt.red, QBrush(Qt.red, Qt.BDiagPattern))
        scene.addRect(QRectF(0, 25, 300, 100))
        scene.addRect(
            QRectF(0, 130, 300, 20),
            Qt.darkGray, QBrush(Qt.darkGray, Qt.BDiagPattern)
        )
        view.setHeaderSceneRect(QRectF(0, 0, 300, 20))
        view.setFooterSceneRect(QRectF(0, 130, 300, 20))

        header = view.headerView()
        footer = view.footerView()

        view.resize(310, 310)
        view.grab()

        self.assertFalse(header.isVisibleTo(view))
        self.assertFalse(footer.isVisibleTo(view))

        view.resize(310, 100)
        view.verticalScrollBar().setValue(0)  # scroll to top
        view.grab()

        self.assertFalse(header.isVisibleTo(view))
        self.assertTrue(footer.isVisibleTo(view))

        view.verticalScrollBar().setValue(
            view.verticalScrollBar().maximum())  # scroll to bottom
        view.grab()

        self.assertTrue(header.isVisibleTo(view))
        self.assertFalse(footer.isVisibleTo(view))

        qWheelScroll(header.viewport(), angleDelta=QPoint(0, -720 * 8))


def qWheelScroll(
        widget: QWidget, buttons=Qt.NoButton, modifiers=Qt.NoModifier,
        pos=QPoint(), angleDelta=QPoint(0, 1),
):
    if pos.isNull():
        pos = widget.rect().center()
    globalPos = widget.mapToGlobal(pos)

    if angleDelta.y() >= angleDelta.x():
        qt4orient = Qt.Vertical
        qt4delta = angleDelta.y()
    else:
        qt4orient = Qt.Horizontal
        qt4delta = angleDelta.x()

    event = QWheelEvent(
        QPointF(pos), QPointF(globalPos), QPoint(), angleDelta,
        qt4delta, qt4orient, buttons, modifiers
    )
    QApplication.sendEvent(widget, event)
