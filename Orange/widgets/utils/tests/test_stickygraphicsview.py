from AnyQt.QtCore import Qt, QRectF, QPoint, QPointF
from AnyQt.QtGui import QBrush, QWheelEvent
from AnyQt.QtWidgets import QGraphicsScene, QWidget, QApplication, QStyle

from Orange.widgets.tests.base import GuiTest

from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView


class TestStickyGraphicsView(GuiTest):
    def create_view(self):
        view = StickyGraphicsView()
        scene = QGraphicsScene(view)
        view.setScene(scene)
        return view

    def test(self):
        view = self.create_view()
        scene = view.scene()
        scene.setBackgroundBrush(QBrush(Qt.lightGray, Qt.CrossPattern))
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

    @staticmethod
    def _ensure_laid_out(view: QWidget) -> None:
        """Ensure view has had pending resize events flushed."""
        # when a widget is not visible it does not get resizeEvents dispatched
        # immediately, only before it is actually shown or its rendering is
        # requested.
        view.grab()

    def _test_visibility(self, view: StickyGraphicsView) -> None:
        header = view.headerView()
        footer = view.footerView()
        vsbar = view.verticalScrollBar()
        vsbar.triggerAction(vsbar.SliderToMinimum)
        self._ensure_laid_out(view)

        self.assertFalse(header.isVisibleTo(view))
        self.assertTrue(footer.isVisibleTo(view))

        vsbar.triggerAction(vsbar.SliderSingleStepAdd)
        self._ensure_laid_out(view)

        self.assertTrue(header.isVisibleTo(view))
        self.assertTrue(footer.isVisibleTo(view))

        vsbar.triggerAction(vsbar.SliderToMaximum)
        self._ensure_laid_out(view)

        self.assertTrue(header.isVisibleTo(view))
        self.assertFalse(footer.isVisibleTo(view))

        vsbar.triggerAction(vsbar.SliderSingleStepSub)
        self._ensure_laid_out(view)
        if not view.style().styleHint(QStyle.SH_ScrollBar_Transient, None, vsbar):
            # cannot reliably test due to QTBUG-65074
            self.assertTrue(header.isVisibleTo(view))
            self.assertTrue(footer.isVisibleTo(view))

    def test_fractional_1(self):
        view = self.create_view()
        view.resize(300, 100)
        scenerect = QRectF(-0.1, -0.1, 300.2, 300.2)
        headerrect = QRectF(-0.1, -0.1, 300.2, 20.2)
        footerrect = QRectF(-0.1, 279.9, 300.2, 20.2)
        view.setSceneRect(scenerect)
        view.setHeaderSceneRect(headerrect)
        view.setFooterSceneRect(footerrect)
        self._test_visibility(view)

    def test_fractional_2(self):
        view = self.create_view()
        view.resize(300, 100)
        view.grab()
        scenerect = QRectF(0.1, 0.1, 300, 299.8)
        headerrect = QRectF(0.1, 0.1, 300, 20)
        footerrect = QRectF(0.1, 299.9 - 20, 300, 20)
        view.setSceneRect(scenerect)
        view.setHeaderSceneRect(headerrect)
        view.setFooterSceneRect(footerrect)
        self._test_visibility(view)


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
