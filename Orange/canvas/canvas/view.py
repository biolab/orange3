"""
Canvas Graphics View
"""
import logging

from PyQt4.QtGui import QGraphicsView, QCursor, QIcon
from PyQt4.QtCore import Qt, QRect, QSize, QRectF, QPoint, QTimer

log = logging.getLogger(__name__)


class CanvasView(QGraphicsView):
    """Canvas View handles the zooming.
    """

    def __init__(self, *args):
        QGraphicsView.__init__(self, *args)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.__backgroundIcon = QIcon()

        self.__autoScroll = False
        self.__autoScrollMargin = 16
        self.__autoScrollTimer = QTimer(self)
        self.__autoScrollTimer.timeout.connect(self.__autoScrollAdvance)

    def setScene(self, scene):
        QGraphicsView.setScene(self, scene)
        self._ensureSceneRect(scene)

    def _ensureSceneRect(self, scene):
        r = scene.addRect(QRectF(0, 0, 400, 400))
        scene.sceneRect()
        scene.removeItem(r)

    def setAutoScrollMargin(self, margin):
        self.__autoScrollMargin = margin

    def autoScrollMargin(self):
        return self.__autoScrollMargin

    def setAutoScroll(self, enable):
        self.__autoScroll = enable

    def autoScroll(self):
        return self.__autoScroll

    def mousePressEvent(self, event):
        QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if not self.__autoScrollTimer.isActive() and \
                    self.__shouldAutoScroll(event.pos()):
                self.__startAutoScroll()

        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() & Qt.LeftButton:
            self.__stopAutoScroll()

        return QGraphicsView.mouseReleaseEvent(self, event)

    def __shouldAutoScroll(self, pos):
        if self.__autoScroll:
            margin = self.__autoScrollMargin
            viewrect = self.contentsRect()
            rect = viewrect.adjusted(margin, margin, -margin, -margin)
            # only do auto scroll when on the viewport's margins
            return not rect.contains(pos) and viewrect.contains(pos)
        else:
            return False

    def __startAutoScroll(self):
        self.__autoScrollTimer.start(10)
        log.debug("Auto scroll timer started")

    def __stopAutoScroll(self):
        if self.__autoScrollTimer.isActive():
            self.__autoScrollTimer.stop()
            log.debug("Auto scroll timer stopped")

    def __autoScrollAdvance(self):
        """Advance the auto scroll
        """
        pos = QCursor.pos()
        pos = self.mapFromGlobal(pos)
        margin = self.__autoScrollMargin

        vvalue = self.verticalScrollBar().value()
        hvalue = self.horizontalScrollBar().value()

        vrect = QRect(0, 0, self.width(), self.height())

        # What should be the speed
        advance = 10

        # We only do auto scroll if the mouse is inside the view.
        if vrect.contains(pos):
            if pos.x() < vrect.left() + margin:
                self.horizontalScrollBar().setValue(hvalue - advance)
            if pos.y() < vrect.top() + margin:
                self.verticalScrollBar().setValue(vvalue - advance)
            if pos.x() > vrect.right() - margin:
                self.horizontalScrollBar().setValue(hvalue + advance)
            if pos.y() > vrect.bottom() - margin:
                self.verticalScrollBar().setValue(vvalue + advance)

            if self.verticalScrollBar().value() == vvalue and \
                    self.horizontalScrollBar().value() == hvalue:
                self.__stopAutoScroll()
        else:
            self.__stopAutoScroll()

        log.debug("Auto scroll advance")

    def setBackgroundIcon(self, icon):
        if not isinstance(icon, QIcon):
            raise TypeError("A QIcon expected.")

        if self.__backgroundIcon != icon:
            self.__backgroundIcon = icon
            self.viewport().update()

    def backgroundIcon(self):
        return QIcon(self.__backgroundIcon)

    def drawBackground(self, painter, rect):
        QGraphicsView.drawBackground(self, painter, rect)

        if not self.__backgroundIcon.isNull():
            painter.setClipRect(rect)
            vrect = QRect(QPoint(0, 0), self.viewport().size())
            vrect = self.mapToScene(vrect).boundingRect()

            pm = self.__backgroundIcon.pixmap(
                vrect.size().toSize().boundedTo(QSize(200, 200))
            )
            pmrect = QRect(QPoint(0, 0), pm.size())
            pmrect.moveCenter(vrect.center().toPoint())
            if rect.toRect().intersects(pmrect):
                painter.drawPixmap(pmrect, pm)
