import numpy as np

from AnyQt.QtCore import (
    QRectF, QLineF, QObject, QEvent, Qt, pyqtSignal as Signal
)
from AnyQt.QtGui import QTransform
from AnyQt.QtWidgets import (
    QGraphicsLineItem, QGraphicsSceneMouseEvent, QPinchGesture
)

import pyqtgraph as pg

from Orange.widgets.utils.plot import SELECT, PANNING, ZOOMING


class TextItem(pg.TextItem):
    if not hasattr(pg.TextItem, "setAnchor"):
        # Compatibility with pyqtgraph <= 0.9.10; in (as of yet unreleased)
        # 0.9.11 the TextItem has a `setAnchor`, but not `updateText`
        def setAnchor(self, anchor):
            self.anchor = pg.Point(anchor)
            self.updateText()


class AnchorItem(pg.GraphicsObject):
    def __init__(self, parent=None, line=QLineF(), text="", **kwargs):
        super().__init__(parent, **kwargs)
        self._text = text
        self.setFlag(pg.GraphicsObject.ItemHasNoContents)

        self._spine = QGraphicsLineItem(line, self)
        angle = line.angle()

        self._arrow = pg.ArrowItem(parent=self, angle=0)
        self._arrow.setPos(self._spine.line().p2())
        self._arrow.setRotation(angle)
        self._arrow.setStyle(headLen=10)

        self._label = TextItem(text=text, color=(10, 10, 10))
        self._label.setParentItem(self)
        self._label.setPos(self._spine.line().p2())

        if parent is not None:
            self.setParentItem(parent)

    def setText(self, text):
        if text != self._text:
            self._text = text
            self._label.setText(text)
            self._label.setVisible(bool(text))

    def text(self):
        return self._text

    def setLine(self, *line):
        line = QLineF(*line)
        if line != self._spine.line():
            self._spine.setLine(line)
            self.__updateLayout()

    def line(self):
        return self._spine.line()

    def setPen(self, pen):
        self._spine.setPen(pen)

    def setArrowVisible(self, visible):
        self._arrow.setVisible(visible)

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()

    def viewTransformChanged(self):
        self.__updateLayout()

    def __updateLayout(self):
        T = self.sceneTransform()
        if T is None:
            T = QTransform()

        # map the axis spine to scene coord. system.
        viewbox_line = T.map(self._spine.line())
        angle = viewbox_line.angle()
        assert not np.isnan(angle)
        # note in Qt the y axis is inverted (90 degree angle 'points' down)
        left_quad = 270 < angle <= 360 or -0.0 <= angle < 90

        # position the text label along the viewbox_line
        label_pos = self._spine.line().pointAt(0.90)

        if left_quad:
            # Anchor the text under the axis spine
            anchor = (0.5, -0.1)
        else:
            # Anchor the text over the axis spine
            anchor = (0.5, 1.1)

        self._label.setPos(label_pos)
        self._label.setAnchor(pg.Point(*anchor))
        self._label.setRotation(-angle if left_quad else 180 - angle)

        self._arrow.setPos(self._spine.line().p2())
        self._arrow.setRotation(180 - angle)


class HelpEventDelegate(QObject):
    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate

    def eventFilter(self, obj, event):
        if event.type() == QEvent.GraphicsSceneHelp:
            return self.delegate(event)
        else:
            return False


class MouseEventDelegate(HelpEventDelegate):
    def __init__(self, delegate, delegate2, parent=None):
        self.delegate2 = delegate2
        super().__init__(delegate, parent=parent)

    def eventFilter(self, obj, ev):
        if isinstance(ev, QGraphicsSceneMouseEvent):
            self.delegate2(ev)
        return super().eventFilter(obj, ev)


class InteractiveViewBox(pg.ViewBox):
    def __init__(self, graph, enable_menu=False):
        self.init_history()
        pg.ViewBox.__init__(self, enableMenu=enable_menu)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.grabGesture(Qt.PinchGesture)

    def _dragtip_pos(self):
        return 10, 10

    def safe_update_scale_box(self, buttonDownPos, currentPos):
        x, y = currentPos
        if buttonDownPos[0] == x:
            x += 1
        if buttonDownPos[1] == y:
            y += 1
        self.updateScaleBox(buttonDownPos, pg.Point(x, y))

    # noinspection PyPep8Naming,PyMethodOverriding
    def mouseDragEvent(self, ev, axis=None):
        if self.graph.state == SELECT and axis is None:
            ev.accept()
            pos = ev.pos()
            if ev.button() == Qt.LeftButton:
                self.safe_update_scale_box(ev.buttonDownPos(), ev.pos())
                scene = self.scene()
                dragtip = scene.drag_tooltip
                if ev.isFinish():
                    dragtip.hide()
                    self.rbScaleBox.hide()
                    pixel_rect = QRectF(ev.buttonDownPos(ev.button()), pos)
                    value_rect = self.childGroup.mapRectFromParent(pixel_rect)
                    self.graph.select_by_rectangle(value_rect)
                else:
                    dragtip.setPos(*self._dragtip_pos())
                    dragtip.show()  # although possibly already shown
                    self.safe_update_scale_box(ev.buttonDownPos(), ev.pos())
        elif self.graph.state == ZOOMING or self.graph.state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()

    def updateAutoRange(self):
        # indirectly called by the autorange button on the graph
        super().updateAutoRange()
        self.tag_history()

    def tag_history(self):
        #add current view to history if it differs from the last view
        if self.axHistory:
            currentview = self.viewRect()
            lastview = self.axHistory[self.axHistoryPointer]
            inters = currentview & lastview
            united = currentview.united(lastview)
            if inters.width()*inters.height()/(united.width()*united.height()) > 0.95:
                return
        self.axHistoryPointer += 1
        self.axHistory = self.axHistory[:self.axHistoryPointer] + \
                         [self.viewRect()]

    def init_history(self):
        self.axHistory = []
        self.axHistoryPointer = -1

    def autoRange(self, padding=None, items=None, item=None):
        super().autoRange(padding=padding, items=items, item=item)
        self.tag_history()

    def suggestPadding(self, axis): #no padding so that undo works correcty
        return 0.

    def scaleHistory(self, d):
        self.tag_history()
        super().scaleHistory(d)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:  # undo zoom
            self.scaleHistory(-1)
        else:
            ev.accept()
            self.graph.unselect_all()

    def sceneEvent(self, event):
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        return super().sceneEvent(event)

    def gestureEvent(self, event):
        gesture = event.gesture(Qt.PinchGesture)
        if gesture.state() == Qt.GestureStarted:
            event.accept(gesture)
        elif gesture.changeFlags() & QPinchGesture.ScaleFactorChanged:
            center = self.mapSceneToView(gesture.centerPoint())
            scale_prev = gesture.lastScaleFactor()
            scale = gesture.scaleFactor()
            if scale_prev != 0:
                scale = scale / scale_prev
            if scale > 0:
                self.scaleBy((1 / scale, 1 / scale), center)
        elif gesture.state() == Qt.GestureFinished:
            self.tag_history()

        return True


class VizInteractiveViewBox(InteractiveViewBox):
    started = Signal()
    moved = Signal()
    finished = Signal()

    def __init__(self, graph, enable_menu=False):
        self.mouse_state = 0
        self.point_i = None
        super().__init__(graph, enable_menu)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        pos = self.childGroup.mapFromParent(ev.pos())
        if self.graph.can_show_indicator(pos)[0]:
            self.setCursor(Qt.ClosedHandCursor)

    def mouseDragEvent(self, ev, axis=None):
        pos = self.childGroup.mapFromParent(ev.pos())
        can_show, point_i = self.graph.can_show_indicator(pos)
        if ev.button() != Qt.LeftButton or (ev.start and not can_show):
            self.mouse_state = 2
        if self.mouse_state == 2:
            if ev.finish:
                self.mouse_state = 0
            super().mouseDragEvent(ev, axis)
            return

        ev.accept()
        if ev.start:
            self.setCursor(Qt.ClosedHandCursor)
            self.mouse_state = 1
            self.point_i = point_i
            self.started.emit()

        if self.mouse_state == 1:
            self.graph.set_point(self.point_i, pos.x(), pos.y())
            if ev.finish:
                self.setCursor(Qt.OpenHandCursor)
                self.mouse_state = 0
                self.finished.emit()
            else:
                self._show_tooltip(ev)
                self.moved.emit()
            self.graph.show_indicator(self.point_i)

    def _show_tooltip(self, ev):
        pass
