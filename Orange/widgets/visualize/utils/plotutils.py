import itertools

import numpy as np

from AnyQt.QtCore import (
    QRectF, QLineF, QObject, QEvent, Qt, pyqtSignal as Signal
)
from AnyQt.QtGui import QTransform, QFontMetrics, QStaticText, QBrush, QPen, \
    QFont, QPalette
from AnyQt.QtWidgets import (
    QGraphicsLineItem, QGraphicsSceneMouseEvent, QPinchGesture,
    QGraphicsItemGroup, QWidget
)

import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from pyqtgraph.graphicsItems.ScatterPlotItem import drawSymbol

from Orange.widgets.utils.plot import SELECT, PANNING, ZOOMING


class TextItem(pg.TextItem):
    if not hasattr(pg.TextItem, "setAnchor"):
        # Compatibility with pyqtgraph <= 0.9.10; in (as of yet unreleased)
        # 0.9.11 the TextItem has a `setAnchor`, but not `updateText`
        def setAnchor(self, anchor):
            self.anchor = pg.Point(anchor)
            self.updateText()

    def get_xy(self):
        point = self.pos()
        return point.x(), point.y()


class AnchorItem(pg.GraphicsWidget):
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
        self._label.setPos(*self.get_xy())
        self._label.setColor(self.palette().color(QPalette.Text))

        if parent is not None:
            self.setParentItem(parent)

    def get_xy(self):
        point = self._spine.line().p2()
        return point.x(), point.y()

    def setFont(self, font):
        self._label.setFont(font)

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

    def changeEvent(self, event):
        if event.type() == QEvent.PaletteChange:
            self._label.setColor(self.palette().color(QPalette.Text))
        super().changeEvent(event)


class HelpEventDelegate(QObject):
    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate

    def eventFilter(self, _, event):
        if event.type() == QEvent.GraphicsSceneHelp:
            return self.delegate(event)
        else:
            return False


class MouseEventDelegate(HelpEventDelegate):
    def __init__(self, delegate, delegate2, parent=None):
        self.delegate2 = delegate2
        super().__init__(delegate, parent=parent)

    def eventFilter(self, obj, event):
        if isinstance(event, QGraphicsSceneMouseEvent):
            self.delegate2(event)
        return super().eventFilter(obj, event)


class InteractiveViewBox(pg.ViewBox):
    def __init__(self, graph, enable_menu=False):
        self.init_history()
        pg.ViewBox.__init__(self, enableMenu=enable_menu)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.grabGesture(Qt.PinchGesture)

    @staticmethod
    def _dragtip_pos():
        return 10, 10

    def setDragTooltip(self, tooltip):
        scene = self.scene()
        scene.addItem(tooltip)
        tooltip.setPos(*self._dragtip_pos())
        tooltip.hide()
        scene.drag_tooltip = tooltip

    def updateScaleBox(self, p1, p2):
        """
        Overload to use ViewBox.mapToView instead of mapRectFromParent
        mapRectFromParent (from Qt) uses QTransform.invert() which has
        floating-point issues and can't invert the matrix with large
        coefficients. ViewBox.mapToView uses invertQTransform from pyqtgraph.

        This code, except for first three lines, are copied from the overloaded
        method.
        """
        p1 = self.mapToView(p1)
        p2 = self.mapToView(p2)
        r = QRectF(p1, p2)
        self.rbScaleBox.setPos(r.topLeft())
        tr = QTransform.fromScale(r.width(), r.height())
        self.rbScaleBox.setTransform(tr)
        self.rbScaleBox.show()

    def safe_update_scale_box(self, buttonDownPos, currentPos):
        x, y = currentPos
        if buttonDownPos[0] == x:
            x += 1
        if buttonDownPos[1] == y:
            y += 1
        self.updateScaleBox(buttonDownPos, pg.Point(x, y))

    def _updateDragtipShown(self, enabled):
        scene = self.scene()
        dragtip = scene.drag_tooltip
        if enabled != dragtip.isVisible():
            dragtip.setVisible(enabled)

    # noinspection PyPep8Naming,PyMethodOverriding
    def mouseDragEvent(self, ev, axis=None):
        def get_mapped_rect():
            p1, p2 = ev.buttonDownPos(ev.button()), ev.pos()
            p1 = self.mapToView(p1)
            p2 = self.mapToView(p2)
            return QRectF(p1, p2)

        def select():
            ev.accept()
            if ev.button() == Qt.LeftButton:
                self.safe_update_scale_box(ev.buttonDownPos(), ev.pos())
                if ev.isFinish():
                    self._updateDragtipShown(False)
                    self.graph.unsuspend_jittering()
                    self.rbScaleBox.hide()
                    value_rect = get_mapped_rect()
                    self.graph.select_by_rectangle(value_rect)
                else:
                    self._updateDragtipShown(True)
                    self.graph.suspend_jittering()
                    self.safe_update_scale_box(ev.buttonDownPos(), ev.pos())

        def zoom():
            # A fixed version of the code from the inherited mouseDragEvent
            # Use mapToView instead of mapRectFromParent
            ev.accept()
            self.rbScaleBox.hide()
            ax = get_mapped_rect()
            self.showAxRect(ax)
            self.axHistoryPointer += 1
            self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]

        if self.graph.state == SELECT and axis is None:
            select()
        elif self.graph.state == ZOOMING or self.graph.state == PANNING:
            # Inherited mouseDragEvent doesn't work for large zooms because it
            # uses mapRectFromParent. We don't want to copy the parts of the
            # method that work, hence we only use our code under the following
            # conditions.
            if ev.button() & (Qt.LeftButton | Qt.MidButton) \
                    and self.state['mouseMode'] == pg.ViewBox.RectMode \
                    and ev.isFinish():
                zoom()
            else:
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

    def suggestPadding(self, _):  # no padding so that undo works correcty
        return 0.

    def scaleHistory(self, d):
        self.tag_history()
        super().scaleHistory(d)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:  # undo zoom
            self.scaleHistory(-1)
        elif ev.modifiers() == Qt.NoModifier:
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


class DraggableItemsViewBox(InteractiveViewBox):
    """
    A viewbox with draggable items

    Graph that uses it must provide two methods:
    - `closest_draggable_item(pos)` returns an int representing the id of the
      draggable item that is closest (and close enough) to `QPoint` pos, or
      `None`;
    - `show_indicator(item_id)` shows or updates an indicator for moving
      the item with the given `item_id`.

    Viewbox emits three signals:
    - `started = Signal(item_id)`
    - `moved = Signal(item_id, x, y)`
    - `finished = Signal(item_id, x, y)`
    """
    started = Signal(int)
    moved = Signal(int, float, float)
    finished = Signal(int, float, float)

    def __init__(self, graph, enable_menu=False):
        self.mouse_state = 0
        self.item_id = None
        super().__init__(graph, enable_menu)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        pos = self.childGroup.mapFromParent(ev.pos())
        if self.graph.closest_draggable_item(pos) is not None:
            self.setCursor(Qt.ClosedHandCursor)

    def mouseDragEvent(self, ev, axis=None):
        pos = self.childGroup.mapFromParent(ev.pos())
        item_id = self.graph.closest_draggable_item(pos)
        if ev.button() != Qt.LeftButton or (ev.start and item_id is None):
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
            self.item_id = item_id
            self.started.emit(self.item_id)

        if self.mouse_state == 1:
            if ev.finish:
                self.mouse_state = 0
                self.finished.emit(self.item_id, pos.x(), pos.y())
                if self.graph.closest_draggable_item(pos) is not None:
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                    self.item_id = None
            else:
                self.moved.emit(self.item_id, pos.x(), pos.y())
            self.graph.show_indicator(self.item_id)


def wrap_legend_items(items, max_width, hspacing, vspacing):
    def line_width(line):
        return sum(item.boundingRect().width() for item in line) \
            + hspacing * (len(line) - 1)

    def create_line(line, yi, fixed_width=None):
        x = 0
        for item in line:
            item.setPos(x, yi * vspacing)
            paragraph.addToGroup(item)
            if fixed_width:
                x += fixed_width
            else:
                x += item.boundingRect().width() + hspacing

    max_item = max(item.boundingRect().width() + hspacing for item in items)
    in_line = int(max_width // max_item)
    if line_width(items) < max_width:  # single line
        lines = [items]
        fixed_width = None
    elif in_line < 2:
        lines = [[]]
        for i, item in enumerate(items):  # just a single column - free wrap
            lines[-1].append(item)
            if line_width(lines[-1]) > max_width and len(lines[-1]) > 1:
                lines.append([lines[-1].pop()])
        fixed_width = None
    else:  # arrange into grid
        lines = [items[i:i + in_line]
                 for i in range(0, len(items) + in_line - 1, in_line)]
        fixed_width = max_item

    paragraph = QGraphicsItemGroup()
    for yi, line in enumerate(lines):
        create_line(line, yi, fixed_width=fixed_width)
    return paragraph


class ElidedLabelsAxis(pg.AxisItem):
    """
    Horizontal axis that elides long text labels

    The class assumes that ticks with labels are distributed equally, and that
    standard `QWidget.font()` is used for printing them.
    """
    def generateDrawSpecs(self, p):
        axis_spec, tick_specs, text_specs = super().generateDrawSpecs(p)
        bounds = self.mapRectFromParent(self.geometry())
        max_width = int(0.9 * bounds.width() / (len(text_specs) or 1))
        elide = QFontMetrics(QWidget().font()).elidedText
        text_specs = [(rect, flags, elide(text, Qt.ElideRight, max_width))
                      for rect, flags, text in text_specs]
        return axis_spec, tick_specs, text_specs


class PaletteItemSample(ItemSample):
    """A color strip to insert into legends for discretized continuous values"""

    def __init__(self, palette, scale, label_formatter=None):
        """
        :param palette: palette used for showing continuous values
        :type palette: BinnedContinuousPalette
        :param scale: an instance of DiscretizedScale that defines the
                      conversion of values into bins
        :type scale: DiscretizedScale
        """
        super().__init__(None)
        self.palette = palette
        self.scale = scale
        if label_formatter is None:
            label_formatter = "{{:.{}f}}".format(scale.decimals).format
        cuts = [label_formatter(scale.offset + i * scale.width)
                for i in range(scale.bins + 1)]
        self.labels = [QStaticText("{} - {}".format(fr, to))
                       for fr, to in zip(cuts, cuts[1:])]
        self.font = self.font()
        self.font.setPointSize(11)

    @property
    def bin_height(self):
        return self.font.pointSize() + 4

    @property
    def text_width(self):
        for label in self.labels:
            label.prepare(font=self.font)
        return max(label.size().width() for label in self.labels)

    def set_font(self, font: QFont):
        self.font = font
        self.update()

    def boundingRect(self):
        return QRectF(0, 0,
                      25 + self.text_width + self.bin_height,
                      20 + self.scale.bins * self.bin_height)

    def paint(self, p, *args):
        p.setRenderHint(p.Antialiasing)
        p.translate(5, 5)
        p.setFont(self.font)
        colors = self.palette.qcolors
        foreground = super().palette().color(QPalette.Text)
        h = self.bin_height
        for i, color, label in zip(itertools.count(), colors, self.labels):
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRect(0, i * h, h, h)
            p.setPen(QPen(foreground))
            p.drawStaticText(h + 5, i * h + 1, label)


class SymbolItemSample(ItemSample):
    """Adjust position for symbols"""
    def __init__(self, pen, brush, size, symbol):
        super().__init__(None)
        self.__pen = fn.mkPen(pen)
        self.__brush = fn.mkBrush(brush)
        self.__size = size
        self.__symbol = symbol

    def paint(self, p, *args):
        p.translate(8, 12)
        drawSymbol(p, self.__symbol, self.__size, self.__pen, self.__brush)


class StyledAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label.setDefaultTextColor(self.palette().color(QPalette.Text))

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.FontChange:
            self.picture = None
            self.update()
        elif event.type() == QEvent.PaletteChange:
            self.picture = None
            self.label.setDefaultTextColor(self.palette().color(QPalette.Text))
            self.update()
        super().changeEvent(event)

    __hasTextPen = False

    def setTextPen(self, *args, **kwargs):
        self.__hasTextPen = args or kwargs
        super().setTextPen(*args, **kwargs)
        if not self.__hasTextPen:
            self.__clear_labelStyle_color()

    def textPen(self):
        if self.__hasTextPen:
            return super().textPen()
        else:  # bypass pg.AxisItem
            return QPen(self.palette().brush(QPalette.Text), 1)

    __hasPen = False

    def setPen(self, *args, **kwargs):
        self.__hasPen = bool(args or kwargs)
        super().setPen(*args, **kwargs)
        if not self.__hasPen:
            self.__clear_labelStyle_color()

    def pen(self):
        if self.__hasPen:
            return super().pen()
        else:  # bypass pg.AxisItem
            return QPen(self.palette().brush(QPalette.Text), 1)

    def __clear_labelStyle_color(self):
        try:
            self.labelStyle.pop("color")
        except AttributeError:
            pass


class AxisItem(StyledAxisItem):
    def __init__(self, orientation, rotate_ticks=False, **kwargs):
        super().__init__(orientation, **kwargs)
        self.style["rotateTicks"] = rotate_ticks

    def setRotateTicks(self, rotate):
        self.style["rotateTicks"] = rotate
        self.picture = None  # pylint: disable=attribute-defined-outside-init
        self.prepareGeometryChange()
        self.update()

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        if self.orientation in ["bottom", "top"] and self.style["rotateTicks"]:
            p.setRenderHint(p.Antialiasing, False)
            p.setRenderHint(p.TextAntialiasing, True)

            # draw long line along axis
            pen, p1, p2 = axisSpec
            p.setPen(pen)
            p.drawLine(p1, p2)
            p.translate(0.5, 0)  # resolves some damn pixel ambiguity

            # draw ticks
            for pen, p1, p2 in tickSpecs:
                p.setPen(pen)
                p.drawLine(p1, p2)

            # draw all text
            if self.style['tickFont'] is not None:
                p.setFont(self.style['tickFont'])
            p.setPen(self.pen())

            offset = self.style["tickTextOffset"][0]
            max_text_size = 0
            for rect, flags, text in textSpecs:
                p.save()
                p.translate(rect.x() + rect.width() / 2
                            - rect.y() - rect.height() / 2,
                            rect.x() + rect.width() + offset)
                p.rotate(-90)
                p.drawText(rect, flags, text)
                p.restore()
                max_text_size = max(max_text_size, rect.width())
            self._updateMaxTextSize(max_text_size + offset)
        else:
            super().drawPicture(p, axisSpec, tickSpecs, textSpecs)


class PlotWidget(pg.PlotWidget):
    """
    A pyqtgraph.PlotWidget with better QPalette integration.

    A default constructed plot will respect and adapt to the current palette
    """
    def __init__(self, *args, background=None, **kwargs):
        axisItems = kwargs.pop("axisItems", None)
        if axisItems is None:  # Use palette aware AxisItems
            axisItems = {"left": AxisItem("left"), "bottom": AxisItem("bottom")}
        super().__init__(*args, background=background, axisItems=axisItems,
                         **kwargs)
        if background is None:
            # Revert the pg.PlotWidget's modifications, use default
            # for QGraphicsView background role
            self.setBackgroundRole(QPalette.Base)
        # Reset changes to the palette (undo changes from pg.GraphicsView)
        self.setPalette(QPalette())
        self.__updateScenePalette()

    def setScene(self, scene):
        super().setScene(scene)
        self.__updateScenePalette()

    def changeEvent(self, event):
        if event.type() == QEvent.PaletteChange:
            self.__updateScenePalette()
            self.resetCachedContent()
        super().changeEvent(event)

    def __updateScenePalette(self):
        scene = self.scene()
        if scene is not None:
            scene.setPalette(self.palette())


class GraphicsView(pg.GraphicsView):
    """
    A pyqtgraph.GraphicsView with better QPalette integration.

    A default constructed plot will respect and adapt to the current palette
    """
    def __init__(self, *args, background=None, **kwargs):
        super().__init__(*args, background=background, **kwargs)
        if background is None:
            # Revert the pg.PlotWidget's modifications, use default
            # for QGraphicsView
            self.setBackgroundRole(QPalette.Base)
        # Reset changes to the palette (undo changes from pg.GraphicsView)
        self.setPalette(QPalette())
        self.__updateScenePalette()

    def setScene(self, scene):
        super().setScene(scene)
        self.__updateScenePalette()

    def changeEvent(self, event):
        if event.type() == QEvent.PaletteChange:
            self.__updateScenePalette()
            self.resetCachedContent()

        super().changeEvent(event)

    def __updateScenePalette(self):
        scene = self.scene()
        if scene is not None:
            scene.setPalette(self.palette())


class PlotItem(pg.PlotItem):
    """
    A pyqtgraph.PlotItem with better QPalette integration.

    A default constructed plot will respect and adapt to the current palette
    """
    def __init__(self, *args, **kwargs):
        axisItems = kwargs.pop("axisItems", None)
        if axisItems is None:
            axisItems = {"left": AxisItem("left"), "bottom": AxisItem("bottom")}
        super().__init__(*args, axisItems=axisItems, **kwargs)
