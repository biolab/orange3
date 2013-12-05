"""
    The Axis class display an axis on a graph

    The axis contains a line with configurable style, possible arrows, and a title

    .. attribute:: line_style
        The LineStyle with which the axis line is drawn

    .. attribute:: title
        The string to be displayed alongside the axis

    .. attribute:: title_above
        A boolean which specifies whether the title should be placed above or below the axis
        Normally the title would be above for top and left axes.

    .. attribute:: title_location
        can either be AxisStart, AxisEnd or AxisMiddle. The default is AxisMiddle

    .. attribute:: arrows
        A bitfield containing AxisEnd if an arrow should be drawn at the line's end (line.p2())
        and AxisStart if there should be an arrows at the first point.

        By default, there's an arrow at the end of the line

    .. attribute:: zoomable
        If this is set to True, the axis line will zoom together with the rest of the graph.
        Otherwise, the line will remain in place and only tick marks will zoom.

    .. method:: make_title
        Makes a pretty title, with the quantity title in italics and the unit in normal text

    .. method:: label_pos
        Controls where the axis title and tick marks are placed relative to the axis
"""

from math import *

from PyQt4.QtGui import QGraphicsItem, QGraphicsLineItem, QGraphicsTextItem, QPainterPath, QGraphicsPathItem, \
    QTransform, QGraphicsRectItem, QPen, QFontMetrics
from PyQt4.QtCore import QLineF, QPointF, QRectF, Qt

from .owconstants import *
from .owtools import resize_plot_item_list
from .owpalette import OWPalette


class OWAxis(QGraphicsItem):
    Role = OWPalette.Axis

    def __init__(self, id, title='', title_above=False, title_location=AxisMiddle, line=None, arrows=0, plot=None):
        QGraphicsItem.__init__(self)
        self.setFlag(QGraphicsItem.ItemHasNoContents)
        self.setZValue(AxisZValue)
        self.id = id
        self.title = title
        self.title_location = title_location
        self.data_line = line
        self.plot = plot
        self.graph_line = None
        self.size = None
        self.scale = None
        self.tick_length = (10, 5, 0)
        self.arrows = arrows
        self.title_above = title_above
        self.line_item = QGraphicsLineItem(self)
        self.title_item = QGraphicsTextItem(self)
        self.end_arrow_item = None
        self.start_arrow_item = None
        self.show_title = False
        self.scale = None
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.moveTo(0, 3.09)
        path.lineTo(0, -3.09)
        path.lineTo(9.51, 0)
        path.closeSubpath()
        self.arrow_path = path
        self.label_items = []
        self.label_bg_items = []
        self.tick_items = []
        self._ticks = []
        self.zoom_transform = QTransform()
        self.labels = None
        self.auto_range = None
        self.auto_scale = True

        self.zoomable = False
        self.update_callback = None
        self.max_text_width = 50
        self.text_margin = 5
        self.always_horizontal_text = False

    @staticmethod
    def compute_scale(min, max):
        magnitude = int(3 * log10(abs(max - min)) + 1)
        if magnitude % 3 == 0:
            first_place = 1
        elif magnitude % 3 == 1:
            first_place = 2
        else:
            first_place = 5
        magnitude = magnitude // 3 - 1
        step = first_place * pow(10, magnitude)
        first_val = ceil(min / step) * step
        return first_val, step

    def update_ticks(self):
        self._ticks = []
        major, medium, minor = self.tick_length
        if self.labels is not None and not self.auto_scale:
            for i, text in enumerate(self.labels):
                self._ticks.append((i, text, medium, 1))
        else:
            if self.scale and not self.auto_scale:
                min, max, step = self.scale
            elif self.auto_range:
                min, max = self.auto_range
                if min is not None and max is not None:
                    step = (max - min) / 10
                else:
                    return
            else:
                return

            if max == min:
                return

            val, step = self.compute_scale(min, max)
            while val <= max:
                self._ticks.append((val, "%.4g" % val, medium, step))
                val += step

    def update_graph(self):
        if self.update_callback:
            self.update_callback()

    def update(self, zoom_only=False):
        self.update_ticks()
        line_color = self.plot.color(OWPalette.Axis)
        text_color = self.plot.color(OWPalette.Text)
        if not self.graph_line or not self.scene():
            return
        self.line_item.setLine(self.graph_line)
        self.line_item.setPen(line_color)
        if self.title:
            self.title_item.setHtml('<b>' + self.title + '</b>')
            self.title_item.setDefaultTextColor(text_color)
        if self.title_location == AxisMiddle:
            title_p = 0.5
        elif self.title_location == AxisEnd:
            title_p = 0.95
        else:
            title_p = 0.05
        title_pos = self.graph_line.pointAt(title_p)
        v = self.graph_line.normalVector().unitVector()

        dense_text = False
        if hasattr(self, 'title_margin'):
            offset = self.title_margin
        elif self._ticks:
            if self.should_be_expanded():
                offset = 55
                dense_text = True
            else:
                offset = 35
        else:
            offset = 10

        if self.title_above:
            title_pos += (v.p2() - v.p1()) * (offset + QFontMetrics(self.title_item.font()).height())
        else:
            title_pos -= (v.p2() - v.p1()) * offset
            ## TODO: Move it according to self.label_pos
        self.title_item.setVisible(self.show_title)
        self.title_item.setRotation(-self.graph_line.angle())
        c = self.title_item.mapToParent(self.title_item.boundingRect().center())
        tl = self.title_item.mapToParent(self.title_item.boundingRect().topLeft())
        self.title_item.setPos(title_pos - c + tl)

        ## Arrows
        if not zoom_only:
            if self.start_arrow_item:
                self.scene().removeItem(self.start_arrow_item)
                self.start_arrow_item = None
            if self.end_arrow_item:
                self.scene().removeItem(self.end_arrow_item)
                self.end_arrow_item = None

        if self.arrows & AxisStart:
            if not zoom_only or not self.start_arrow_item:
                self.start_arrow_item = QGraphicsPathItem(self.arrow_path, self)
            self.start_arrow_item.setPos(self.graph_line.p1())
            self.start_arrow_item.setRotation(-self.graph_line.angle() + 180)
            self.start_arrow_item.setBrush(line_color)
            self.start_arrow_item.setPen(line_color)
        if self.arrows & AxisEnd:
            if not zoom_only or not self.end_arrow_item:
                self.end_arrow_item = QGraphicsPathItem(self.arrow_path, self)
            self.end_arrow_item.setPos(self.graph_line.p2())
            self.end_arrow_item.setRotation(-self.graph_line.angle())
            self.end_arrow_item.setBrush(line_color)
            self.end_arrow_item.setPen(line_color)

        ## Labels

        n = len(self._ticks)
        resize_plot_item_list(self.label_items, n, QGraphicsTextItem, self)
        resize_plot_item_list(self.label_bg_items, n, QGraphicsRectItem, self)
        resize_plot_item_list(self.tick_items, n, QGraphicsLineItem, self)

        test_rect = QRectF(self.graph_line.p1(), self.graph_line.p2()).normalized()
        test_rect.adjust(-1, -1, 1, 1)

        n_v = self.graph_line.normalVector().unitVector()
        if self.title_above:
            n_p = n_v.p2() - n_v.p1()
        else:
            n_p = n_v.p1() - n_v.p2()
        l_v = self.graph_line.unitVector()
        l_p = l_v.p2() - l_v.p1()
        for i in range(n):
            pos, text, size, step = self._ticks[i]
            hs = 0.5 * step
            tick_pos = self.map_to_graph(pos)
            if not test_rect.contains(tick_pos):
                self.tick_items[i].setVisible(False)
                self.label_items[i].setVisible(False)
                continue
            item = self.label_items[i]
            item.setVisible(True)
            if not zoom_only:
                if self.id in XAxes or getattr(self, 'is_horizontal', False):
                    item.setHtml('<center>' + Qt.escape(text.strip()) + '</center>')
                else:
                    item.setHtml(Qt.escape(text.strip()))

            item.setTextWidth(-1)
            text_angle = 0
            if dense_text:
                w = min(item.boundingRect().width(), self.max_text_width)
                item.setTextWidth(w)
                if self.title_above:
                    label_pos = tick_pos + n_p * (w + self.text_margin) + l_p * item.boundingRect().height() / 2
                else:
                    label_pos = tick_pos + n_p * self.text_margin + l_p * item.boundingRect().height() / 2
                text_angle = -90 if self.title_above else 90
            else:
                w = min(item.boundingRect().width(),
                        QLineF(self.map_to_graph(pos - hs), self.map_to_graph(pos + hs)).length())
                label_pos = tick_pos + n_p * self.text_margin + l_p * item.boundingRect().height() / 2
                item.setTextWidth(w)

            if not self.always_horizontal_text:
                if self.title_above:
                    item.setRotation(-self.graph_line.angle() - text_angle)
                else:
                    item.setRotation(self.graph_line.angle() - text_angle)

            item.setPos(label_pos)
            item.setDefaultTextColor(text_color)

            self.label_bg_items[i].setRect(item.boundingRect())
            self.label_bg_items[i].setPen(QPen(Qt.NoPen))
            self.label_bg_items[i].setBrush(self.plot.color(OWPalette.Canvas))

            item = self.tick_items[i]
            item.setVisible(True)
            tick_line = QLineF(v)
            tick_line.translate(-tick_line.p1())
            tick_line.setLength(size)
            if self.title_above:
                tick_line.setAngle(tick_line.angle() + 180)
            item.setLine(tick_line)
            item.setPen(line_color)
            item.setPos(self.map_to_graph(pos))

    @staticmethod
    def make_title(label, unit=None):
        lab = '<i>' + label + '</i>'
        if unit:
            lab = lab + ' [' + unit + ']'
        return lab

    def set_line(self, line):
        self.graph_line = line
        self.update()

    def set_title(self, title):
        self.title = title
        self.update()

    def set_show_title(self, b):
        self.show_title = b
        self.update()

    def set_labels(self, labels):
        self.labels = labels
        self.graph_line = None
        self.auto_scale = False
        self.update_ticks()
        self.update_graph()

    def set_scale(self, min, max, step_size):
        self.scale = (min, max, step_size)
        self.graph_line = None
        self.auto_scale = False
        self.update_ticks()
        self.update_graph()

    def set_tick_length(self, minor, medium, major):
        self.tick_length = (minor, medium, major)
        self.update()

    def map_to_graph(self, x):
        min, max = self.plot.bounds_for_axis(self.id)
        if min == max:
            return QPointF()
        line_point = self.graph_line.pointAt((x - min) / (max - min))
        end_point = line_point * self.zoom_transform
        return self.projection(end_point, self.graph_line)

    @staticmethod
    def projection(point, line):
        norm = line.normalVector()
        norm.translate(point - norm.p1())
        p = QPointF()
        type = line.intersect(norm, p)
        return p

    def continuous_labels(self):
        min, max, step = self.scale
        magnitude = log10(abs(max - min))

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()

    def ticks(self):
        if not self._ticks:
            self.update_ticks()
        return self._ticks

    def bounds(self):
        if self.labels:
            return -0.2, len(self.labels) - 0.8
        elif self.scale:
            min, max, _step = self.scale
            return min, max
        elif self.auto_range:
            return self.auto_range
        else:
            return 0, 1

    def should_be_expanded(self):
        self.update_ticks()
        return self.id in YAxes or self.always_horizontal_text or sum(
            len(t[1]) for t in self._ticks) * 12 > self.plot.width()

