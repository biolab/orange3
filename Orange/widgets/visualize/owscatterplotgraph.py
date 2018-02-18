from collections import Counter
import sys
import itertools
from xml.sax.saxutils import escape
from math import log10, floor, ceil

import numpy as np
from scipy.stats import linregress

from AnyQt.QtCore import Qt, QObject, QEvent, QRectF, QPointF, QSize
from AnyQt.QtGui import (
    QStaticText, QColor, QPen, QBrush, QPainterPath, QTransform, QPainter, QKeySequence)
from AnyQt.QtWidgets import QApplication, QToolTip, QPinchGesture, \
    QGraphicsTextItem, QGraphicsRectItem, QAction

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
import pyqtgraph.graphicsItems.ScatterPlotItem
from pyqtgraph.graphicsItems.LegendItem import LegendItem, ItemSample
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
from pyqtgraph.graphicsItems.TextItem import TextItem
from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine
from pyqtgraph.Point import Point

from Orange.widgets import gui
from Orange.widgets.utils import classdensity, get_variable_values_sorted
from Orange.widgets.utils.colorpalette import (ColorPaletteGenerator,
                                               ContinuousPaletteGenerator, DefaultRGBColors)
from Orange.widgets.utils.plot import \
    OWPalette, OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.scaling import ScaleScatterPlotData
from Orange.widgets.settings import Setting, ContextSetting


# TODO Move utility classes to another module, so they can be used elsewhere

SELECTION_WIDTH = 5
MAX = 11  # maximum number of colors or shapes (including Other)

class PaletteItemSample(ItemSample):
    """A color strip to insert into legends for discretized continuous values"""

    def __init__(self, palette, scale):
        """
        :param palette: palette used for showing continuous values
        :type palette: ContinuousPaletteGenerator
        :param scale: an instance of DiscretizedScale that defines the
                      conversion of values into bins
        :type scale: DiscretizedScale
        """
        super().__init__(None)
        self.palette = palette
        self.scale = scale
        cuts = ["{0:.{1}f}".format(scale.offset + i * scale.width, scale.decimals)
                for i in range(scale.bins + 1)]
        self.labels = [QStaticText("{} - {}".format(fr, to))
                       for fr, to in zip(cuts, cuts[1:])]
        for label in self.labels:
            label.prepare()
        self.text_width = max(label.size().width() for label in self.labels)

    def boundingRect(self):
        return QRectF(0, 0, 40 + self.text_width, 20 + self.scale.bins * 15)

    def paint(self, p, *args):
        p.setRenderHint(p.Antialiasing)
        scale = self.scale
        palette = self.palette
        font = p.font()
        font.setPixelSize(11)
        p.setFont(font)
        for i, label in enumerate(self.labels):
            color = QColor(*palette.getRGB((i + 0.5) / scale.bins))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRect(0, i * 15, 15, 15)
            p.setPen(QPen(Qt.black))
            p.drawStaticText(20, i * 15 + 1, label)


class LegendItem(LegendItem):
    def __init__(self, size=None, offset=None, pen=None, brush=None):
        super().__init__(size, offset)

        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setHorizontalSpacing(15)
        self.layout.setColumnAlignment(1, Qt.AlignLeft | Qt.AlignVCenter)

        if pen is None:
            pen = QPen(QColor(196, 197, 193, 200), 1)
            pen.setCosmetic(True)
        self.__pen = pen

        if brush is None:
            brush = QBrush(QColor(232, 232, 232, 100))
        self.__brush = brush

    def storeAnchor(self):
        """
        Return the current relative anchor position (relative to the parent)
        """
        anchor = legend_anchor_pos(self)
        if anchor is None:
            anchor = ((1, 0), (1, 0))
        return anchor

    def restoreAnchor(self, anchors):
        """
        Restore (parent) relative position from stored anchors.

        The restored position is within the parent bounds.
        """
        anchor, parentanchor = anchors
        self.anchor(*bound_anchor_pos(anchor, parentanchor))

    def setPen(self, pen):
        """Set the legend frame pen."""
        pen = QPen(pen)
        if pen != self.__pen:
            self.prepareGeometryChange()
            self.__pen = pen
            self.updateGeometry()

    def pen(self):
        """Pen used to draw the legend frame."""
        return QPen(self.__pen)

    def setBrush(self, brush):
        """Set background brush"""
        brush = QBrush(brush)
        if brush != self.__brush:
            self.__brush = brush
            self.update()

    def brush(self):
        """Background brush."""
        return QBrush(self._brush)

    def paint(self, painter, option, widget=None):
        painter.setPen(self.__pen)
        painter.setBrush(self.__brush)
        rect = self.contentsRect()
        painter.drawRoundedRect(rect, 2, 2)

    def addItem(self, item, name):
        super().addItem(item, name)
        # Fix-up the label alignment
        _, label = self.items[-1]
        label.setText(name, justify="left")

    def clear(self):
        """
        Clear all legend items.
        """
        items = list(self.items)
        self.items = []
        for sample, label in items:
            self.layout.removeItem(sample)
            self.layout.removeItem(label)
            sample.hide()
            label.hide()

        self.updateSize()


ANCHORS = {
    Qt.TopLeftCorner: (0, 0),
    Qt.TopRightCorner: (1, 0),
    Qt.BottomLeftCorner: (0, 1),
    Qt.BottomRightCorner: (1, 1)
}


def corner_anchor(corner):
    """Return the relative corner coordinates for Qt.Corner
    """
    return ANCHORS[corner]


def legend_anchor_pos(legend):
    """
    Return the legend's anchor positions relative to it's parent (if defined).

    Return `None` if legend does not have a parent or the parent's size
    is empty.

    .. seealso:: LegendItem.anchor, rect_anchor_pos

    """
    parent = legend.parentItem()
    if parent is None or parent.size().isEmpty():
        return None

    rect = legend.geometry()  # in parent coordinates.
    parent_rect = QRectF(QPointF(0, 0), parent.size())

    # Find the closest corner of rect to parent rect
    c1, _, *parentPos = rect_anchor_pos(rect, parent_rect)
    return corner_anchor(c1), tuple(parentPos)


def bound_anchor_pos(corner, parentpos):
    corner = np.clip(corner, 0, 1)
    parentpos = np.clip(parentpos, 0, 1)

    irx, iry = corner
    prx, pry = parentpos

    if irx > 0.9 and prx < 0.1:
        irx = prx = 0.0
    if iry > 0.9 and pry < 0.1:
        iry = pry = 0.0
    if irx < 0.1 and prx > 0.9:
        irx = prx = 1.0
    if iry < 0.1 and pry > 0.9:
        iry = pry = 1.0
    return (irx, iry), (prx, pry)


def rect_anchor_pos(rect, parent_rect):
    """
    Find the 'best' anchor corners of rect within parent_rect.

    Return a tuple of (rect_corner, parent_corner, rx, ry),
    where rect/parent_corners are Qt.Corners which are closest and
    rx, ry are the relative positions of the rect_corner within
    parent_rect. If the parent_rect is empty return `None`.

    """
    if parent_rect.isEmpty():
        return None

    # Find the closest corner of rect to parent rect
    corners = (Qt.TopLeftCorner, Qt.TopRightCorner,
               Qt.BottomRightCorner, Qt.BottomLeftCorner)

    def rect_corner(rect, corner):
        if corner == Qt.TopLeftCorner:
            return rect.topLeft()
        elif corner == Qt.TopRightCorner:
            return rect.topRight()
        elif corner == Qt.BottomLeftCorner:
            return rect.bottomLeft()
        elif corner == Qt.BottomRightCorner:
            return rect.bottomRight()
        else:
            assert False

    def corner_dist(c1, c2):
        d = (rect_corner(rect, c1) - rect_corner(parent_rect, c2))
        return d.x() ** 2 + d.y() ** 2

    if parent_rect.contains(rect):
        closest = min(corners,
                      key=lambda corner: corner_dist(corner, corner))
        p = rect_corner(rect, closest)

        return (closest, closest,
                (p.x() - parent_rect.left()) / parent_rect.width(),
                (p.y() - parent_rect.top()) / parent_rect.height())
    else:

        c1, c2 = min(itertools.product(corners, corners),
                     key=lambda pair: corner_dist(*pair))

        p = rect_corner(rect, c1)

        return (c1, c2,
                (p.x() - parent_rect.left()) / parent_rect.width(),
                (p.y() - parent_rect.top()) / parent_rect.height())


class DiscretizedScale:
    """
    Compute suitable bins for continuous value from its minimal and
    maximal value.

    The width of the bin is a power of 10 (including negative powers).
    The minimal value is rounded up and the maximal is rounded down. If this
    gives less than 3 bins, the width is divided by four; if it gives
    less than 6, it is halved.

    .. attribute:: offset
        The start of the first bin.

    .. attribute:: width
        The width of the bins

    .. attribute:: bins
        The number of bins

    .. attribute:: decimals
        The number of decimals used for printing out the boundaries
    """
    def __init__(self, min_v, max_v):
        """
        :param min_v: Minimal value
        :type min_v: float
        :param max_v: Maximal value
        :type max_v: float
        """
        super().__init__()
        dif = max_v - min_v if max_v != min_v else 1
        if np.isnan(dif):
            min_v = 0
            dif = decimals = 1
        else:
            decimals = -floor(log10(dif))
        resolution = 10 ** -decimals
        bins = ceil(dif / resolution)
        if bins < 6:
            decimals += 1
            if bins < 3:
                resolution /= 4
            else:
                resolution /= 2
            bins = ceil(dif / resolution)
        self.offset = resolution * floor(min_v // resolution)
        self.bins = bins
        self.decimals = max(decimals, 0)
        self.width = resolution

    def compute_bins(self, a):
        """
        Compute bin number(s) for the given value(s).

        :param a: value(s)
        :type a: a number or numpy.ndarray
        """
        a = (a - self.offset) / self.width
        if isinstance(a, np.ndarray):
            a.clip(0, self.bins - 1)
        else:
            a = min(self.bins - 1, max(0, a))
        return a


class InteractiveViewBox(ViewBox):
    def __init__(self, graph, enable_menu=False):
        self.init_history()
        ViewBox.__init__(self, enableMenu=enable_menu)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.grabGesture(Qt.PinchGesture)

    def _dragtip_pos(self):
        return 10, self.height()

    def safe_update_scale_box(self, buttonDownPos, currentPos):
        x, y = currentPos
        if buttonDownPos[0] == x:
            x += 1
        if buttonDownPos[1] == y:
            y += 1
        self.updateScaleBox(buttonDownPos, Point(x, y))

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


class ScatterPlotItem(pg.ScatterPlotItem):
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        super().paint(painter, option, widget)


def _define_symbols():
    """
    Add symbol ? to ScatterPlotItemSymbols,
    reflect the triangle to point upwards
    """
    symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols
    path = QPainterPath()
    path.addEllipse(QRectF(-0.35, -0.35, 0.7, 0.7))
    path.moveTo(-0.5, 0.5)
    path.lineTo(0.5, -0.5)
    path.moveTo(-0.5, -0.5)
    path.lineTo(0.5, 0.5)
    symbols["?"] = path

    tr = QTransform()
    tr.rotate(180)
    symbols['t'] = tr.map(symbols['t'])

_define_symbols()


def _make_pen(color, width):
    p = QPen(color, width)
    p.setCosmetic(True)
    return p


class OWScatterPlotGraph(gui.OWComponent, ScaleScatterPlotData):
    attr_color = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_label = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_shape = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_size = ContextSetting(None, required=ContextSetting.OPTIONAL)
    label_only_selected = Setting(False)

    point_width = Setting(10)
    alpha_value = Setting(128)
    show_grid = Setting(False)
    show_legend = Setting(True)
    tooltip_shows_all = Setting(False)
    class_density = Setting(False)
    show_reg_line = Setting(False)
    resolution = 256

    CurveSymbols = np.array("o x t + d s t2 t3 p h star ?".split())
    MinShapeSize = 6
    DarkerValue = 120
    UnknownColor = (168, 50, 168)

    def __init__(self, scatter_widget, parent=None, _="None", view_box=InteractiveViewBox):
        gui.OWComponent.__init__(self, scatter_widget)
        self.view_box = view_box(self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, parent=parent,
                                         background="w")
        self.plot_widget.getPlotItem().buttonsHidden = True
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.sizeHint = lambda: QSize(500, 500)
        scene = self.plot_widget.scene()
        self._create_drag_tooltip(scene)
        self._data = None  # Original Table as passed from widget to new_data before transformations

        self.replot = self.plot_widget.replot
        ScaleScatterPlotData.__init__(self)
        self.density_img = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None
        self.reg_line_item = None

        self.labels = []

        self.master = scatter_widget
        self.master.Warning.add_message(
            "missing_coords",
            "Plot cannot be displayed because '{}' or '{}' is missing for "
            "all data points")
        self.master.Information.add_message(
            "missing_coords",
            "Points with missing '{}' or '{}' are not displayed")
        self.master.Information.add_message(
            "missing_size",
            "Points with undefined '{}' are shown in smaller size")
        self.master.Information.add_message(
            "missing_shape",
            "Points with undefined '{}' are shown as crossed circles")
        self.shown_attribute_indices = []
        self.shown_x = self.shown_y = None
        self.pen_colors = self.brush_colors = None

        self.valid_data = None  # np.ndarray
        self.selection = None  # np.ndarray
        self.n_points = 0

        self.gui = OWPlotGUI(self)
        self.continuous_palette = ContinuousPaletteGenerator(
            QColor(255, 255, 0), QColor(0, 0, 255), True)
        self.discrete_palette = ColorPaletteGenerator()

        self.selection_behavior = 0

        self.legend = self.color_legend = None
        self.__legend_anchor = (1, 0), (1, 0)
        self.__color_legend_anchor = (1, 1), (1, 1)

        self.scale = None  # DiscretizedScale

        self.subset_indices = None

        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.update_grid()

        self._tooltip_delegate = HelpEventDelegate(self.help_event)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)

    def _create_drag_tooltip(self, scene):
        tip_parts = [
            (Qt.ShiftModifier, "Shift: Add group"),
            (Qt.ShiftModifier + Qt.ControlModifier,
             "Shift-{}: Append to group".
             format("Cmd" if sys.platform == "darwin" else "Ctrl")),
            (Qt.AltModifier, "Alt: Remove")
        ]
        all_parts = ", ".join(part for _, part in tip_parts)
        self.tiptexts = {
            int(modifier): all_parts.replace(part, "<b>{}</b>".format(part))
            for modifier, part in tip_parts
        }
        self.tiptexts[0] = all_parts

        self.tip_textitem = text = QGraphicsTextItem()
        # Set to the longest text
        text.setHtml(self.tiptexts[Qt.ShiftModifier + Qt.ControlModifier])
        text.setPos(4, 2)
        r = text.boundingRect()
        rect = QGraphicsRectItem(0, 0, r.width() + 8, r.height() + 4)
        rect.setBrush(QColor(224, 224, 224, 212))
        rect.setPen(QPen(Qt.NoPen))
        self.update_tooltip(Qt.NoModifier)

        scene.drag_tooltip = scene.createItemGroup([rect, text])
        scene.drag_tooltip.hide()

    def update_tooltip(self, modifiers):
        modifiers &= Qt.ShiftModifier + Qt.ControlModifier + Qt.AltModifier
        text = self.tiptexts.get(int(modifiers), self.tiptexts[0])
        self.tip_textitem.setHtml(text)

    def new_data(self, data, subset_data=None, new=True, **args):
        if new:
            self.plot_widget.clear()
            self.remove_legend()

            self.density_img = None
            self.scatterplot_item = None
            self.scatterplot_item_sel = None
            self.reg_line_item = None
            self.labels = []
            self.selection = None
            self.valid_data = None

        self.subset_indices = set(e.id for e in subset_data) if subset_data else None

        self._data = data
        data = self.sparse_to_dense()
        self.set_data(data, **args)

    def set_domain(self, data):
        domain = data.domain if data and len(data) else None
        for attr in ("attr_color", "attr_shape", "attr_size", "attr_label"):
            getattr(self.controls, attr).model().set_domain(domain)
            setattr(self, attr, None)
        if domain is not None:
            self.attr_color = domain.class_var

    def sparse_to_dense(self):
        data = self._data
        if data is None or not data.is_sparse():
            return data

        attrs = {self.shown_x,
                 self.shown_y,
                 self.attr_color,
                 self.attr_shape,
                 self.attr_size,
                 self.attr_label}
        domain = data.domain
        all_attrs = domain.variables + domain.metas
        attrs = list(set(all_attrs) & attrs)
        selected_data = data[:, attrs].to_dense()
        return selected_data

    def _clear_plot_widget(self):
        self.remove_legend()
        if self.density_img:
            self.plot_widget.removeItem(self.density_img)
            self.density_img = None
        if self.scatterplot_item:
            self.plot_widget.removeItem(self.scatterplot_item)
            self.scatterplot_item = None
        if self.scatterplot_item_sel:
            self.plot_widget.removeItem(self.scatterplot_item_sel)
            self.scatterplot_item_sel = None
        if self.reg_line_item:
            self.plot_widget.removeItem(self.reg_line_item)
            self.reg_line_item = None
        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []
        self.set_axis_title("bottom", "")
        self.set_axis_title("left", "")

    def update_data(self, attr_x, attr_y, reset_view=True):
        self.master.Warning.missing_coords.clear()
        self.master.Information.missing_coords.clear()
        self._clear_plot_widget()

        if self.shown_y != attr_y:
            # 'reset' the axis text width estimation. Without this the left
            # axis tick labels space only ever expands
            yaxis = self.plot_widget.getAxis("left")
            yaxis.textWidth = 30

        self.shown_x, self.shown_y = attr_x, attr_y
        if attr_x not in self.data.domain or attr_y not in self.data.domain:
            data = self.sparse_to_dense()
            self.set_data(data)

        if self.jittered_data is None or not len(self.jittered_data):
            self.valid_data = None
        else:
            self.valid_data = self.get_valid_list([attr_x, attr_y])
            if not np.any(self.valid_data):
                self.valid_data = None
        if self.valid_data is None:
            self.selection = None
            self.n_points = 0
            self.master.Warning.missing_coords(
                self.shown_x.name, self.shown_y.name)
            return

        x_data, y_data = self.get_xy_data_positions(
            attr_x, attr_y, self.valid_data)
        self.n_points = len(x_data)

        if reset_view:
            min_x, max_x = np.nanmin(x_data), np.nanmax(x_data)
            min_y, max_y = np.nanmin(y_data), np.nanmax(y_data)
            self.view_box.setRange(
                QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
                padding=0.025)
            self.view_box.init_history()
            self.view_box.tag_history()
        [min_x, max_x], [min_y, max_y] = self.view_box.viewRange()

        for axis, var in (("bottom", attr_x), ("left", attr_y)):
            self.set_axis_title(axis, var)
            if var.is_discrete:
                self.set_labels(axis, get_variable_values_sorted(var))
            else:
                self.set_labels(axis, None)

        color_data, brush_data = self.compute_colors()
        color_data_sel, brush_data_sel = self.compute_colors_sel()
        size_data = self.compute_sizes()
        shape_data = self.compute_symbols()

        if self.should_draw_density():
            rgb_data = [pen.color().getRgb()[:3] for pen in color_data]
            self.density_img = classdensity.class_density_image(
                min_x, max_x, min_y, max_y, self.resolution,
                x_data, y_data, rgb_data)
            self.plot_widget.addItem(self.density_img)

        self.data_indices = np.flatnonzero(self.valid_data)
        if len(self.data_indices) != len(self.data):
            self.master.Information.missing_coords(
                self.shown_x.name, self.shown_y.name)

        self.scatterplot_item = ScatterPlotItem(
            x=x_data, y=y_data, data=self.data_indices,
            symbol=shape_data, size=size_data, pen=color_data, brush=brush_data
        )
        self.scatterplot_item_sel = ScatterPlotItem(
            x=x_data, y=y_data, data=self.data_indices,
            symbol=shape_data, size=size_data + SELECTION_WIDTH,
            pen=color_data_sel, brush=brush_data_sel
        )
        self.plot_widget.addItem(self.scatterplot_item_sel)
        self.plot_widget.addItem(self.scatterplot_item)

        self.scatterplot_item.selected_points = []
        self.scatterplot_item.sigClicked.connect(self.select_by_click)

        self.draw_regression_line(x_data, y_data, min_x, max_x)
        self.update_labels()
        self.make_legend()
        self.plot_widget.replot()

    def draw_regression_line(self, x_data, y_data, min_x, max_x):
        if self.show_reg_line and self.can_draw_regresssion_line():
            slope, intercept, rvalue, _, _ = linregress(x_data, y_data)
            start_y = min_x * slope + intercept
            end_y = max_x * slope + intercept
            angle = np.degrees(np.arctan((end_y - start_y) / (max_x - min_x)))
            rotate = ((angle + 45) % 180) - 45 > 90
            color = QColor("#505050")
            l_opts = dict(color=color, position=abs(int(rotate) - 0.85),
                          rotateAxis=(1, 0), movable=True)
            self.reg_line_item = InfiniteLine(
                pos=QPointF(min_x, start_y), pen=pg.mkPen(color=color, width=1),
                angle=angle, label="r = {:.2f}".format(rvalue), labelOpts=l_opts)
            if rotate:
                self.reg_line_item.label.angle = 180
                self.reg_line_item.label.updateTransform()
            self.plot_widget.addItem(self.reg_line_item)

    def can_draw_density(self):
        return self.domain is not None and \
            self.attr_color is not None and \
            self.attr_color.is_discrete and \
            self.shown_x.is_continuous and \
            self.shown_y.is_continuous

    def should_draw_density(self):
        return self.class_density and self.n_points > 1 and self.can_draw_density()

    def can_draw_regresssion_line(self):
        return self.domain is not None and \
               self.shown_x.is_continuous and \
               self.shown_y.is_continuous

    def set_labels(self, axis, labels):
        axis = self.plot_widget.getAxis(axis)
        if labels:
            ticks = [[(i, labels[i]) for i in range(len(labels))]]
            axis.setTicks(ticks)
        else:
            axis.setTicks(None)

    def set_axis_title(self, axis, title):
        self.plot_widget.setLabel(axis=axis, text=title)

    def compute_sizes(self):
        self.master.Information.missing_size.clear()
        if self.attr_size is None:
            size_data = np.full((self.n_points,), self.point_width,
                                dtype=float)
        else:
            size_data = \
                self.MinShapeSize + \
                self.scaled_data.get_column_view(self.attr_size)[0][self.valid_data] * \
                self.point_width
        nans = np.isnan(size_data)
        if np.any(nans):
            size_data[nans] = self.MinShapeSize - 2
            self.master.Information.missing_size(self.attr_size)
        return size_data

    def update_sizes(self):
        self.set_data(self.sparse_to_dense())
        self.update_point_size()

    def update_point_size(self):
        if self.scatterplot_item:
            size_data = self.compute_sizes()
            self.scatterplot_item.setSize(size_data)
            self.scatterplot_item_sel.setSize(size_data + SELECTION_WIDTH)

    def get_color(self):
        if self.attr_color is None:
            return None
        colors = self.attr_color.colors
        if self.attr_color.is_discrete:
            self.discrete_palette = ColorPaletteGenerator(
                number_of_colors=min(len(colors), MAX), rgb_colors=colors if len(colors) <= MAX
                else DefaultRGBColors)
        else:
            self.continuous_palette = ContinuousPaletteGenerator(*colors)
        return self.attr_color

    def compute_colors_sel(self, keep_colors=False):
        if not keep_colors:
            self.pen_colors_sel = self.brush_colors_sel = None

        nopen = QPen(Qt.NoPen)
        if self.selection is not None:
            sels = np.max(self.selection)
            if sels == 1:
                pens = [nopen,
                        _make_pen(QColor(255, 190, 0, 255),
                                  SELECTION_WIDTH + 1.)]
            else:
                # Start with the first color so that the colors of the
                # additional attribute in annotation (which start with 0,
                # unselected) will match these colors
                palette = ColorPaletteGenerator(number_of_colors=sels + 1)
                pens = [nopen] + \
                       [_make_pen(palette[i + 1], SELECTION_WIDTH + 1.)
                        for i in range(sels)]
            pen = [pens[a] for a in self.selection[self.valid_data]]
        else:
            pen = [nopen] * self.n_points
        brush = [QBrush(QColor(255, 255, 255, 0))] * self.n_points
        return pen, brush

    def _reduce_values(self, attr):
        """
        If discrete variable has more than maximium allowed values,
        less used values are joined as "Other"
        """
        c_data = self.data.get_column_view(attr)[0][self.valid_data]
        if attr.is_continuous or len(attr.values) <= MAX:
            return None, c_data
        values_to_replace = Counter(c_data)
        values_to_replace = sorted(
            values_to_replace, key=values_to_replace.get, reverse=True
        )
        return values_to_replace, c_data

    def _get_values(self, attr):
        if len(attr.values) <= MAX:
            return attr.values
        values_to_replace, _ = self._reduce_values(attr)
        return [attr.values[int(i)] for i in values_to_replace
                if not np.isnan(i)][:MAX - 1] + ["Other"]

    def _get_data(self, attr):
        values_to_replace, c_data = self._reduce_values(attr)
        if values_to_replace is not None:
            c_data_2 = c_data.copy()
            for i, v in enumerate(values_to_replace):
                c_data[c_data_2 == v] = i if i < MAX - 1 else MAX - 1
        return c_data

    def compute_colors(self, keep_colors=False):
        if not keep_colors:
            self.pen_colors = self.brush_colors = None
        self.get_color()

        subset = None
        if self.subset_indices:
            subset = np.array([ex.id in self.subset_indices
                               for ex in self.data[self.valid_data]])

        if self.attr_color is None:  # same color
            color = self.plot_widget.palette().color(OWPalette.Data)
            pen = [_make_pen(color, 1.5)] * self.n_points
            if subset is not None:
                brush = [(QBrush(QColor(128, 128, 128, 0)),
                          QBrush(QColor(128, 128, 128, 255)))[s]
                         for s in subset]
            else:
                brush = [QBrush(QColor(128, 128, 128, self.alpha_value))] \
                        * self.n_points
            return pen, brush

        c_data = self._get_data(self.attr_color)
        if self.attr_color.is_continuous:
            if self.pen_colors is None:
                self.scale = DiscretizedScale(np.nanmin(c_data), np.nanmax(c_data))
                c_data -= self.scale.offset
                c_data /= self.scale.width
                c_data = np.floor(c_data) + 0.5
                c_data /= self.scale.bins
                c_data = np.clip(c_data, 0, 1)
                palette = self.continuous_palette
                self.pen_colors = palette.getRGB(c_data)
                self.brush_colors = np.hstack(
                    [self.pen_colors,
                     np.full((self.n_points, 1), self.alpha_value, dtype=int)])
                self.pen_colors *= 100
                self.pen_colors //= self.DarkerValue
                self.pen_colors = [_make_pen(QColor(*col), 1.5)
                                   for col in self.pen_colors.tolist()]
            if subset is not None:
                self.brush_colors[:, 3] = 0
                self.brush_colors[subset, 3] = 255
            else:
                self.brush_colors[:, 3] = self.alpha_value
            pen = self.pen_colors
            brush = np.array([QBrush(QColor(*col))
                              for col in self.brush_colors.tolist()])
        else:
            if self.pen_colors is None:
                palette = self.discrete_palette
                n_colors = palette.number_of_colors
                c_data = c_data.copy()
                c_data[np.isnan(c_data)] = n_colors
                c_data = c_data.astype(int)
                colors = np.r_[palette.getRGB(np.arange(n_colors)),
                               [[128, 128, 128]]]
                pens = np.array(
                    [_make_pen(QColor(*col).darker(self.DarkerValue), 1.5)
                     for col in colors])
                self.pen_colors = pens[c_data]
                alpha = self.alpha_value if subset is None else 255
                self.brush_colors = np.array([
                    [QBrush(QColor(0, 0, 0, 0)),
                     QBrush(QColor(col[0], col[1], col[2], alpha))]
                    for col in colors])
                self.brush_colors = self.brush_colors[c_data]
            if subset is not None:
                brush = np.where(
                    subset,
                    self.brush_colors[:, 1], self.brush_colors[:, 0])
            else:
                brush = self.brush_colors[:, 1]
            pen = self.pen_colors
        return pen, brush

    def update_colors(self, keep_colors=False):
        self.master.update_colors()
        self.set_data(self.sparse_to_dense())
        self.update_alpha_value(keep_colors)

    def update_alpha_value(self, keep_colors=False):
        if self.scatterplot_item:
            pen_data, brush_data = self.compute_colors(keep_colors)
            pen_data_sel, brush_data_sel = self.compute_colors_sel(keep_colors)
            self.scatterplot_item.setPen(pen_data, update=False, mask=None)
            self.scatterplot_item.setBrush(brush_data, mask=None)
            self.scatterplot_item_sel.setPen(pen_data_sel, update=False, mask=None)
            self.scatterplot_item_sel.setBrush(brush_data_sel, mask=None)
            if not keep_colors:
                self.make_legend()

                if self.should_draw_density():
                    self.update_data(self.shown_x, self.shown_y)
                elif self.density_img:
                    self.plot_widget.removeItem(self.density_img)

    def create_labels(self):
        for x, y in zip(*self.scatterplot_item.getData()):
            ti = TextItem()
            self.plot_widget.addItem(ti)
            ti.setPos(x, y)
            self.labels.append(ti)

    def _create_label_column(self):
        if self.attr_label in self.data.domain:
            label_column = self.data.get_column_view(self.attr_label)[0]
        else:
            label_column = self.master.data.get_column_view(self.attr_label)[0]
        return label_column[self.data_indices]


    def update_labels(self):
        if self.attr_label is None or \
                self.label_only_selected and self.selection is None:
            for label in self.labels:
                label.setText("")
            return
        self.assure_attribute_present(self.attr_label)
        if not self.labels:
            self.create_labels()
        label_column = self._create_label_column()
        formatter = self.attr_label.str_val
        label_data = map(formatter, label_column)
        black = pg.mkColor(0, 0, 0)
        selection = self.selection[self.valid_data] if self.selection is not None else []
        if self.label_only_selected:
            for label, text, selected \
                    in zip(self.labels, label_data, selection):
                label.setText(text if selected else "", black)
        else:
            for label, text in zip(self.labels, label_data):
                label.setText(text, black)

    def compute_symbols(self):
        self.master.Information.missing_shape.clear()
        if self.attr_shape is None:
            shape_data = self.CurveSymbols[np.zeros(self.n_points, dtype=int)]
        else:
            shape_data = self._get_data(self.attr_shape)
            nans = np.isnan(shape_data)
            if np.any(nans):
                shape_data[nans] = len(self.CurveSymbols) - 1
                self.master.Information.missing_shape(self.attr_shape)
            shape_data = self.CurveSymbols[shape_data.astype(int)]
        return shape_data

    def update_shapes(self):
        self.assure_attribute_present(self.attr_shape)
        if self.scatterplot_item:
            shape_data = self.compute_symbols()
            self.scatterplot_item.setSymbol(shape_data)
        self.make_legend()

    def assure_attribute_present(self, attr):
        if self.data is not None and attr not in self.data.domain:
            self.set_data(self.sparse_to_dense())

    def update_grid(self):
        self.plot_widget.showGrid(x=self.show_grid, y=self.show_grid)

    def update_legend(self):
        if self.legend:
            self.legend.setVisible(self.show_legend)

    def create_legend(self):
        self.legend = LegendItem()
        self.legend.setParentItem(self.plot_widget.getViewBox())
        self.legend.restoreAnchor(self.__legend_anchor)

    def remove_legend(self):
        if self.legend:
            anchor = legend_anchor_pos(self.legend)
            if anchor is not None:
                self.__legend_anchor = anchor
            self.legend.setParent(None)
            self.legend = None
        if self.color_legend:
            anchor = legend_anchor_pos(self.color_legend)
            if anchor is not None:
                self.__color_legend_anchor = anchor
            self.color_legend.setParent(None)
            self.color_legend = None

    def make_legend(self):
        self.remove_legend()
        self.make_color_legend()
        self.make_shape_legend()
        self.update_legend()

    def make_color_legend(self):
        if self.attr_color is None:
            return
        use_shape = self.attr_shape == self.get_color()
        if self.attr_color.is_discrete:
            if not self.legend:
                self.create_legend()
            palette = self.discrete_palette
            for i, value in enumerate(self._get_values(self.attr_color)):
                color = QColor(*palette.getRGB(i))
                pen = _make_pen(color.darker(self.DarkerValue), 1.5)
                color.setAlpha(self.alpha_value if self.subset_indices is None else 255)
                brush = QBrush(color)
                self.legend.addItem(
                    ScatterPlotItem(
                        pen=pen, brush=brush, size=10,
                        symbol=self.CurveSymbols[i] if use_shape else "o"),
                    escape(value))
        else:
            legend = self.color_legend = LegendItem()
            legend.setParentItem(self.plot_widget.getViewBox())
            legend.restoreAnchor(self.__color_legend_anchor)

            label = PaletteItemSample(self.continuous_palette, self.scale)
            legend.addItem(label, "")
            legend.setGeometry(label.boundingRect())

    def make_shape_legend(self):
        if self.attr_shape is None or self.attr_shape == self.get_color():
            return
        if not self.legend:
            self.create_legend()
        color = QColor(0, 0, 0)
        color.setAlpha(self.alpha_value)
        for i, value in enumerate(self._get_values(self.attr_shape)):
            self.legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10,
                                symbol=self.CurveSymbols[i]), escape(value))

    def zoom_button_clicked(self):
        self.plot_widget.getViewBox().setMouseMode(
            self.plot_widget.getViewBox().RectMode)

    def pan_button_clicked(self):
        self.plot_widget.getViewBox().setMouseMode(
            self.plot_widget.getViewBox().PanMode)

    def select_button_clicked(self):
        self.plot_widget.getViewBox().setMouseMode(
            self.plot_widget.getViewBox().RectMode)

    def reset_button_clicked(self):
        self.update_data(self.shown_x, self.shown_y, reset_view=True)  # also redraw density image
        # self.view_box.autoRange()

    def select_by_click(self, _, points):
        if self.scatterplot_item is not None:
            self.select(points)

    def select_by_rectangle(self, value_rect):
        if self.scatterplot_item is not None:
            points = [point
                      for point in self.scatterplot_item.points()
                      if value_rect.contains(QPointF(point.pos()))]
            self.select(points)

    def unselect_all(self):
        self.selection = None
        self.update_colors(keep_colors=True)
        if self.label_only_selected:
            self.update_labels()
        self.master.selection_changed()

    def select(self, points):
        # noinspection PyArgumentList
        if self.data is None:
            return
        if self.selection is None:
            self.selection = np.zeros(len(self.data), dtype=np.uint8)
        indices = [p.data() for p in points]
        keys = QApplication.keyboardModifiers()
        # Remove from selection
        if keys & Qt.AltModifier:
            self.selection[indices] = 0
        # Append to the last group
        elif keys & Qt.ShiftModifier and keys & Qt.ControlModifier:
            self.selection[indices] = np.max(self.selection)
        # Create a new group
        elif keys & Qt.ShiftModifier:
            self.selection[indices] = np.max(self.selection) + 1
        # No modifiers: new selection
        else:
            self.selection = np.zeros(len(self.data), dtype=np.uint8)
            self.selection[indices] = 1
        self.update_colors(keep_colors=True)
        if self.label_only_selected:
            self.update_labels()
        self.master.selection_changed()

    def get_selection(self):
        if self.selection is None:
            return np.array([], dtype=np.uint8)
        else:
            return np.flatnonzero(self.selection)

    def set_palette(self, p):
        self.plot_widget.setPalette(p)

    def save_to_file(self, size):
        pass

    def help_event(self, event):
        if self.scatterplot_item is None:
            return False

        domain = self.data.domain
        PARTS = (("Class", "Classes", 4, domain.class_vars),
                 ("Meta", "Metas", 4, domain.metas),
                 ("Feature", "Features", 10, domain.attributes))

        def format_val(var, point_data, bold=False):
            text = escape('{} = {}'.format(var.name, point_data[var]))
            if bold:
                text = "<b>{}</b>".format(text)
            return text

        def show_part(point_data, singular, plural, max_shown, vars):
            cols = [format_val(var, point_data)
                    for var in vars[:max_shown + 2]
                    if vars == domain.class_vars
                    or var not in (self.shown_x, self.shown_y)][:max_shown]
            if not cols:
                return ""
            n_vars = len(vars)
            if n_vars > max_shown:
                cols[-1] = "... and {} others".format(n_vars - max_shown + 1)
            return \
                "<br/><b>{}</b>:<br/>".format(singular if n_vars < 2
                                              else plural) \
                + "<br/>".join(cols)

        def point_data(p):
            point_data = self.data[p.data()]
            text = "<br/>".join(
                format_val(var, point_data, bold=self.tooltip_shows_all)
                for var in (self.shown_x, self.shown_y))
            if self.tooltip_shows_all:
                text += "<br/>" + \
                        "".join(show_part(point_data, *columns)
                                for columns in PARTS)
            return text

        act_pos = self.scatterplot_item.mapFromScene(event.scenePos())
        points = self.scatterplot_item.pointsAt(act_pos)
        if len(points):
            text = "<hr/>".join(point_data(point) for point in points)
            QToolTip.showText(event.screenPos(), text, widget=self.plot_widget)
            return True
        else:
            return False

    def box_zoom_select(self, parent):
        g = self.gui
        box_zoom_select = gui.vBox(parent, "Zoom/Select")
        zoom_select_toolbar = g.zoom_select_toolbar(
            box_zoom_select, nomargin=True,
            buttons=[g.StateButtonsBegin, g.SimpleSelect, g.Pan, g.Zoom,
                     g.StateButtonsEnd, g.ZoomReset]
        )
        buttons = zoom_select_toolbar.buttons
        buttons[g.Zoom].clicked.connect(self.zoom_button_clicked)
        buttons[g.Pan].clicked.connect(self.pan_button_clicked)
        buttons[g.SimpleSelect].clicked.connect(self.select_button_clicked)
        buttons[g.ZoomReset].clicked.connect(self.reset_button_clicked)
        return box_zoom_select

    def zoom_actions(self, parent):
        def zoom(s):
            """
            Zoom in/out by factor `s`.
            scaleBy scales the view's bounds (the axis range)
            """
            self.view_box.scaleBy((1 / s, 1 / s))

        def fit_to_view():
            self.viewbox.autoRange()

        zoom_in = QAction(
            "Zoom in", parent, triggered=lambda: zoom(1.25)
        )
        zoom_in.setShortcuts([QKeySequence(QKeySequence.ZoomIn),
                              QKeySequence(parent.tr("Ctrl+="))])
        zoom_out = QAction(
            "Zoom out", parent, shortcut=QKeySequence.ZoomOut,
            triggered=lambda: zoom(1 / 1.25)
        )
        zoom_fit = QAction(
            "Fit in view", parent,
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
            triggered=fit_to_view
        )
        parent.addActions([zoom_in, zoom_out, zoom_fit])

class HelpEventDelegate(QObject): #also used by owdistributions
    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate

    def eventFilter(self, obj, event):
        if event.type() == QEvent.GraphicsSceneHelp:
            return self.delegate(event)
        else:
            return False
