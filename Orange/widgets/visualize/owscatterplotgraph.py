from collections import Counter, defaultdict
import sys
import itertools
from xml.sax.saxutils import escape
from math import log2, log10, floor, ceil

import numpy as np
from scipy import sparse as sp

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
from pyqtgraph.Point import Point

from Orange.statistics.util import bincount
from Orange.widgets import gui
from Orange.widgets.utils import classdensity
from Orange.widgets.utils.colorpalette import (ColorPaletteGenerator,
                                               ContinuousPaletteGenerator, DefaultRGBColors)
from Orange.widgets.utils.plot import \
    OWPalette, OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.settings import Setting, ContextSetting


# TODO Move utility classes to another module, so they can be used elsewhere
from Orange.widgets.widget import OWWidget, Msg

SELECTION_WIDTH = 5
MAX = 11  # maximum number of colors or shapes (including Other)
MAX_POINTS_IN_TOOLTIP = 5


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
        font = p.font()
        font.setPixelSize(11)
        p.setFont(font)
        for i, label in enumerate(self.labels):
            color = QColor(*self.palette.getRGB((i + 0.5) / scale.bins))
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

# TODO: Add this into pull request:
# This widget handled sparse data in an interesting manner: it stored the
# original data into an attribute `_data`. Then upon each change of color,
# size, shape or label, it called its own self.setData with densified
# data that contained only the needed attributes. I think that widget should
# not call its own signal handler, in particular not to transform its data


# For each property (size, color, shape, label) there is a method that returns
# a column (from a table or elsewhere) called get_size_data, get_color_data...
# Another method calls the above method and returns actual sizes, colors or
# labels; these are called get_sizes, get_colors, get_shapes, get_labels.
# Override one or another, whichever is more convenient.
# The former methods are called only from the latter. The latter are called
# from two places: when the graph is first set up with the new data, and when
# it is updated (in which case only one method is called, e.g. as a result
# of the user changing the color attribute).

# update_density called update_graph, which update the entire graph ... and,
# oh, yes, reset the zoom stack for no reason

# Basically, there were a zillion update_this, reset_this, update_that routines
# that reset this and that and called each other, passing arguments like
# "new=True", "keep_foo=False" to prevent resetting some parts of the graph.

# When unifying all projections, the common scatterpotgraph called a method
# update_regression_line (or sth like that) ... which then had to be implemented
# in all derived widgets, including, say, MDS. Same for "update_colors".

# MDS tooltip shows mds-x, mds-y in tooltips, because all data had to be sent
# as attributes...

class OWScatterPlotBase(gui.OWComponent):
    label_only_selected = Setting(False)

    point_width = Setting(10)
    alpha_value = Setting(128)
    show_grid = Setting(False)
    show_legend = Setting(True)
    class_density = Setting(False)
    jitter_size = Setting(10)

    resolution = 256

    CurveSymbols = np.array("o x t + d s t2 t3 p h star ?".split())
    MinShapeSize = 6
    DarkerValue = 120
    UnknownColor = (168, 50, 168)

    def __init__(self, scatter_widget, parent=None, _="None", view_box=InteractiveViewBox):
        super().__init__(scatter_widget)

        self.subset_is_shown = False

        self.view_box = view_box(self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, parent=parent,
                                         background="w")
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.getPlotItem().buttonsHidden = True
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.sizeHint = lambda: QSize(500, 500)
        scene = self.plot_widget.scene()
        self._create_drag_tooltip(scene)

        self.replot = self.plot_widget.replot
        self.density_img = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None

        self.labels = []

        self.master = scatter_widget

        self.selection = None  # np.ndarray
        self.n_points = 0

        self.gui = OWPlotGUI(self)
        self.palette = None

        self.legend = self.color_legend = None
        self.__legend_anchor = (1, 0), (1, 0)
        self.__color_legend_anchor = (1, 1), (1, 1)

        self.scale = None  # DiscretizedScale

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

    def clear(self):
        self.remove_legend()
        self.plot_widget.clear()

        self.density_img = None
        self.scatterplot_item = None
        self.scatterplot_item_sel = None
        self.labels = []
        self.selection = None

    def reset_graph(self):
        self.clear()
        self.update_coordinates()
        self.update_point_props()

    def update_point_props(self):
        self.update_sizes()
        self.update_colors()
        self.update_selection_colors()
        self.update_shapes()
        self.update_labels()

    # Coordinates
    def _reset_view(self, x_data, y_data):
        min_x, max_x = np.nanmin(x_data), np.nanmax(x_data)
        min_y, max_y = np.nanmin(y_data), np.nanmax(y_data)
        self.view_box.setRange(
            QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
            padding=0.025)
        self.view_box.init_history()
        self.view_box.tag_history()

    def get_coordinates(self):
        x, y = self.master.get_coordinates_data()
        if x is None:
            self.n_points = 0
            return None, None
        x, y = self.jitter_coordinates(x, y)
        self.n_points = len(x) if x is not None else 0
        return x, y

    def jitter_coordinates(self, x, y):
        if self.jitter_size == 0:
            return x, y
        return self._jitter_data(x, 0), self._jitter_data(y, 1)

    @classmethod
    def _update_plot_coordinates(cls, plot, x, y):
        data = dict(x=x, y=y)
        for prop in ('pen', 'brush', 'size', 'symbol', 'data'):
            data[prop] = plot.data[prop]
        plot.setData(**data)

    def update_coordinates(self):
        x, y = self.get_coordinates()
        if x is None:
            return
        if self.scatterplot_item is None:
            data_indices = np.arange(self.n_points)
            self.scatterplot_item = ScatterPlotItem(x=x, y=y, data=data_indices)
            self.plot_widget.addItem(self.scatterplot_item)
            self.scatterplot_item.sigClicked.connect(self.select_by_click)

            self.scatterplot_item_sel = ScatterPlotItem(
                x=x, y=y, data=data_indices)
            self.plot_widget.addItem(self.scatterplot_item_sel)

            self.update_point_props()
        else:
            self._update_plot_coordinates(self.scatterplot_item, x, y)
            self._update_plot_coordinates(self.scatterplot_item_sel, x, y)

        self.update_label_coords(x, y)
        self.update_density()  # Todo: doesn't work: try MDS with density on
        self._reset_view(x, y)

    def _jitter_data(self, x, seed=0):
        random = np.random.RandomState(seed=seed)
        off = self.jitter_size * (np.max(x) - np.min(x)) / 25
        return x + random.uniform(-off, off, len(x))

    # Sizes
    def get_sizes(self):
        size_column = self.master.get_size_data()
        if size_column is None or self.attr_size == OWPlotGUI.SizeByOverlap:
            return self.point_width
        size_column = size_column.copy()
        size_column -= np.min(size_column)
        mx = np.max(size_column)
        if mx > 0:
            size_column /= mx
        return self.MinShapeSize + self.point_width * size_column

    def update_sizes(self):
        if self.scatterplot_item:
            size_data = self.get_sizes()
            size_data = self.master.impute_sizes(size_data)
            self.scatterplot_item.setSize(size_data)
            self.scatterplot_item_sel.setSize(size_data + SELECTION_WIDTH)

    update_point_size = update_sizes  # backward compatibility (needed?!)
    update_size = update_sizes

    # Colors
    def get_colors(self):
        self.master.set_palette()
        c_data = self.master.get_color_data()
        subset = self.master.get_subset_mask()
        self.subset_is_shown = subset is not None
        if c_data is None:  # same color
            return self._get_same_colors(subset)
        elif self.master.is_continuous_color():
            return self._get_continuous_colors(c_data, subset)
        else:
            return self._get_discrete_colors(c_data, subset)

    def _get_same_colors(self, subset):
        color = self.plot_widget.palette().color(OWPalette.Data)
        pen = _make_pen(color, 1.5)
        if subset is not None:
            brush = [(QBrush(QColor(128, 128, 128, 0)),
                      QBrush(QColor(128, 128, 128, 255)))[s]
                     for s in subset]
        else:
            brush = QBrush(QColor(128, 128, 128, self.alpha_value))
        return pen, brush

    def _get_continuous_colors(self, c_data, subset):
        self.scale = DiscretizedScale(np.nanmin(c_data), np.nanmax(c_data))
        c_data -= self.scale.offset
        c_data /= self.scale.width
        c_data = np.floor(c_data) + 0.5
        c_data /= self.scale.bins
        c_data = np.clip(c_data, 0, 1)
        pen = self.palette.getRGB(c_data)
        brush = np.hstack(
            [pen, np.full((len(pen), 1), self.alpha_value, dtype=int)])
        pen *= 100
        pen //= self.DarkerValue
        pen = [_make_pen(QColor(*col), 1.5) for col in pen.tolist()]

        if subset is not None:
            brush[:, 3] = 0
            brush[subset, 3] = 255
        else:
            brush[:, 3] = self.alpha_value
        brush = np.array([QBrush(QColor(*col)) for col in brush.tolist()])
        return pen, brush

    def _get_discrete_colors(self, c_data, subset):
        n_colors = self.palette.number_of_colors
        c_data = c_data.copy()
        c_data[np.isnan(c_data)] = n_colors
        c_data = c_data.astype(int)
        colors = np.r_[self.palette.getRGB(np.arange(n_colors)),
                       [[128, 128, 128]]]
        pens = np.array(
            [_make_pen(QColor(*col).darker(self.DarkerValue), 1.5)
             for col in colors])
        pen = pens[c_data]
        alpha = self.alpha_value if subset is None else 255
        brushes = np.array([
            [QBrush(QColor(0, 0, 0, 0)),
             QBrush(QColor(col[0], col[1], col[2], alpha))]
            for col in colors])
        brush = brushes[c_data]

        if subset is not None:
            brush = np.where(subset, brush[:, 1], brush[:, 0])
        else:
            brush = self.brush[:, 1]
        return pen, brush

    def update_colors(self):
        if self.scatterplot_item is None:
            return
        pen_data, brush_data = self.get_colors()
        self.scatterplot_item.setPen(pen_data, update=False, mask=None)
        self.scatterplot_item.setBrush(brush_data, mask=None)
        self.make_legend()
        self.update_density()

    update_alpha_value = update_colors

    def update_selection_colors(self):
        if self.scatterplot_item_sel is None:
            return
        pen, brush = self.get_colors_sel()
        self.scatterplot_item_sel.setPen(pen, update=False, mask=None)
        self.scatterplot_item_sel.setBrush(brush, mask=None)

    def update_density(self):
        if self.density_img:
            self.plot_widget.removeItem(self.density_img)
        if self.scatterplot_item is not None \
                and self.master.can_draw_density() and self.class_density:
            [min_x, max_x], [min_y, max_y] = self.view_box.viewRange()
            x_data, y_data = self.scatterplot_item.getData()
            rgb_data = [pen.color().getRgb()[:3]
                        for pen in self.scatterplot_item.data['pen']]
            self.density_img = classdensity.class_density_image(
                min_x, max_x, min_y, max_y, self.resolution,
                x_data, y_data, rgb_data)
            self.plot_widget.addItem(self.density_img)
        else:
            self.density_img = None

    def get_colors_sel(self):
        nopen = QPen(Qt.NoPen)
        if self.selection is None:
            pen = nopen
        else:
            sels = np.max(self.selection)
            if sels == 1:
                pen = np.where(
                    self.selection,
                    _make_pen(QColor(255, 190, 0, 255), SELECTION_WIDTH + 1),
                    nopen)
            else:
                palette = ColorPaletteGenerator(number_of_colors=sels + 1)
                pen = np.choose(
                    self.selection,
                    [nopen] + [_make_pen(palette[i], SELECTION_WIDTH + 1)
                               for i in range(sels)])
        brush = QBrush(QColor(255, 255, 255, 0))
        return pen, brush

    # Labels
    def get_labels(self):
        return self.master.get_label_data()

    def update_labels(self):
        if self.label_only_selected and self.selection is None:
            label_data = None
        else:
            label_data = self.get_labels()
        if label_data is None:
            for label in self.labels:
                label.setText("")
            return
        if not self.labels:
            self._create_labels()
        black = pg.mkColor(0, 0, 0)
        if self.label_only_selected:
            for label, text, selected \
                    in zip(self.labels, label_data, self.selection):
                label.setText(text if selected else "", black)
        else:
            for label, text in zip(self.labels, label_data):
                label.setText(text, black)

    def _create_labels(self):
        for x, y in zip(*self.scatterplot_item.getData()):
            ti = TextItem()
            self.plot_widget.addItem(ti)
            ti.setPos(x, y)
            self.labels.append(ti)

    def update_label_coords(self, x, y):
        if self.label_only_selected:
            if self.selection:
                for label, selected, xp, yp in zip(
                        self.labels, self.selection, x, y):
                    if selected:
                        label.setPos(xp, yp)
        else:
            for label, xp, yp in zip(self.labels, x, y):
                label.setPos(xp, yp)

    # Shapes
    def get_shapes(self):
        shape_data = self.master.get_shape_data()
        shape_data = self.master.impute_shapes(shape_data, len(self.CurveSymbols) - 1)
        if isinstance(shape_data, np.ndarray):
            shape_data = shape_data.astype(int)
        return self.CurveSymbols[shape_data]

    def update_shapes(self):
        if self.scatterplot_item:
            shape_data = self.get_shapes()
            self.scatterplot_item.setSymbol(shape_data)
        self.make_legend()

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
        if not self.legend:
            self.create_legend()
        self._make_color_legend()
        self._make_shape_legend()
        self.update_legend()

    def _make_color_legend(self):
        if self.master.is_continuous_color():
            if not self.scale:
                return
            legend = self.color_legend = LegendItem()
            legend.setParentItem(self.plot_widget.getViewBox())
            legend.restoreAnchor(self.__color_legend_anchor)

            label = PaletteItemSample(self.palette, self.scale)
            legend.addItem(label, "")
            legend.setGeometry(label.boundingRect())
        else:
            labels = self.master.get_color_labels()
            if not labels or not self.palette:
                return
            use_shape = self.master.combined_legend()
            for i, value in enumerate(labels):
                color = QColor(*self.palette.getRGB(i))
                pen = _make_pen(color.darker(self.DarkerValue), 1.5)
                color.setAlpha(255 if self.subset_is_shown else self.alpha_value)
                brush = QBrush(color)
                self.legend.addItem(
                    ScatterPlotItem(
                        pen=pen, brush=brush, size=10,
                        symbol=self.CurveSymbols[i] if use_shape else "o"),
                    escape(value))

    def _make_shape_legend(self):
        if self.master.combined_legend():
            return
        labels = self.master.get_shape_labels()
        if labels is None:
            return
        color = QColor(0, 0, 0)
        color.setAlpha(self.alpha_value)
        for i, value in enumerate(labels):
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
        self.reset_graph()

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
        self.update_selection_colors()
        if self.label_only_selected:
            self.update_labels()
        self.master.selection_changed()

    def select(self, points):
        # noinspection PyArgumentList
        if self.n_points == 0:
            return
        if self.selection is None:
            self.selection = np.zeros(self.n_points, dtype=np.uint8)
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
            self.selection = np.zeros(self.n_points, dtype=np.uint8)
            self.selection[indices] = 1
        self.update_selection_colors()
        if self.label_only_selected:
            self.update_labels()
        self.master.selection_changed()

    def get_selection(self):
        if self.selection is None:
            return np.array([], dtype=np.uint8)
        else:
            return np.flatnonzero(self.selection)

    def save_to_file(self, size):
        pass

    def help_event(self, event):
        if self.scatterplot_item is None:
            return False
        act_pos = self.scatterplot_item.mapFromScene(event.scenePos())
        point_data = [p.data() for p in self.scatterplot_item.pointsAt(act_pos)]
        text = self.master.get_tooltip(point_data)
        if text:
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


class OWProjectionWidget(OWWidget):
    attr_color = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_label = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_shape = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_size = ContextSetting(None, required=ContextSetting.OPTIONAL)

    class Information(OWWidget.Information):
        missing_size = Msg(
            "Points with undefined '{}' are shown in smaller size")
        missing_shape = Msg(
            "Points with undefined '{}' are shown as crossed circles")

    def __init__(self):
        super().__init__()
        self.data = None
        self.valid_data = None

        self.set_palette()

    def init_attr_values(self):
        data = self.data
        domain = data.domain if data and len(data) else None
        for attr in ("attr_color", "attr_shape", "attr_size", "attr_label"):
            getattr(self.controls, attr).model().set_domain(domain)
            setattr(self, attr, None)
        if domain is not None:
            self.attr_color = domain.class_var

    def get_xy_data(self):
        return None, None

    def get_column(self, attr, filter_valid=True,
                   merge_infrequent=False, return_labels=False):
        if attr is None:
            return None
        all_data = self.data.get_column_view(attr)[0]
        if sp.issparse(all_data):
            all_data = all_data.toDense()  # TODO -- just guessing; fix this!
        if filter_valid and self.valid_data is not None:
            all_data = all_data[self.valid_data]
        if not merge_infrequent or attr.is_continuous \
                or len(attr.values) <= MAX:
            return attr.values if return_labels else all_data
        dist = bincount(all_data, max_val=len(attr.values) - 1)
        infrequent = np.zeros(len(attr.values), dtype=bool)
        infrequent[np.argsort(dist)[:-MAX]] = True
        if return_labels:
            return [value for value, infreq in zip(attr.values, infrequent)
                    if not infreq] + ["Other"]
        else:
            return np.choose(infrequent, (all_data, MAX))

    # Sizes
    def get_size_data(self):
        return self.get_column(self.attr_size)

    def impute_sizes(self, size_data):
        nans = np.isnan(size_data)
        if np.any(nans):
            size_data[nans] = self.graph.MinShapeSize - 2
            self.Information.missing_size(self.attr_size)
        else:
            self.Information.missing_size.clear()

        # scale sizes because of overlaps
        if self.attr_size == OWPlotGUI.SizeByOverlap:
            size_data = np.multiply(size_data, self.overlap_factor)
        return size_data

    # Colors
    def get_color_data(self):
        return self.get_column(self.attr_color, merge_infrequent=True)

    def get_color_labels(self):
        return self.get_column(self.attr_color, merge_infrequent=True,
                               return_labels=True)

    def is_continuous_color(self):
        return self.attr_color is not None and self.attr_color.is_continuous

    def set_palette(self):
        if self.attr_color is None:
            self.graph.palette = None
            return
        colors = self.attr_color.colors
        if self.attr_color.is_discrete:
            self.graph.palette = ColorPaletteGenerator(
                number_of_colors=min(len(colors), MAX),
                rgb_colors=colors if len(colors) <= MAX
                else DefaultRGBColors)
        else:
            self.graph.palette = ContinuousPaletteGenerator(*colors)

    def can_draw_density(self):
        return self.data is not None and \
               self.data.domain is not None and \
               len(self.data) > 1 and \
               self.attr_color is not None

    # Labels
    def get_label_data(self, formatter=None):
        if self.attr_label:
            label_data = self.get_column(self.attr_label)
            return map(formatter or self.attr_label.str_val, label_data)

    # Shapes
    def get_shape_data(self):
        return self.get_column(self.attr_shape, merge_infrequent=True)

    def get_shape_labels(self):
        return self.get_column(self.attr_shape, merge_infrequent=True,
                               return_labels=True)

    def impute_shapes(self, shape_data, default_symbol):
        if shape_data is None:
            return 0
        nans = np.isnan(shape_data)
        if np.any(nans):
            shape_data[nans] = default_symbol
            self.Information.missing_shape(self.attr_shape)
        else:
            self.Information.missing_shape.clear()
        return shape_data

    # Todo: Bug: color, shape and size do not update when changed from something
    # to 'Same'

    # Tooltip
    def _point_tooltip(self, point_id, skip_attrs=()):
        def show_part(point_data, singular, plural, max_shown, vars):
            cols = [escape('{} = {}'.format(var.name, point_data[var]))
                    for var in vars[:max_shown + 2]
                    if vars == domain.class_vars
                    or var not in skip_attrs][:max_shown]
            if not cols:
                return ""
            n_vars = len(vars)
            if n_vars > max_shown:
                cols[-1] = "... and {} others".format(n_vars - max_shown + 1)
            return \
                "<b>{}</b>:<br/>".format(singular if n_vars < 2 else plural) \
                + "<br/>".join(cols)

        domain = self.data.domain
        parts = (("Class", "Classes", 4, domain.class_vars),
                 ("Meta", "Metas", 4, domain.metas),
                 ("Feature", "Features", 10, domain.attributes))

        point_data = self.data[point_id]
        return "<br/>".join(show_part(point_data, *columns)
                            for columns in parts)

    def get_tooltip(self, point_ids):
        text = "<hr/>".join(self._point_tooltip(point_id)
                            for point_id in point_ids[:MAX_POINTS_IN_TOOLTIP])
        if len(point_ids) > MAX_POINTS_IN_TOOLTIP:
            text = "{} instances<hr/>{}<hr/>...".format(len(point_ids), text)
        return text

    # Legend
    def combined_legend(self):
        return self.attr_shape == self.attr_color
