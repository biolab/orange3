import itertools
from xml.sax.saxutils import escape
from math import log10, floor, ceil
import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
import pyqtgraph.graphicsItems.ScatterPlotItem
from pyqtgraph.graphicsItems.LegendItem import LegendItem, ItemSample
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
from pyqtgraph.graphicsItems.TextItem import TextItem
from pyqtgraph.Point import Point
from PyQt4.QtCore import Qt, QObject, QEvent, QRectF, QPointF
from PyQt4 import QtCore
from PyQt4.QtGui import QApplication, QColor, QPen, QBrush, QToolTip
from PyQt4.QtGui import QStaticText, QPainterPath, QTransform, QPinchGesture, QPainter

from Orange.widgets import gui
from Orange.widgets.utils.colorpalette import (ColorPaletteGenerator,
                                               ContinuousPaletteGenerator)
from Orange.widgets.utils.plot import \
    OWPalette, OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.scaling import (get_variable_values_sorted,
                                          ScaleScatterPlotData)
from Orange.widgets.settings import Setting, ContextSetting


# TODO Move utility classes to another module, so they can be used elsewhere

SELECTION_WIDTH = 4

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
        cuts = ["{0:{1}}".format(scale.offset + i * scale.width, scale.decimals)
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
        self.layout.setVerticalSpacing(0)
        self.layout.setHorizontalSpacing(15)
        self.layout.setColumnAlignment(1, Qt.AlignLeft | Qt.AlignVCenter)

        if pen is None:
            pen = QPen(QColor(196, 197, 193, 200), 1)
            pen.setCosmetic(True)
        self.__pen = pen

        if brush is None:
            brush = QBrush(QColor(232, 232, 232, 100))
        self.__brush = brush

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
    c1, c2, *parentPos = rect_anchor_pos(rect, parent_rect)
    return corner_anchor(c1), parentPos


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
        dif = max_v - min_v
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
                if ev.isFinish():
                    self.rbScaleBox.hide()
                    pixel_rect = QRectF(ev.buttonDownPos(ev.button()), pos)
                    value_rect = self.childGroup.mapRectFromParent(pixel_rect)
                    self.graph.select_by_rectangle(value_rect)
                else:
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
        self.axHistory = self.axHistory[:self.axHistoryPointer] + [ self.viewRect() ]

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
        if ev.button() ==  QtCore.Qt.RightButton: # undo zoom
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
    path.addEllipse(QRectF(-0.25, -0.25, 0.5, 0.5))
    path.moveTo(-0.5, 0.5)
    path.lineTo(0.5, -0.5)
    path.moveTo(-0.5, -0.5)
    path.lineTo(0.5, 0.5)
    symbols["?"] = path

    tr = QTransform()
    tr.rotate(180)
    symbols['t'] = tr.map(symbols['t'])

_define_symbols()


class OWScatterPlotGraph(gui.OWComponent, ScaleScatterPlotData):
    attr_color = ContextSetting("", ContextSetting.OPTIONAL)
    attr_label = ContextSetting("", ContextSetting.OPTIONAL)
    attr_shape = ContextSetting("", ContextSetting.OPTIONAL)
    attr_size = ContextSetting("", ContextSetting.OPTIONAL)

    point_width = Setting(10)
    alpha_value = Setting(128)
    show_grid = Setting(False)
    show_legend = Setting(True)
    tooltip_shows_all = Setting(False)
    square_granularity = Setting(3)
    space_between_cells = Setting(True)

    CurveSymbols = np.array("o x t + d s ?".split())
    MinShapeSize = 6
    DarkerValue = 120
    UnknownColor = (168, 50, 168)

    def __init__(self, scatter_widget, parent=None, _="None"):
        gui.OWComponent.__init__(self, scatter_widget)
        self.view_box = InteractiveViewBox(self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, parent=parent,
                                         background="w")
        self.plot_widget.getPlotItem().buttonsHidden = True
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.sizeHint = lambda: QtCore.QSize(500,500)

        self.replot = self.plot_widget.replot
        ScaleScatterPlotData.__init__(self)
        self.scatterplot_item = None
        self.scatterplot_item_sel = None

        self.labels = []

        self.master = scatter_widget
        self.shown_attribute_indices = []
        self.shown_x = ""
        self.shown_y = ""
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

    def new_data(self, data, subset_data=None, **args):
        self.plot_widget.clear()
        self.subset_indices = set(e.id for e in subset_data) if subset_data else None
        self.selection = None
        self.set_data(data, **args)

    def update_data(self, attr_x, attr_y):
        self.shown_x = attr_x
        self.shown_y = attr_y

        self.remove_legend()
        if self.scatterplot_item:
            self.plot_widget.removeItem(self.scatterplot_item)
        if self.scatterplot_item_sel:
            self.plot_widget.removeItem(self.scatterplot_item_sel)
        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []
        self.set_axis_title("bottom", "")
        self.set_axis_title("left", "")

        if self.scaled_data is None or not len(self.scaled_data):
            self.valid_data = None
            self.n_points = 0
            return

        index_x = self.attribute_name_index[attr_x]
        index_y = self.attribute_name_index[attr_y]
        self.valid_data = self.get_valid_list([index_x, index_y])
        x_data, y_data = self.get_xy_data_positions(
            attr_x, attr_y, self.valid_data)
        x_data = x_data[self.valid_data]
        y_data = y_data[self.valid_data]
        self.n_points = len(x_data)

        for axis, name, index in (("bottom", attr_x, index_x),
                                  ("left", attr_y, index_y)):
            self.set_axis_title(axis, name)
            var = self.data_domain[index]
            if var.is_discrete:
                self.set_labels(axis, get_variable_values_sorted(var))
            else:
                self.set_labels(axis, None)

        color_data, brush_data = self.compute_colors()
        color_data_sel, brush_data_sel = self.compute_colors_sel()
        size_data = self.compute_sizes()
        shape_data = self.compute_symbols()
        self.scatterplot_item = ScatterPlotItem(
            x=x_data, y=y_data, data=np.arange(self.n_points),
            symbol=shape_data, size=size_data, pen=color_data, brush=brush_data
        )
        self.scatterplot_item_sel = ScatterPlotItem(
            x=x_data, y=y_data, data=np.arange(self.n_points),
            symbol=shape_data, size=size_data + SELECTION_WIDTH,
            pen=color_data_sel, brush=brush_data_sel
        )
        self.plot_widget.addItem(self.scatterplot_item)
        self.plot_widget.addItem(self.scatterplot_item_sel)

        self.scatterplot_item.selected_points = []
        self.scatterplot_item.sigClicked.connect(self.select_by_click)

        self.update_labels()
        self.make_legend()
        self.view_box.init_history()
        self.plot_widget.replot()

        min_x, max_x = np.nanmin(x_data), np.nanmax(x_data)
        min_y, max_y = np.nanmin(y_data), np.nanmax(y_data)
        self.view_box.setRange(
            QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
            padding=0.025)
        self.view_box.tag_history()

    def set_labels(self, axis, labels):
        axis = self.plot_widget.getAxis(axis)
        if labels:
            ticks = [[(i, labels[i]) for i in range(len(labels))]]
            axis.setTicks(ticks)
        else:
            axis.setTicks(None)

    def set_axis_title(self, axis, title):
        self.plot_widget.setLabel(axis=axis, text=title)

    def get_size_index(self):
        size_index = -1
        attr_size = self.attr_size
        if attr_size != "" and attr_size != "(Same size)":
            size_index = self.attribute_name_index[attr_size]
        return size_index

    def compute_sizes(self):
        size_index = self.get_size_index()
        if size_index == -1:
            size_data = np.full((self.n_points,), self.point_width)
        else:
            size_data = \
                self.MinShapeSize + \
                self.no_jittering_scaled_data[size_index] * self.point_width
        size_data[np.isnan(size_data)] = self.MinShapeSize - 2
        return size_data

    def update_sizes(self):
        if self.scatterplot_item:
            size_data = self.compute_sizes()
            self.scatterplot_item.setSize(size_data)
            self.scatterplot_item_sel.setSize(size_data + SELECTION_WIDTH)

    update_point_size = update_sizes

    def get_color_index(self):
        color_index = -1
        attr_color = self.attr_color
        if attr_color != "" and attr_color != "(Same color)":
            color_index = self.attribute_name_index[attr_color]
            color_var = self.data_domain[attr_color]
            if color_var.is_discrete:
                self.discrete_palette.set_number_of_colors(
                    len(color_var.values))
        return color_index

    def compute_colors_sel(self, keep_colors=False):
        if not keep_colors:
            self.pen_colors_sel = self.brush_colors_sel = None

        def make_pen(color, width):
            p = QPen(color, width)
            p.setCosmetic(True)
            return p

        pens = [ QPen(Qt.NoPen),
                 make_pen(QColor(255, 190, 0, 255), SELECTION_WIDTH + 1.) ]
        if self.selection is not None:
            pen = [ pens[a] for a in self.selection ]
        else:
            pen = [pens[0]] * self.n_points
        brush = [QBrush(QColor(255, 255, 255, 0))] * self.n_points
        return pen, brush

    def compute_colors(self, keep_colors=False):
        if not keep_colors:
            self.pen_colors = self.brush_colors = None
        color_index = self.get_color_index()

        def make_pen(color, width):
            p = QPen(color, width)
            p.setCosmetic(True)
            return p

        subset = None
        if self.subset_indices:
            subset = np.array([ ex.id in self.subset_indices
                for ex in self.raw_data[self.valid_data] ])

        if color_index == -1: #color = "Same color"
            color = self.plot_widget.palette().color(OWPalette.Data)
            pen = [make_pen(color, 1.5)] * self.n_points
            if subset is not None:
                brush = [(QBrush(QColor(128, 128, 128, 0)),
                          QBrush(QColor(128, 128, 128, self.alpha_value)))[s]
                         for s in subset]
            else:
                brush = [QBrush(QColor(128, 128, 128))] * self.n_points
            return pen, brush

        c_data = self.original_data[color_index, self.valid_data]
        if self.data_domain[color_index].is_continuous:
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
                     np.full((self.n_points, 1), self.alpha_value)])
                self.pen_colors *= 100 / self.DarkerValue
                self.pen_colors = [make_pen(QColor(*col), 1.5)
                                   for col in self.pen_colors.tolist()]
            if subset is not None:
                self.brush_colors[:, 3] = 0
                self.brush_colors[subset, 3] = self.alpha_value
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
                    [make_pen(QColor(*col).darker(self.DarkerValue), 1.5)
                     for col in colors])
                self.pen_colors = pens[c_data]
                self.brush_colors = np.array([
                    [QBrush(QColor(0, 0, 0, 0)),
                     QBrush(QColor(col[0], col[1], col[2], self.alpha_value))]
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
        if self.scatterplot_item:
            pen_data, brush_data = self.compute_colors(keep_colors)
            pen_data_sel, brush_data_sel = self.compute_colors_sel(keep_colors)
            self.scatterplot_item.setPen(pen_data, update=False, mask=None)
            self.scatterplot_item.setBrush(brush_data, mask=None)
            self.scatterplot_item_sel.setPen(pen_data_sel, update=False, mask=None)
            self.scatterplot_item_sel.setBrush(brush_data_sel, mask=None)
            if not keep_colors:
                self.make_legend()

    update_alpha_value = update_colors

    def create_labels(self):
        for x, y in zip(*self.scatterplot_item.getData()):
            ti = TextItem()
            self.plot_widget.addItem(ti)
            ti.setPos(x, y)
            self.labels.append(ti)

    def update_labels(self):
        if not self.attr_label:
            for label in self.labels:
                label.setText("")
            return
        if not self.labels:
            self.create_labels()
        label_column = self.raw_data.get_column_view(self.attr_label)[0]
        formatter = self.raw_data.domain[self.attr_label].str_val
        label_data = map(formatter, label_column)
        black = pg.mkColor(0, 0, 0)
        for label, text in zip(self.labels, label_data):
            label.setText(text, black)

    def get_shape_index(self):
        shape_index = -1
        attr_shape = self.attr_shape
        if attr_shape and attr_shape != "(Same shape)" and \
                len(self.data_domain[attr_shape].values) <= \
                len(self.CurveSymbols):
            shape_index = self.attribute_name_index[attr_shape]
        return shape_index

    def compute_symbols(self):
        shape_index = self.get_shape_index()
        if shape_index == -1:
            shape_data = self.CurveSymbols[np.zeros(self.n_points, dtype=int)]
        else:
            shape_data = self.original_data[shape_index]
            shape_data[np.isnan(shape_data)] = len(self.CurveSymbols) - 1
            shape_data = self.CurveSymbols[shape_data.astype(int)]
        return shape_data

    def update_shapes(self):
        if self.scatterplot_item:
            shape_data = self.compute_symbols()
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
        self.legend.anchor(*self.__legend_anchor)

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
        color_index = self.get_color_index()
        if color_index == -1:
            return
        color_var = self.data_domain[color_index]
        use_shape = self.get_shape_index() == color_index
        if color_var.is_discrete:
            if not self.legend:
                self.create_legend()
            palette = self.discrete_palette
            for i, value in enumerate(color_var.values):
                color = QColor(*palette.getRGB(i))
                brush = color.lighter(self.DarkerValue)
                self.legend.addItem(
                    ScatterPlotItem(
                        pen=color, brush=brush, size=10,
                        symbol=self.CurveSymbols[i] if use_shape else "o"),
                    value)
        else:
            legend = self.color_legend = LegendItem()
            legend.setParentItem(self.plot_widget.getViewBox())
            legend.anchor(*self.__color_legend_anchor)

            label = PaletteItemSample(self.continuous_palette, self.scale)
            legend.addItem(label, "")
            legend.setGeometry(label.boundingRect())

    def make_shape_legend(self):
        shape_index = self.get_shape_index()
        if shape_index == -1 or shape_index == self.get_color_index():
            return
        if not self.legend:
            self.create_legend()
        shape_var = self.data_domain[shape_index]
        color = self.plot_widget.palette().color(OWPalette.Data)
        pen = QPen(color.darker(self.DarkerValue))
        color.setAlpha(self.alpha_value)
        for i, value in enumerate(shape_var.values):
            self.legend.addItem(
                ScatterPlotItem(pen=pen, brush=color, size=10,
                                symbol=self.CurveSymbols[i]), value)

    def zoom_button_clicked(self):
        self.scatterplot_item.getViewBox().setMouseMode(
            self.scatterplot_item.getViewBox().RectMode)

    def pan_button_clicked(self):
        self.scatterplot_item.getViewBox().setMouseMode(
            self.scatterplot_item.getViewBox().PanMode)

    def select_button_clicked(self):
        self.scatterplot_item.getViewBox().setMouseMode(
            self.scatterplot_item.getViewBox().RectMode)

    def reset_button_clicked(self):
        self.view_box.autoRange()

    def select_by_click(self, _, points):
        self.select(points)

    def select_by_rectangle(self, value_rect):
        points = [point
                  for point in self.scatterplot_item.points()
                  if value_rect.contains(QPointF(point.pos()))]
        self.select(points)

    def unselect_all(self):
        self.selection = None
        self.update_colors(keep_colors=True)

    def select(self, points):
        # noinspection PyArgumentList
        keys = QApplication.keyboardModifiers()
        if self.selection is None or not keys & (
                        Qt.ShiftModifier + Qt.ControlModifier + Qt.AltModifier):
            self.selection = np.full(self.n_points, False, dtype=np.bool)
        indices = [p.data() for p in points]
        if keys & Qt.ControlModifier:
            self.selection[indices] = False
        elif keys & Qt.AltModifier:
            self.selection[indices] = 1 - self.selection[indices]
        else:  # Handle shift and no modifiers
            self.selection[indices] = True
        self.update_colors(keep_colors=True)
        self.master.selection_changed()

    def get_selection(self):
        if self.selection is None:
            return np.array([], dtype=int)
        else:
            return np.arange(len(self.raw_data)
                )[self.valid_data][self.selection]

    def set_palette(self, p):
        self.plot_widget.setPalette(p)

    def save_to_file(self, size):
        pass

    def help_event(self, event):
        if self.scatterplot_item is None:
            return False

        act_pos = self.scatterplot_item.mapFromScene(event.scenePos())
        points = self.scatterplot_item.pointsAt(act_pos)
        text = ""
        if len(points):
            for i, p in enumerate(points):
                index = p.data()
                text += "Attributes:\n"
                if self.tooltip_shows_all and \
                        len(self.data_domain.attributes) < 30:
                    text += "".join(
                        '   {} = {}\n'.format(attr.name,
                                              self.raw_data[index][attr])
                        for attr in self.data_domain.attributes)
                else:
                    text += '   {} = {}\n   {} = {}\n'.format(
                        self.shown_x, self.raw_data[index][self.shown_x],
                        self.shown_y, self.raw_data[index][self.shown_y])
                    if self.tooltip_shows_all:
                        text += "   ... and {} others\n\n".format(
                            len(self.data_domain.attributes) - 2)
                if self.data_domain.class_var:
                    text += 'Class:\n   {} = {}\n'.format(
                        self.data_domain.class_var.name,
                        self.raw_data[index][self.raw_data.domain.class_var])
                if i < len(points) - 1:
                    text += '------------------\n'

            text = ('<span style="white-space:pre">{}</span>'
                    .format(escape(text)))

            QToolTip.showText(event.screenPos(), text, widget=self.plot_widget)
            return True
        else:
            return False


class HelpEventDelegate(QObject):
    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate

    def eventFilter(self, obj, event):
        if event.type() == QEvent.GraphicsSceneHelp:
            return self.delegate(event)
        else:
            return False
