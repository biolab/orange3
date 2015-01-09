from xml.sax.saxutils import escape
from math import log10, floor, ceil
import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
import pyqtgraph.graphicsItems.ScatterPlotItem
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
from pyqtgraph.graphicsItems.TextItem import TextItem
from pyqtgraph.Point import Point
from PyQt4.QtCore import Qt, QObject, QEvent, QRectF, QPointF
from PyQt4 import QtCore
from PyQt4.QtGui import QApplication, QColor, QPen, QBrush, QToolTip
from PyQt4.QtGui import QStaticText, QPainterPath, QTransform

from Orange.data import DiscreteVariable, ContinuousVariable

from Orange.widgets import gui
from Orange.widgets.utils.colorpalette import (ColorPaletteGenerator,
                                               ContinuousPaletteGenerator)
from Orange.widgets.utils.plot import \
    OWPalette, OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.scaling import (get_variable_values_sorted,
                                          ScaleScatterPlotData)
from Orange.widgets.settings import Setting, ContextSetting

# TODO Move utility classes to another module, so they can be used elsewhere


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


class PositionedLegendItem(pg.graphicsItems.LegendItem.LegendItem):
    """
    LegendItem that remembers its last position. The position is related to the
    actual widget (it is not retained over sessions). If the widget has multiple
    legends, they can be assigned different appendices to the id.

    The id of the legend is computed from the widget's id and the optional
    additional id.
    """
    positions = {}

    def __init__(self, plot_item, widget, legend_id="", at_bottom=False):
        """
        Construct a legend and insert it into a plot item.

        :param plot_item: PlotItem into which the legend is inserted
        :type: plot_item: PlotItem
        :param widget: the widget with which the legend is associated; used
          only for constructing the id
        :type widget: object
        :param legend_id: appendix used to distinguish between multiple legends
          in the same widget
        :type legend_id: str
        :param at_bottom: if `True` (default is `False`) the default legend
          position is at the bottom
        :type at_bottom: bool
        """
        super().__init__()
        self.id = "{}-{}".format(id(widget), legend_id)
        self.layout.setHorizontalSpacing(15)
        self.layout.setVerticalSpacing(0)
        self.setParentItem(plot_item)
        position = PositionedLegendItem.positions.get(self.id)
        if position:
            self.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=position)
        elif at_bottom:
            self.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -50))
        else:
            self.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

    def setParent(self, parent):
        super().setParent(parent)
        PositionedLegendItem.positions[self.id] = self.pos()


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
        ViewBox.__init__(self, enableMenu=enable_menu)
        self.graph = graph
        self.setMouseMode(self.PanMode)

    def safe_update_scale_box(self, buttonDownPos, currentPos):
        x, y = currentPos
        if buttonDownPos[0] == x:
            x += 1
        if buttonDownPos[1] == y:
            y += 1
        self.updateScaleBox(buttonDownPos, Point(x, y))

    # noinspection PyPep8Naming,PyMethodOverriding
    def mouseDragEvent(self, ev):
        if self.graph.state == SELECT:
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
            super().mouseDragEvent(ev)
        else:
            ev.ignore()

    def mouseClickEvent(self, ev):
        ev.accept()
        self.graph.unselect_all()


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
    alpha_value = Setting(255)
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
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.sizeHint = lambda: QtCore.QSize(500,500)

        self.replot = self.plot_widget.replot
        ScaleScatterPlotData.__init__(self)
        self.scatterplot_item = None

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
        self.scale = None  # DiscretizedScale

        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.update_grid()

        self._tooltip_delegate = HelpEventDelegate(self.help_event)
        self.plot_widget.scene().installEventFilter(self._tooltip_delegate)

    def set_data(self, data, subset_data=None, **args):
        self.plot_widget.clear()
        ScaleScatterPlotData.set_data(self, data, subset_data, **args)

    def update_data(self, attr_x, attr_y):
        self.shown_x = attr_x
        self.shown_y = attr_y

        self.remove_legend()
        if self.scatterplot_item:
            self.plot_widget.removeItem(self.scatterplot_item)
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
            if isinstance(var, DiscreteVariable):
                self.set_labels(axis, get_variable_values_sorted(var))
            else:
                self.set_labels(axis, None)

        color_data, brush_data = self.compute_colors()
        size_data = self.compute_sizes()
        shape_data = self.compute_symbols()
        self.scatterplot_item = ScatterPlotItem(
            x=x_data, y=y_data, data=np.arange(self.n_points),
            symbol=shape_data, size=size_data, pen=color_data, brush=brush_data
        )

        self.plot_widget.addItem(self.scatterplot_item)

        self.scatterplot_item.selected_points = []
        self.scatterplot_item.sigClicked.connect(self.select_by_click)

        self.update_labels()
        self.make_legend()
        self.plot_widget.replot()

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

    update_point_size = update_sizes

    def get_color_index(self):
        color_index = -1
        attr_color = self.attr_color
        if attr_color != "" and attr_color != "(Same color)":
            color_index = self.attribute_name_index[attr_color]
            color_var = self.data_domain[attr_color]
            if isinstance(color_var, DiscreteVariable):
                self.discrete_palette.set_number_of_colors(
                    len(color_var.values))
        return color_index

    def compute_colors(self, keep_colors=False):
        if not keep_colors:
            self.pen_colors = self.brush_colors = None
        color_index = self.get_color_index()

        def make_pen(color, width):
            p = QPen(color, width)
            p.setCosmetic(True)
            return p

        if color_index == -1:
            color = self.plot_widget.palette().color(OWPalette.Data)
            pen = [make_pen(color, 1.5)] * self.n_points
            if self.selection is not None:
                brush = [(QBrush(QColor(128, 128, 128, 255)),
                          QBrush(QColor(128, 128, 128)))[s]
                         for s in self.selection]
            else:
                brush = [QBrush(QColor(128, 128, 128))] * self.n_points
            return pen, brush

        c_data = self.original_data[color_index, self.valid_data]
        if isinstance(self.data_domain[color_index], ContinuousVariable):
            if self.pen_colors is None:
                self.scale = DiscretizedScale(np.min(c_data), np.max(c_data))
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
            if self.selection is not None:
                self.brush_colors[:, 3] = 0
                self.brush_colors[self.selection, 3] = self.alpha_value
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
                colors = palette.getRGB(np.arange(n_colors + 1))
                colors[n_colors] = (128, 128, 128)
                pens = np.array(
                    [make_pen(QColor(*col).darker(self.DarkerValue), 1.5)
                     for col in colors])
                self.pen_colors = pens[c_data]
                self.brush_colors = np.array([
                    [QBrush(QColor(0, 0, 0, 0)),
                     QBrush(QColor(col[0], col[1], col[2], self.alpha_value))]
                    for col in colors])
                self.brush_colors = self.brush_colors[c_data]
            if self.selection is not None:
                brush = np.where(
                    self.selection,
                    self.brush_colors[:, 1], self.brush_colors[:, 0])
            else:
                brush = self.brush_colors[:, 1]
            pen = self.pen_colors
        return pen, brush

    def update_colors(self, keep_colors=False):
        if self.scatterplot_item:
            pen_data, brush_data = self.compute_colors(keep_colors)
            self.scatterplot_item.setPen(pen_data, update=False, mask=None)
            self.scatterplot_item.setBrush(brush_data, mask=None)
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
        self.legend = PositionedLegendItem(self.plot_widget.plotItem, self)

    def remove_legend(self):
        if self.legend:
            self.legend.setParent(None)
            self.legend = None
        if self.color_legend:
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
        if isinstance(color_var, DiscreteVariable):
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
            legend = self.color_legend = PositionedLegendItem(
                self.plot_widget.plotItem,
                self, legend_id="colors", at_bottom=True)
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
                if self.tooltip_shows_all:
                    text += "".join(
                        '   {} = {}\n'.format(attr.name,
                                              self.raw_data[index][attr])
                        for attr in self.data_domain.attributes)
                else:
                    text += '   {} = {}\n   {} = {}\n'.format(
                        self.shown_x, self.raw_data[index][self.shown_x],
                        self.shown_y, self.raw_data[index][self.shown_y])
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
