from math import log10, floor, ceil
import numpy as np
import pyqtgraph as pg
import pyqtgraph.graphicsItems.ScatterPlotItem
from pyqtgraph.graphicsItems.GraphicsWidgetAnchor import GraphicsWidgetAnchor
from pyqtgraph.graphicsItems.ScatterPlotItem import SpotItem
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt, QRectF, QPointF
from PyQt4.QtGui import (QColor, QTransform,
                         QGraphicsObject, QGraphicsTextItem, QLinearGradient,
                         QPen, QBrush, QGraphicsRectItem, QGraphicsItem)

from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.data.sql.table import SqlTable
from Orange.widgets import gui
from Orange.widgets.utils.colorpalette import (ColorPaletteGenerator,
                                               ContinuousPaletteGenerator)
from Orange.widgets.utils.plot import (OWPalette, OWPlotGUI,
                                       TooltipManager, NOTHING, SELECT, PANNING,
                                       ZOOMING, SELECTION_ADD, SELECTION_REMOVE,
                                       SELECTION_TOGGLE, move_item_xy)
from Orange.widgets.utils.scaling import (get_variable_values_sorted,
                                          ScaleScatterPlotData)
from Orange.widgets.settings import Setting, ContextSetting


class DiscretizedScale:
    def __init__(self, min_v, max_v):
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


class ScatterViewBox(pg.ViewBox):
    def __init__(self, graph):
        pg.ViewBox.__init__(self)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.setMenuEnabled(False)  # should disable the right click menu

    # noinspection PyPep8Naming
    def mouseDragEvent(self, ev):
        if self.graph.state == SELECT:
            ev.accept()
            pos = ev.pos()
            if ev.button() == Qt.LeftButton:
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                if ev.isFinish():
                    self.rbScaleBox.hide()
                    selection_pixel_rect = \
                        QRectF(ev.buttonDownPos(ev.button()), pos)
                    points = self.calculate_points_in_rect(selection_pixel_rect)
                    self.toggle_points_selection(points)
                    self.graph.selection_changed.emit()
                else:
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        elif self.graph.state == ZOOMING or self.graph.state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev)
        else:
            ev.ignore()

    def calculate_points_in_rect(self, pixel_rect):
        # Get the data from the GraphicsItem
        item = self.graph.pgScatterPlotItem
        all_points = item.points()
        value_rect = self.childGroup.mapRectFromParent(pixel_rect)
        points_in_rect = [all_points[i]
                          for i in range(len(all_points))
                          if value_rect.contains(QPointF(all_points[i].pos()))]
        return points_in_rect

    def unselect_all(self):
        item = self.graph.scatterplot_item
        points = item.points()
        for p in points:
            to_unselected_color(p)
        self.graph.selected_points = []

    def toggle_points_selection(self, points):
        if self.graph.selection_behavior == SELECTION_ADD:
            for p in points:
                to_selected_color(p)
                self.graph.selected_points.append(p)
        elif self.graph.selection_behavior == SELECTION_REMOVE:
            for p in points:
                to_unselected_color(p)
                if p in self.graph.selected_points:
                    self.graph.selected_points.remove(p)
        elif self.graph.selection_behavior == SELECTION_TOGGLE:
            for p in points:
                if is_selected(p):
                    to_unselected_color(p)
                    self.graph.selected_points.remove(p)
                else:
                    to_selected_color(p)
                    self.graph.selected_points.append(p)


def _define_symbols():
    symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols
    path = QtGui.QPainterPath()
    path.addEllipse(QtCore.QRectF(-0.25, -0.25, 0.5, 0.5))
    path.moveTo(-0.5, 0.5)
    path.lineTo(0.5, -0.5)
    path.moveTo(-0.5, -0.5)
    path.lineTo(0.5, 0.5)
    symbols["?"] = path

    tr = QtGui.QTransform()
    tr.rotate(180)
    symbols['t'] = tr.map(symbols['t'])

_define_symbols()


class OWScatterPlotGraph(gui.OWComponent, ScaleScatterPlotData):
    selection_changed = QtCore.Signal()

    attr_color = ContextSetting("", ContextSetting.OPTIONAL)
    attr_label = ContextSetting("", ContextSetting.OPTIONAL)
    attr_shape = ContextSetting("", ContextSetting.OPTIONAL)
    attr_size = ContextSetting("", ContextSetting.OPTIONAL)

    point_width = Setting(10)
    alpha_value = Setting(255)
    show_grid = Setting(False)
    show_legend = Setting(True)
    send_selection_on_update = Setting(True)
    tooltip_shows_all = Setting(False)
    square_granularity = Setting(3)
    space_between_cells = Setting(True)

    CurveSymbols = np.array("o x t + d s ?".split())
    MinShapeSize = 6
    LighterValue = 160

    def __init__(self, scatter_widget, parent=None, _="None"):
        gui.OWComponent.__init__(self, scatter_widget)
        svb = ScatterViewBox(self)
        self.plot_widget = pg.PlotWidget(viewBox=svb, parent=parent)
        self.plot_widget.setAntialiasing(True)
        self.replot = self.plot_widget
        ScaleScatterPlotData.__init__(self)
        self.scatterplot_item = None

        self.tooltip_data = []
        self.tooltip = pg.TextItem(border=pg.mkPen(200, 200, 200),
                                   fill=pg.mkBrush(250, 250, 200, 220))
        self.tooltip.hide()

        self.labels = []

        self.master = scatter_widget
        self.inside_colors = None
        self.shown_attribute_indices = []
        self.shown_x = ""
        self.shown_y = ""

        self.valid_data = None  # np.array
        self.n_points = 0

        self.gui = OWPlotGUI(self)
        self.continuous_palette = \
            ContinuousPaletteGenerator(QColor(200, 200, 200),
                                       QColor(0, 0, 0), True)
        self.discrete_palette = ColorPaletteGenerator()

        self.selection_behavior = 0

        self.legend = self.color_legend = None
        self.legend_position = self.color_legend_position = None

        self.tips = TooltipManager(self)
        # self.setMouseTracking(True)
        # self.grabGesture(QPinchGesture)
        # self.grabGesture(QPanGesture)

        self.state = NOTHING
        self._pressed_mouse_button = 0  # Qt.NoButton
        self._pressed_point = None
        self.selection_items = []
        self._current_rs_item = None
        self._current_ps_item = None
        self.polygon_close_treshold = 10
        self.auto_send_selection_callback = None

        self.data_range = {}
        self.map_transform = QTransform()
        self.graph_area = QRectF()
        self.selected_points = []

        self.update_grid()

    def spot_item_clicked(self, plot, points):
        self.scatterplot_item.getViewBox().unselect_all()
        for p in points:
            to_selected_color(p)
            self.selected_points.append(p)
        self.selection_changed.emit()

    # noinspection PyPep8Naming
    def mouseMoved(self, pos):
        act_pos = self.scatterplot_item.mapFromScene(pos)
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
            self.tooltip.setText(text, color=(0, 0, 0))
            self.tooltip.setPos(act_pos)
            self.tooltip.show()
            self.tooltip.setZValue(10)
        else:
            self.tooltip.hide()

    def zoom_button_clicked(self):
        self.scatterplot_item.getViewBox().setMouseMode(
            self.scatterplot_item.getViewBox().RectMode)

    def pan_button_clicked(self):
        self.scatterplot_item.getViewBox().setMouseMode(
            self.scatterplot_item.getViewBox().PanMode)

    def select_button_clicked(self):
        self.scatterplot_item.getViewBox().setMouseMode(
            self.scatterplot_item.getViewBox().RectMode)

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
        self.tooltip_data = []
        self.set_axis_title("bottom", "")
        self.set_axis_title("left", "")

        if self.scaled_data is None or not len(self.scaled_data):
            self.valid_data = None
            self.n_points = 0
            return

        index_x = self.attribute_name_index[attr_x]
        index_y = self.attribute_name_index[attr_y]
        x_data, y_data = self.get_xy_data_positions(attr_x, attr_y)
        self.valid_data = self.get_valid_list([index_x, index_y])
        x_data = x_data[self.valid_data]
        y_data = y_data[self.valid_data]
        self.n_points = len(x_data)

        for axis, name, index in (("bottom", attr_x, index_x),
                                  ("left", attr_y, index_y)):
            self.set_axis_title(axis, name)
            var = self.data_domain[index]
            if isinstance(var, DiscreteVariable):
                self.set_labels(axis, get_variable_values_sorted(var))

        color_data, brush_data = self.compute_colors()
        size_data = self.compute_sizes()
        shape_data = self.compute_symbols()
        self.scatterplot_item = pg.ScatterPlotItem(
            x=x_data, y=y_data,
            symbol=shape_data, size=size_data, pen=color_data, brush=brush_data,
            data=np.arange(self.n_points))
        self.plot_widget.addItem(self.scatterplot_item)
        self.plot_widget.addItem(self.tooltip)
        self.scatterplot_item.selected_points = []
        self.scatterplot_item.sigClicked.connect(self.spot_item_clicked)
        self.scatterplot_item.scene().sigMouseMoved.connect(self.mouseMoved)

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
            if isinstance(self.data_domain[attr_color], DiscreteVariable):
                self.discrete_palette.setNumberOfColors(
                    len(self.data_domain[attr_color].values))
        return color_index

    def compute_colors(self):
                #        if self.have_subset_data:
        #            subset_ids = [example.id for example in self.raw_subset_data]
        #            marked_data = [example.id in subset_ids
        #                           for example in self.raw_data]  # FIX!
        #        else:
        #            marked_data = []
        color_index = self.get_color_index()
        if color_index == -1:
            color = self.color(OWPalette.Data)
            pen = [QPen(QBrush(color), 1.5)] * self.n_points
            brush = [QBrush(QColor(128, 128, 128))] * self.n_points
        else:
            if isinstance(self.data_domain[color_index], ContinuousVariable):
                c_data = self.original_data[color_index, self.valid_data]
                self.scale = DiscretizedScale(np.min(c_data), np.max(c_data))
                c_data -= self.scale.offset
                c_data /= self.scale.width
                c_data = np.floor(c_data) + 0.5
                c_data /= self.scale.bins
                c_data = np.clip(c_data, 0, 1)
                palette = self.continuous_palette
                color = [QColor(*palette.getRGB(i)) for i in c_data]
                pen = np.array([QPen(QBrush(col), 1.5) for col in color])
                for col in color:
                    col.setAlpha(self.alpha_value)
                brush = [QBrush(col.lighter(self.LighterValue))
                         for col in color]
            else:
                palette = self.discrete_palette
                n_colors = palette.numberOfColors
                c_data = self.original_data[color_index, self.valid_data]
                c_data[np.isnan(c_data)] = n_colors
                c_data = c_data.astype(int)
                colors = [palette[i] for i in range(n_colors)] + \
                         [QColor(128, 128, 128)]
                pens = np.array([QPen(QBrush(col), 1.5) for col in colors])
                pen = pens[c_data]
                for color in colors:
                    color.setAlpha(self.alpha_value)
                brushes = np.array(
                    [QBrush(col.lighter(self.LighterValue)) for col in colors])
                brush = brushes[c_data]
        return pen, brush

    def update_colors(self):
        if self.scatterplot_item:
            color_data, brush_data = self.compute_colors()
            self.scatterplot_item.setPen(color_data, update=False, mask=None)
            self.scatterplot_item.setBrush(brush_data, mask=None)
            self.make_legend()

    update_alpha_value = update_colors

    def create_labels(self):
        for x, y in zip(*self.scatterplot_item.getData()):
            ti = pg.TextItem()
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
        legend = self.legend = pg.graphicsItems.LegendItem.LegendItem()
        legend.layout.setHorizontalSpacing(15)
        legend.setParentItem(self.plot_widget.plotItem)
        if self.legend_position:
            legend.anchor(itemPos=(0, 0), parentPos=(0, 0),
                          offset=self.legend_position)
        else:
            legend.anchor(itemPos=(1, 0), parentPos=(1, 0),
                          offset=(-10, 10))

    def remove_legend(self):
        if self.legend:
            self.legend_position = self.legend.pos()
            self.legend.setParent(None)
            self.legend = None
        if self.color_legend:
            self.color_legend_position = self.color_legend.pos()
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
                brush = color.lighter(self.LighterValue)
                self.legend.addItem(
                    pg.ScatterPlotItem(
                        pen=color, brush=brush, size=10,
                        symbol=self.CurveSymbols[i] if use_shape else "o"),
                    value)
        else:
            legend = self.color_legend = pg.graphicsItems.LegendItem.LegendItem()
            legend.layout.setHorizontalSpacing(20)
            legend.layout.setVerticalSpacing(0)
            legend.setParentItem(self.plot_widget.plotItem)
            if self.color_legend_position:
                legend.anchor(itemPos=(0, 0), parentPos=(0, 0),
                              offset=self.color_legend_position)
            else:
                legend.anchor(itemPos=(1, 0), parentPos=(1, 0),
                              offset=(-10, 10))

            scale = self.scale
            labels = ["{0:{1}}".format(scale.offset + i * scale.width,
                                       scale.decimals)
                      for i in range(scale.bins + 1)]
            palette = self.continuous_palette
            symbol = "os"[self.get_shape_index() == -1]
            for i in range(scale.bins):
                color = QColor(*palette.getRGB((i + 0.5) / scale.bins))
                brush = QBrush(color.lighter(self.LighterValue))
                legend.addItem(
                    pg.ScatterPlotItem(pen=color, brush=brush, size=15,
                                       symbol=symbol),
                    "   {} - {}".format(labels[i], labels[i + 1])
                )


    def make_shape_legend(self):
        shape_index = self.get_shape_index()
        if shape_index == -1 or shape_index == self.get_color_index():
            return
        if not self.legend:
            self.create_legend()
        shape_var = self.data_domain[shape_index]
        color = self.color(OWPalette.Data)
        brush = color.lighter(self.LighterValue)
        brush.setAlpha(self.alpha_value)
        for i, value in enumerate(shape_var.values):
            self.legend.addItem(
                pg.ScatterPlotItem(pen=color, brush=brush, size=10,
                                   symbol=self.CurveSymbols[i]), value)

    def get_selections_as_tables(self, attr_list):
        attr_x, attr_y = attr_list
        if not self.have_data:
            return None, None

        sel_indices, unsel_indices = self.get_selections_as_indices(attr_list)

        if type(self.raw_data) is SqlTable:
            selected = [self.raw_data[i]
                        for i, val in enumerate(sel_indices) if val]
            unselected = [self.raw_data[i]
                          for (i, val) in enumerate(unsel_indices) if val]
        else:
            selected = self.raw_data[np.array(sel_indices)]
            unselected = self.raw_data[np.array(unsel_indices)]

        if len(selected) == 0:
            selected = None
        if len(unselected) == 0:
            unselected = None

        return selected, unselected

    def get_selections_as_indices(self, attr_list, valid_data=None):
        attr_x, attr_y = attr_list
        if not self.have_data:
            return [], []
        attr_indices = [self.attribute_name_index[attr] for attr in attr_list]
        if valid_data is None:
            valid_data = self.get_valid_list(attr_indices)
        x_array, y_array = self.get_xy_data_positions(attr_x, attr_y)
        return self.get_selected_points(x_array, y_array, valid_data)

    def get_selected_points(self, xData, yData, validData):
        # hoping that the indices will be in same order as raw_data
        ## TODO check if actually selecting the right points
        selectedIndices = [p.data() for p in self.selected_points]
        selected = [i in selectedIndices for i in range(len(self.raw_data))]
        unselected = [i not in selectedIndices for i in range(len(self.raw_data))]
        return selected, unselected

    def color(self, role, group=None):
        if group:
            return self.plot_widget.palette().color(group, role)
        else:
            return self.plot_widget.palette().color(role)

    def set_palette(self, p):
        self.plot_widget.setPalette(p)

    def send_selection(self):
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()

    def clear_selection(self):
        # called from zoom/select toolbar button 'clear selection'
#        self.scatterplot_item.getViewBox().unselect_all()
        pass

    def update_animations(self, use_animations=None):
        if use_animations is not None:
            self.animate_plot = use_animations
            self.animate_points = use_animations

    def save_to_file(self, size):
        pass
