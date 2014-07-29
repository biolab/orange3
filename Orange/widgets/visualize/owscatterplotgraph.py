import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.GraphicsWidgetAnchor import GraphicsWidgetAnchor
from pyqtgraph.graphicsItems.ScatterPlotItem import SpotItem
from PyQt4 import QtCore
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


def is_selected(point):
    if not isinstance(point, SpotItem):
        raise TypeError('Expected SpotItem instead of %s' % point.__class__)
    return point.pen().color() == point.brush().color()


def to_selected_color(point):
    if not isinstance(point, SpotItem):
        raise TypeError('Expected SpotItem instead of %s' % point.__class__)
    point.setBrush(point.pen().color())


def to_unselected_color(point):
    if not isinstance(point, SpotItem):
        raise TypeError('Expected SpotItem instead of %s' % point.__class__)
    color = point.pen().color()
    lighter_color = color.lighter(LIGHTER_VALUE)
    lighter_color.setAlpha(self.alpha_value)
    point.setBrush(lighter_color)


class GradientLegendItem(QGraphicsObject, GraphicsWidgetAnchor):
    gradient_width = 20

    def __init__(self, title, palette, values, parent):
        QGraphicsObject.__init__(self, parent)
        GraphicsWidgetAnchor.__init__(self)
        self.parent = parent
        self.legend = parent

        self.palette = palette
        self.values = values

        self.title = QGraphicsTextItem('%s:' % title, self)
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        self.title_item = QGraphicsRectItem(self.title.boundingRect(), self)
        self.title_item.setPen(QPen(Qt.NoPen))
        self.title_item.stackBefore(self.title)

        self.label_items = [QGraphicsTextItem(text, self) for text in values]
        for i in self.label_items:
            i.setTextWidth(50)

        self.rect = QRectF()

        self.gradient_item = QGraphicsRectItem(self)
        self.gradient = QLinearGradient()
        self.gradient.setStops([(v * 0.1, self.palette[v * 0.1])
                                for v in range(11)])
        self.orientation = Qt.Horizontal
        self.set_orientation(Qt.Vertical)

        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def set_orientation(self, orientation):
        if self.orientation == orientation:
            return

        self.orientation = orientation

        if self.orientation == Qt.Vertical:
            height = max(item.boundingRect().height()
                         for item in self.label_items)
            total_height = height * max(5, len(self.label_items))
            interval = (total_height -
                        self.label_items[-1].boundingRect().height()
                        ) / (len(self.label_items) - 1)
            self.gradient_item.setRect(10, 0, self.gradient_width, total_height)
            self.gradient.setStart(10, 0)
            self.gradient.setFinalStop(10, total_height)
            self.gradient_item.setBrush(QBrush(self.gradient))
            self.gradient_item.setPen(QPen(Qt.NoPen))
            y = -20   # hja, no; dela --> pri boundingRect() zato pristejem +20
            x = 0
            move_item_xy(self.title, x, y, False)
            y = 10
            x = 30
            for item in self.label_items:
                move_item_xy(item, x, y, False)
                                       # self.parent.graph.animate_plot)
                y += interval
            self.rect = QRectF(10, 0,
                               self.gradient_width +
                               max(item.boundingRect().width()
                                   for item in self.label_items),
                               self.label_items[0].boundingRect().height() *
                               max(5, len(self.label_items)))
        else:
            # za horizontalno orientacijo nisem dodajal title-a
            width = 50
            height = max(item.boundingRect().height()
                         for item in self.label_items)
            total_width = width * max(5, len(self.label_items))
            interval = (total_width -
                        self.label_items[-1].boundingRect().width()
                        ) / (len(self.label_items) - 1)

            self.gradient_item.setRect(0, 0, total_width, self.gradient_width)
            self.gradient.setStart(0, 0)
            self.gradient.setFinalStop(total_width, 0)
            self.gradient_item.setBrush(QBrush(self.gradient))
            self.gradient_item.setPen(QPen(Qt.NoPen))
            x = 0
            y = 30
            for item in self.label_items:
                move_item_xy(item, x, y, False)
                                       # self.parent.graph.animate_plot)
                x += interval
            self.rect = QRectF(0, 0, total_width, self.gradient_width + height)

    # noinspection PyPep8Naming
    def boundingRect(self):
        width = max(self.rect.width(), self.title_item.boundingRect().width())
        height = self.rect.height() + self.title_item.boundingRect().height()
        return QRectF(self.rect.left(), self.rect.top(), width, height)

    def paint(self, painter, option, widget):
        pass


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

    def shuffle_points(self):
        pass


class OWScatterPlotGraph(gui.OWComponent, ScaleScatterPlotData):
    selection_changed = QtCore.Signal()

    attr_color = ContextSetting("", ContextSetting.OPTIONAL)
    attr_label = ContextSetting("", ContextSetting.OPTIONAL)
    attr_shape = ContextSetting("", ContextSetting.OPTIONAL)
    attr_size = ContextSetting("", ContextSetting.OPTIONAL)

    point_width = Setting(10)
    alpha_value = Setting(255)
    animate_plot = Setting(True)
    animate_points = Setting(True)
    show_grid = Setting(True)
    show_axes_titles = Setting(True)
    show_legend = Setting(True)
    show_filled_symbols = Setting(True)
    show_probabilities = Setting(False)
    show_distributions = Setting(False)
    send_selection_on_update = Setting(True)
    tooltip_shows_all = Setting(False)
    square_granularity = Setting(3)
    space_between_cells = Setting(True)

    CurveSymbols = "oxt+ds"
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

        self.potentials_classifier = None
        self.potentials_image = None
        self.potentials_curve = None

        self.valid_data = None  # np.array

        self.gui = OWPlotGUI(self)
        self.continuous_palette = \
            ContinuousPaletteGenerator(QColor(200, 200, 200),
                                       QColor(0, 0, 0), True)
        self.discrete_palette = ColorPaletteGenerator()

        self.selection_behavior = 0

        self._legend = None
        self._gradient_legend = None
        self._legend_position = None
        self._gradient_legend_position = None

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

    def update_data(self, attr_x, attr_y, **args):
        # self.legend().clear()
        self.tooltip_data = []
        self.potentials_classifier = None
        self.potentials_image = None
        self.shown_x = attr_x
        self.shown_y = attr_y

        self.remove_legend()
        self.remove_gradient_legend()
        if self.scatterplot_item:
            self.plot_widget.removeItem(self.scatterplot_item)

        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []

        if self.scaled_data is None or not len(self.scaled_data):
            self.set_axis_title("bottom", "")
            self.set_axis_title("left", "")
            return

        self.__dict__.update(args)      # set value from args dictionary

        size_index = self.get_size_index()
        shape_index = self.get_shape_index()
        color_index = self.get_color_index()
        show_continuous_legend = \
            color_index != -1 and \
            isinstance(self.data_domain[color_index], ContinuousVariable)

        index_x = self.attribute_name_index[attr_x]
        index_y = self.attribute_name_index[attr_y]

        attr_indices = self.shown_attribute_indices = [
            x for x in (index_x, index_y, color_index, shape_index, size_index)
            if x != -1]

        self.set_axis_title("bottom", attr_x)
        if isinstance(self.data_domain[index_x], DiscreteVariable):
            labels = get_variable_values_sorted(self.data_domain[index_x])
            self.set_labels("bottom", labels)
        self.set_axis_title("left", attr_y)
        if isinstance(self.data_domain[index_y], DiscreteVariable):
            labels = get_variable_values_sorted(self.data_domain[index_y])
            self.set_labels("left", labels)

        x_data, y_data = self.get_xy_data_positions(attr_x, attr_y)
        self.valid_data = self.get_valid_list(attr_indices)
        self.n_points = len(x_data)

        # if self.potentials_curve:
        #     self.potentials_curve.detach()
        #     self.potentials_curve = None
        # if self.show_probabilities and color_index >= 0:
        #     if isinstance(self.data_domain[color_index], DiscreteVariable):
        #         color_var = DiscreteVariable(
        #             self.attribute_names[color_index],
        #             values=get_variable_values_sorted(
        #                 self.data_domain[color_index]))])
        #     else:
        #         color_var = ContinuousVariable(
        #             self.attributeNames[color_index])
        #     domain = Orange.data.Domain(
        #         [self.data_domain[index_x], self.data_domain[index_y],
        #          color_var])
        #     x_diff = xmax - xmin
        #     y_diff = ymax - ymin
        #     scX = x_data / x_diff
        #     scY = y_data / y_diff
        #     classData = self.original_data[color_index]
        #
        #     probData = numpy.transpose(numpy.array([scX, scY, classData]))
        #     probData= numpy.compress(valid_data, probData, axis = 0)
        #
        #     sys.stderr.flush()
        #     self.xmin = xmin
        #     self.xmax = xmax
        #     self.ymin = ymin
        #     self.ymax = ymax
        #
        #     if probData.any():
        #         self.potentials_classifier = Orange.P2NN(
        #             domain, probData, None, None, None, None)
        #         self.potentials_curve =
        #             ProbabilitiesItem(self.potentials_classifier,
        #                               self.squareGranularity, 1.,
        #                               self.spaceBetweenCells)
        #         self.potentials_curve.attach(self)
        #     else:
        #         self.potentials_classifier = None

        color_data, brush_data = self.compute_colors()
        size_data = self.compute_sizes()
        shape_data = self.compute_symbols()
        self.scatterplot_item = pg.ScatterPlotItem(
            x=x_data, y=y_data, symbol=shape_data, size=size_data,
            pen=color_data, brush=brush_data, data=np.arange(len(x_data)))

        self.scatterplot_item.sigClicked.connect(self.spot_item_clicked)
        self.scatterplot_item.selected_points = []
        self.plot_widget.addItem(self.scatterplot_item)
        self.plot_widget.addItem(self.tooltip)
        self.scatterplot_item.scene().sigMouseMoved.connect(self.mouseMoved)

        self.update_labels()
        self.plot_widget.replot()

        # Here, create (up to) three separate legends

        # def get_discrete_index(ind):
        #     if isinstance(self.data_domain[color_index], DiscreteVariable):
        #         return ind
        #     else:
        #         return -1
        #
        # disc_color_index = get_discrete_index(color_index)
        # disc_shape_index = get_discrete_index(shape_index)
        # disc_size_index = get_discrete_index(size_index)
        #
        # max_index = max(disc_color_index, disc_shape_index, disc_size_index)
        # if max_index != -1:
        #     self.create_legend()
        #
        #
        #     num = len(self.data_domain[max_index].values)
        #     varValues = get_variable_values_sorted(
        #         self.data_domain[max_index])
        #     for ind in range(num):
        #         # construct an item to pass to ItemSample
        #         if disc_color_index != -1:
        #             p = QColor(*palette.getRGB(ind))
        #             b = p.lighter(LighterValue)
        #         else:
        #             p = def_color
        #             b = p.lighter(LighterValue)
        #         if discSizeIndex != -1:
        #             sz = MinShapeSize +
        #                 round(ind*self.point_width/len(varValues))
        #         else:
        #             sz = self.point_width
        #         if disc_shape_index != -1:
        #             sym = self.CurveSymbols[ind]
        #         else:
        #             sym = self.scatterplot_item.opts['symbol']
        #         pg.ItemSampl
        #         sample = lambda: None
        #         sample.opts = {'pen': p,
        #                 'brush': b,
        #                 'size': sz,
        #                 'symbol': sym
        #         }
        #         self.legend().addItem(item=sample, name=varValues[ind])

        # if color_index != -1 and show_continuous_legend:
        #     values = [("%%.%df"
        #         % self.data_domain[attr_color].number_of_decimals % v)
        #         for v in self.attr_values[attr_color]]
        #     self.create_gradient_legend(
        #         attr_color, values=values,
        #         parentSize=self.plot_widget.size())

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
                self.disc_palette.setNumberOfColors(
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
            color_data = self.color(OWPalette.Data)
            brush_data = color_data.lighter(self.LighterValue)
            brush_data.setAlpha(self.alpha_value)
            color_data = [color_data] * self.n_points
            brush_data = [brush_data] * self.n_points
        else:
            if isinstance(self.data_domain[color_index], ContinuousVariable):
                c_data = self.no_jittering_scaled_data[color_index]
                palette = self.continuous_palette
            else:
                c_data = self.original_data[color_index]
                palette = self.discrete_palette
            valid_color_data = c_data * self.valid_data
            color_data = [QColor(*palette.getRGB(i)) for i in valid_color_data]
            brush_data = [color.lighter(self.LighterValue)
                          for color in color_data]
            color_data = [QPen(QBrush(col), 1.5) for col in color_data]
            for i in range(len(brush_data)):
                brush_data[i].setAlpha(self.alpha_value)
        return color_data, brush_data

    def update_colors(self):
        if self.scatterplot_item:
            color_data, brush_data = self.compute_colors()
            self.scatterplot_item.setPen(color_data, update=False, mask=None)
            self.scatterplot_item.setBrush(brush_data, mask=None)

        # self.plot_widget.plotItem.setAlpha(self.alpha_value)  # TODO: FIX

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
            shape_data = [self.CurveSymbols[0]] * self.n_points
        else:
            shape_data = [self.CurveSymbols[i]
                          for i in self.original_data[shape_index].astype(int)]
        return shape_data

    def update_shapes(self):
        if self.scatterplot_item:
            shape_data = self.compute_symbols()
            self.scatterplot_item.setSymbol(shape_data)

    def update_grid(self):
        self.plot_widget.showGrid(x=self.show_grid, y=self.show_grid)

    def add_tip(self, x, y, attr_indices=None, data_index=None, text=None):
        if text is None:
            if self.tooltip_shows_all:
                text = self.get_tooltip_text(
                    self.raw_data[data_index], range(len(self.attributeNames)))
            else:
                text = self.get_tooltip_text(
                    self.raw_data[data_index], attr_indices)
        self.tips.addToolTip(x, y, text)

    # called from OWPlot
    # noinspection PyPep8Naming
    def buildTooltip(self, exampleIndex):
        if exampleIndex < 0:
            example = self.rawSubsetData[-exampleIndex - 1]
        else:
            example = self.raw_data[exampleIndex]

        if self.tooltip_show_all:
            text = self.getExampleTooltipText(example)
        else:
            text = self.getExampleTooltipText(
                example, self.shown_attribute_indices)
        return text

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

    # def computePotentials(self):
    #     # import orangeom
    #     s = self.graph_area.toRect().size()
    #     if not s.isValid():
    #         self.potentials_image = QImage()
    #         return
    #     rx = s.width()
    #     ry = s.height()
    #     rx -= rx % self.squareGranularity
    #     ry -= ry % self.squareGranularity
    #
    #     ox = int(self.transform(xBottom, 0) - self.transform(xBottom, self.xmin))
    #     oy = int(self.transform(yLeft, self.ymin) - self.transform(yLeft, 0))
    #
    #     if not getattr(self, "potentials_image", None) or getattr(self, "potentialContext", None) != (rx, ry, self.shown_x, self.shown_y, self.squareGranularity, self.jitter_size, self.jitter_continuous, self.spaceBetweenCells):
    #         self.potentialContext = (rx, ry, self.shown_x, self.shown_y, self.squareGranularity, self.jitter_size, self.jitter_continuous, self.spaceBetweenCells)
    #         self.potentialsImageFromClassifier = self.potentials_classifier

    def set_labels(self, axis, labels):
        axis = self.plot_widget.getAxis(axis)
        if labels:
            ticks = [[(i, labels[i]) for i in range(len(labels))]]
            axis.setTicks(ticks)
        else:
            axis.setTicks(None)

    def set_axis_title(self, axis, title):
        self.plot_widget.setLabel(axis=axis, text=title)

    def color(self, role, group = None):
        if group:
            return self.plot_widget.palette().color(group, role)
        else:
            return self.plot_widget.palette().color(role)

    def set_palette(self, p):
        self.plot_widget.setPalette(p)

    def legend(self):
        if hasattr(self, '_legend'):
            return self._legend
        else:
            return None

    def gradient_legend(self):
        if hasattr(self, '_gradient_legend'):
            return self._gradient_legend
        else:
            return None

    def create_legend(self, offset):
        self._legend = pg.graphicsItems.LegendItem.LegendItem(offset=offset)
        self._legend.setParentItem(self.plot_widget.plotItem)
        if self._legend_position:
            # if legend was moved from default position, show on the same position as before
            item_pos = (0, 0)
            parent_pos = (0, 0)
            offset = self._legend_position
        else:
            item_pos = (1, 0)   # 0 - left and top; 1 - right and bottom
            parent_pos = (1, 0) # 0 - left and top; 1 - right and bottom
            offset = (-10, 10)
        self._legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)
        self.update_legend()

    def remove_legend(self):
        if self._legend:
            self._legend_position = self._legend.pos() # restore position next time it is shown
            self._legend.items = []
            while self._legend.layout.count() > 0:
                self._legend.removeAt(0)
            if self._legend.scene and self._legend.scene():
                self._legend.scene().removeItem(self._legend)
            self._legend = None

    def create_gradient_legend(self, title, values, parentSize):
        ## TODO: legendi se prekrijeta, ce se malo potrudis
        self._gradient_legend = GradientLegendItem(title, self.cont_palette, [str(v) for v in values], self.plot_widget.plotItem)
        if self._gradient_legend_position:
            # if legend was moved from default position, show on the same position as before
            item_pos = (0, 0)
            parent_pos = (0, 0)
            offset = self._gradient_legend_position
        else:
            y = 20 if not self._legend else self._legend.boundingRect().y() + self._legend.boundingRect().height() + 25 # shown beneath _legend
            item_pos = (1, 0)   # 0 - left and top; 1 - right and bottom
            parent_pos = (1, 0) # 0 - left and top; 1 - right and bottom
            offset = (-10, y)
        self._gradient_legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)
        self.update_legend()

    def remove_gradient_legend(self):
        if self._gradient_legend:
            self._gradient_legend_position = self._gradient_legend.pos() # restore position next time it is shown
            parent = self._gradient_legend.parentItem()
            parent.removeItem(self._gradient_legend)   # tole nic ne naredi
            self._gradient_legend.hide()               # tale skrije, rajsi bi vidu ce izbrise
            self._gradient_legend = None

    def update_legend(self):
        if self._legend:
            if (self._legend.isVisible() == self.show_legend):
                return
            self._legend.setVisible(self.show_legend)

        if self._gradient_legend:
            # if (self._gradient_legend.isVisible() == self.show_legend):
            #     return
            self._gradient_legend.setVisible(self.show_legend)

    def send_selection(self):
        if self.auto_send_selection_callback:
            self.auto_send_selection_callback()

    def clear_selection(self):
        # called from zoom/select toolbar button 'clear selection'
#        self.scatterplot_item.getViewBox().unselect_all()
        pass

    def shuffle_points(self):
        pass
        # if self.main_curve:
        #     self.main_curve.shuffle_points()

    def update_animations(self, use_animations=None):
        if use_animations is not None:
            self.animate_plot = use_animations
            self.animate_points = use_animations

    def setCanvasBackground(self, color):
        # called when closing set colors dialog (ColorPalleteDlg)
        print('setCanvasBackground - color=%s' % color)

    def setGridColor(self, color):
        # called when closing set colors dialog (ColorPalleteDlg)
        print('setGridColor - color=%s' % color)

    def save_to_file(self, size):
        pass
