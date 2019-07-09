from collections import defaultdict

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QPointF, pyqtSignal as Signal
from AnyQt.QtGui import QTransform, QPen, QBrush, QColor, QPainter, QPainterPath
from AnyQt.QtWidgets import \
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, \
    QGraphicsItem, QGraphicsRectItem, QGraphicsItemGroup, QSizePolicy, \
    QGraphicsPathItem, QGraphicsLineItem

from Orange.data import Table, Domain, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.settings import \
    DomainContextHandler, ContextSetting, Setting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.projection.som import SOM


class SomView(QGraphicsView):
    SelectionClear, SelectionAdd, SelectionRemove, SelectionToggle = 1, 2, 4, 8
    SelectionSet = SelectionClear | SelectionAdd
    selection_changed = Signal(set, int)

    def __init__(self, scene, selection_rect):
        super().__init__(scene)
        self.__selectionRect = selection_rect
        self.__button_down_pos = None
        self.size_x = self.size_y = 1

    def set_dimensions(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def _selection_corners(self, event):
        def item_coordinates(x, y):
            return (int(np.clip(np.round(x), 0, self.size_x - 1)),
                    int(np.clip(np.round(y), 0, self.size_y - 1)))

        x0, y0 = self.__button_down_pos.x(), self.__button_down_pos.y()
        pos = self.mapToScene(event.pos())
        x1, y1 = pos.x(), pos.y()
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        outside = x1 < -0.5 or x0 > self.size_x + 0.5 \
            or y1 < -0.5 or y0 > self.size_y + 0.5
        return item_coordinates(x0, y0), item_coordinates(x1, y1), outside

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        self.__button_down_pos = self.mapToScene(event.pos())
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() != Qt.LeftButton:
            return

        (x0, y0), (x1, y1), _ = self._selection_corners(event)
        rect = QRectF(QPointF(x0 - 0.5, y0 - 0.5),
                      QPointF(x1 + 0.5, y1 + 0.5)).normalized()
        self.__selectionRect.setRect(rect)
        self.__selectionRect.show()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        self.__selectionRect.hide()
        (x0, y0), (x1, y1), outside = self._selection_corners(event)
        selection = set() if outside \
            else {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}

        if event.modifiers() & Qt.ControlModifier:
            action = self.SelectionToggle
        elif event.modifiers() & Qt.AltModifier:
            action = self.SelectionRemove
        elif event.modifiers() & Qt.ShiftModifier:
            action = self.SelectionAdd
        else:
            action = self.SelectionClear | self.SelectionAdd

        self.selection_changed.emit(selection, action)
        event.accept()


class PieChart(QGraphicsItem):
    def __init__(self, dist, r, colors):
        super().__init__()
        self.dist = dist
        self.r = r
        self.colors = colors
        self.selected = False

    def set_selected(self, selected):
        self.selected = selected

    def boundingRect(self):
        return QRectF(- self.r, - self.r,
                      2 * self.r, 2 * self.r)

    def paint(self, painter, _option, _index):
        painter.save()
        start_angle = 0
        pen = QPen(QBrush(Qt.black), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawEllipse(self.boundingRect())
        for angle, color in zip(self.dist * 16 * 360, self.colors):
            if angle == 0:
                continue
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawPie(self.boundingRect(), int(start_angle), int(angle))
            start_angle += angle
        painter.restore()


class OWSOM(OWWidget):
    name = "Self-organizing Map"
    description = "Computation of self-organizing map."
    icon = "icons/SOM.svg"
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    settingsHandler = DomainContextHandler()
    manual_dimension = Setting(False)
    size_x = Setting(10)
    size_y = Setting(10)
    shape = Setting(1)
    animate = Setting(True)

    attr_color = ContextSetting(None)
    size_by_instances = Setting(True)
    pie_charts = Setting(False)
    selection = Setting(set(), schema_only=True)

    graph_name = "plot"

    _grid_pen = QPen(QBrush(QColor(224, 224, 224)), 2)
    _grid_pen.setCosmetic(True)

    class Warning(OWWidget.Warning):
        ignoring_disc_variables = Msg("SOM ignores discrete variables.")
        missing_colors = \
            Msg("Some data instances have undefined value of '{}'.")

    class Error(OWWidget.Error):
        empty_data = Msg("Empty dataset")

    def __init__(self):
        super().__init__()
        self.__pending_selection = self.selection

        self.data = self.cont_x = None
        self.valid_indices = None
        self.assignments = None
        self.sizes = None
        self.cells = self.member_data = None
        self.selection = set()

        box = gui.vBox(self.controlArea, "Grid")
        gui.comboBox(
            box, self, "shape", items=("Square grid", "Hexagonal grid"),
            callback=self.on_dimension_change)
        box2 = gui.indentedBox(box, 10)
        gui.checkBox(
            box2, self, "manual_dimension", "Set dimensions manually",
            callback=self.on_manual_dimension_change)
        self.manual_box = box3 = gui.hBox(box2)
        spinargs = dict(
            widget=box3, master=self, minv=5, maxv=100, step=5,
            alignment=Qt.AlignRight, callback=self.on_dimension_change)
        gui.spin(value="size_x", **spinargs)
        gui.widgetLabel(box3, "×")
        gui.spin(value="size_y", **spinargs)
        gui.rubber(box3)

        box = gui.vBox(self.controlArea, "Color")
        gui.comboBox(
            box, self, "attr_color", maximumContentsLength=15,
            callback=self.on_attr_color_change,
            model=DomainModel(placeholder="(Same color)",
                              valid_types=DiscreteVariable))
        gui.checkBox(
            box, self, "pie_charts", label="Show pie charts",
            callback=self.on_pie_chart_change)
        gui.checkBox(
            box, self, "size_by_instances", label="Size by number of instances",
            callback=self.on_attr_size_change)

        gui.rubber(self.controlArea)

        hbox = gui.hBox(self.controlArea, box=True)
        self.restart_button = gui.button(
            hbox, self, "Restart", callback=self.restart_som,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        gui.checkBox(hbox, self, "animate", "Animate")


        self.scene = QGraphicsScene(self)

        selection_rect = self.selection_rect = QGraphicsRectItem()
        pen = QPen(QBrush(Qt.blue), 6)
        pen.setCosmetic(True)
        brush = QBrush(QColor(Qt.blue).lighter(190))
        selection_rect.setPen(pen)
        selection_rect.setBrush(brush)
        selection_rect.setZValue(-100)
        self.scene.addItem(selection_rect)
        selection_rect.hide()

        self.view = SomView(self.scene, selection_rect)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.selection_changed.connect(self.on_selection_change)
        self.mainArea.layout().addWidget(self.view)

        self.elements = None
        self.selection_item = None
        self.grid = None
        self.redraw_grid()

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.Error.clear()
        self.Warning.clear()

        self.data = self.cont_x = None

        if data is not None:
            attrs = data.domain.attributes
            cont_attrs = [var for var in attrs if var.is_continuous]
            if not cont_attrs:
                self.Error.no_numeric_variables()
            else:
                if len(cont_attrs) < len(attrs):
                    self.Warning.ignoring_disc_variables()
                x = Table.from_table(Domain(cont_attrs), data).X
                mask = np.all(np.isfinite(x), axis=1)
                if not np.any(mask):
                    self.Error.no_defined_rows()
                else:
                    self.data = data[mask]
                    self.cont_x = x[mask]
                    self.cont_x -= np.min(self.cont_x, axis=0)[None, :]
                    self.cont_x /= np.sum(self.cont_x, axis=0)[None, :]

        if self.data is not None:
            self.controls.attr_color.model().set_domain(data.domain)
            self.attr_color = data.domain.class_var
            self.openContext(data)
            if self.__pending_selection is not None:
                self.on_selection_change(self.__pending_selection)
                self.__pending_selection = None
        self.recompute_dimensions()
        self.replot()
        self._set_input_summary()

    def _set_input_summary(self):
        if self.data is None:
            self.info.set_input_summary(self.info.NoInput)
            return

        details = f"{len(self.data)} instances"
        ignored = len(self.cont_x) - len(self.data)
        if ignored:
            details += f" {ignored} ignored because of missing values"
        details += f"\n{self.cont_x.shape[1]} numeric variables"
        self.info.set_input_summary(str(len(self.data)), details)

    def clear(self):
        self.data = self.cont_x = None
        self.valid_indices = None
        self.sizes = None
        self.cells = self.member_data = None
        self.assignments = None
        if self.elements is not None:
            self.scene.removeItem(self.elements)
            self.elements = None
        self.clear_selection()
        self.controls.attr_color.model().set_domain(None)
        self.Warning.clear()
        self.Error.clear()

    def clear_selection(self):
        if self.selection_item is not None:
            self.scene.removeItem(self.selection_item)
            self.selection_item = None
        self.selection.clear()

    def recompute_dimensions(self):
        self.manual_box.setEnabled(self.manual_dimension)
        if not self.manual_dimension and self.cont_x is not None:
            self.size_x = self.size_y = \
                max(5, int(np.ceil(np.sqrt(5 * np.sqrt(len(self.cont_x))))))
        else:
            self.size_x = int(5 * np.round(self.size_x / 5))
            self.size_y = int(5 * np.round(self.size_y / 5))

    def on_manual_dimension_change(self):
        self.recompute_dimensions()
        self.redraw_grid()
        self.replot()

    def on_dimension_change(self):
        self.redraw_grid()
        self.replot()

    def on_attr_color_change(self):
        self.controls.pie_charts.setEnabled(self.attr_color is not None)
        self._redraw()

    def on_attr_size_change(self):
        self._redraw()
        self.redraw_selection()

    def on_pie_chart_change(self):
        self._redraw()

    def on_selection_change(self, selection, action=SomView.SelectionSet):
        if action & SomView.SelectionClear:
            self.selection.clear()
        if action & SomView.SelectionAdd:
            self.selection |= selection
        elif action & SomView.SelectionRemove:
            self.selection -= selection
        elif action & SomView.SelectionToggle:
            self.selection ^= selection
        self.redraw_selection()
        self.update_output()

    def redraw_selection(self):
        if self.selection_item:
            self.scene.removeItem(self.selection_item)
        self.selection_item = QGraphicsItemGroup()
        self.selection_item.setZValue(-50)
        pen = QPen(QBrush(Qt.blue), 15)
        pen.setCosmetic(True)
        for x, y in self.selection:
            r = self.sizes[x, y]
            circle = QGraphicsEllipseItem(x - r / 2, y - r / 2, r, r)
            circle.setPen(pen)
            self.selection_item.addToGroup(circle)
        self.scene.addItem(self.selection_item)

    def replot(self):
        self.clear_selection()
        self._set_valid_data()
        self._recompute_som()
        self._redraw()

    def restart_som(self):
        self.clear_selection()
        self._recompute_som()
        self._redraw()

    def _set_valid_data(self):
        if self.attr_color is not None:
            mask = np.isfinite(self.data.get_column_view(self.attr_color)[0])
            self.valid_indices = np.nonzero(mask)[0]
        else:
            self.valid_indices = np.arange(len(self.data))

    def _redraw(self):
        self.Warning.missing_colors.clear()
        if self.elements:
            self.scene.removeItem(self.elements)
        self.view.set_dimensions(self.size_x, self.size_y)

        sizes = self.cells[:, :, 1] - self.cells[:, :, 0]
        sizes = sizes.astype(float)
        if not self.size_by_instances:
            sizes[sizes != 0] = 0.8
        else:
            sizes *= 0.8 / np.max(sizes)
        self.sizes = sizes

        self.elements = QGraphicsItemGroup()
        self.scene.addItem(self.elements)
        if self.attr_color is None:
            self._redraw_same_color()
        else:
            color_column = self.data.get_column_view(self.attr_color)[0]
            colors = [QColor(*color) for color in self.attr_color.colors]
            if self.pie_charts:
                self._redraw_pie_charts(color_column, colors)
            else:
                self._redraw_colored_circles(color_column, colors)
        self._resize()

    def _grid_factors(self):
        if self.shape == 0:
            return 0, 1
        else:
            return 0.5, np.sqrt(3 / 4)

    def _redraw_same_color(self):
        fx, fy = self._grid_factors()
        pen = QPen(QBrush(Qt.black), 4)
        pen.setCosmetic(True)
        brush = QBrush(QColor(192, 192, 192))
        for y in range(self.size_y):
            for x in range(self.size_x - self.shape * (y % 2)):
                r = self.sizes[x, y]
                if not r:
                    continue
                ellipse = QGraphicsEllipseItem()
                ellipse.setRect(x + (y % 2) * fx - r / 2, y * fy - r / 2, r, r)
                ellipse.setPen(pen)
                ellipse.setBrush(brush)
                self.elements.addToGroup(ellipse)

    def _redraw_pie_charts(self, color_column, colors):
        fx, fy = self._grid_factors()
        color_column = color_column.copy()
        color_column[np.isnan(color_column)] = len(colors)
        color_column = color_column.astype(int)
        colors.append(Qt.gray)
        for y in range(self.size_y):
            for x in range(self.size_x - self.shape * (y % 2)):
                r = self.sizes[x, y]
                if not r:
                    continue
                members = self.get_member_indices(x, y)
                color_dist = np.bincount(
                    color_column[members], minlength=len(self.attr_color.values))
                color_dist = color_dist.astype(float) / len(members)
                pie = PieChart(color_dist, r / 2, colors)
                self.elements.addToGroup(pie)
                pie.setPos(x + (y % 2) * fx, y * fy)

    def _redraw_colored_circles(self, color_column, colors):
        fx, fy = self._grid_factors()
        for y in range(self.size_y):
            for x in range(self.size_x - self.shape * (y % 2)):
                r = self.sizes[x, y]
                if not r:
                    continue
                members = self.get_member_indices(x, y)
                color_dist = color_column[members]
                color_dist = color_dist[np.isfinite(color_dist)]
                if len(color_dist) != len(members):
                    self.Warning.missing_colors(self.attr_color.name)
                color_dist = color_dist.astype(int)
                bc = np.bincount(color_dist, minlength=len(self.attr_color.values))
                color = colors[np.argmax(bc)]
                pen = QPen(QBrush(color), 4)
                brush = QBrush(color.lighter(200 - 100 * np.max(bc) / len(members)))
                pen.setCosmetic(True)
                ellipse = QGraphicsEllipseItem()
                ellipse.setRect(x + (y % 2) * fx - r / 2, y * fy - r / 2, r, r)
                ellipse.setPen(pen)
                ellipse.setBrush(brush)
                self.elements.addToGroup(ellipse)
        self._resize()

    def redraw_grid(self):
        if self.grid is not None:
            self.scene.removeItem(self.grid)
        self.grid = QGraphicsItemGroup()
        if self.shape == 0:
            self._draw_square_grid()
        else:
            self._draw_hexagonal_grid()
        self.scene.addItem(self.grid)

    def _draw_square_grid(self):
        for x in range(self.size_x + 1):
            line = QGraphicsLineItem(x - 0.5, -0.5, x - 0.5, self.size_y - 0.5)
            line.setPen(self._grid_pen)
            self.grid.addToGroup(line)
        for y in range(self.size_y + 1):
            line = QGraphicsLineItem(-0.5, y - 0.5, self.size_x - 0.5, y - 0.5)
            line.setPen(self._grid_pen)
            self.grid.addToGroup(line)

    def _draw_hexagonal_grid(self):
        fy = np.sqrt(3) / 2
        for y in range(self.size_y):
            for x in range(self.size_x - y % 2):
                hex = QGraphicsPathItem(_hexagon_path)
                hex.setPen(self._grid_pen)
                self.grid.addToGroup(hex)
                hex.setPos(x + (y % 2) / 2, y * fy)

    def get_member_indices(self, x, y):
        i, j = self.cells[x, y]
        return self.member_data[i:j]

    def _recompute_som(self):
        self.restart_button.setDisabled(True)
        som = SOM(self.size_x, self.size_y, hexagonal=self.shape == 1)
        callback = lambda i: self._animation_step(som) if self.animate else None
        som.fit(self.cont_x, 200, callback=callback)
        if not self.animate:
            self._assign_instances(som)
        self.restart_button.setDisabled(False)

    def _animation_step(self, som):
        from AnyQt.QtWidgets import qApp
        self._assign_instances(som)
        self._redraw()
        qApp.processEvents()

    def _assign_instances(self, som):
        self.assignments = [som.winner(inst) for inst in self.cont_x]
        members = defaultdict(list)
        for i, cell in enumerate(self.assignments):
            members[cell].append(i)
        members.pop(None, None)
        self.cells = np.empty((self.size_x, self.size_y, 2), dtype=int)
        self.member_data = np.empty(len(self.cont_x), dtype=int)
        index = 0
        for x in range(self.size_x):
            for y in range(self.size_y):
                nmembers = len(members[(x, y)])
                self.member_data[index:index + nmembers] = members[(x, y)]
                self.cells[x, y] = [index, index + nmembers]
                index += nmembers

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize()

    def _resize(self):
        vw, vh = self.view.width(), self.view.height()
        scale = min(vw / (self.size_x + 1), vh / (self.size_y + 1))
        self.scene.setSceneRect(-0.5, -0.5, self.size_x - 0.5, self.size_y - 0.5)
        self.view.setTransform(QTransform.fromScale(scale, scale))

    def update_output(self):
        indices = []
        if self.data is not None:
            for (x, y) in self.selection:
                indices.extend(self.get_member_indices(x, y))
        if indices:
            output = self.data[self.valid_indices[indices]]
            self.info.set_output_summary(str(len(indices)))
        else:
            output = None
            self.info.set_output_summary(self.info.NoOutput)
        self.Outputs.data.send(output)


def _draw_hexagon():
    path = QPainterPath()
    s = 0.5 / (np.sqrt(3) / 2)
    path.moveTo(-0.5, -s / 2)
    path.lineTo(-0.5, s / 2)
    path.lineTo(0, s)
    path.lineTo(0.5, s / 2)
    path.lineTo(0.5, -s / 2)
    path.lineTo(0, -s)
    path.lineTo(-0.5, -s / 2)
    return path


_hexagon_path = _draw_hexagon()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSOM).run(Table("heart_disease"))
