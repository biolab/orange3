from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QRectF, QPointF, pyqtSignal as Signal, QObject, \
    QThread
from AnyQt.QtGui import QTransform, QPen, QBrush, QColor, QPainter, QPainterPath
from AnyQt.QtWidgets import \
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, \
    QGraphicsItem, QGraphicsRectItem, QGraphicsItemGroup, QSizePolicy, \
    QGraphicsPathItem
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME, \
    create_annotated_table

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
        self.hexagonal = True

    def set_dimensions(self, size_x, size_y, hexagonal):
        self.size_x = size_x
        self.size_y = size_y
        self.hexagonal = hexagonal

    def _selection_corners(self, event):
        def item_coordinates(x, y):
            if self.hexagonal:
                return x, y
            else:
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
        d = 0 if self.hexagonal else 0.5
        rect = QRectF(QPointF(x0 - d, y0 - d),
                      QPointF(x1 + d, y1 + d)).normalized()
        self.__selectionRect.setRect(rect)
        self.__selectionRect.show()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        self.__selectionRect.hide()
        (x0, y0), (x1, y1), outside = self._selection_corners(event)
        if outside:
            selection = set()
        elif self.hexagonal:
            fy = np.sqrt(3) / 2
            y0 = max(0, int(y0 / fy + 0.5))
            y1 = min(self.size_y, int(np.ceil(y1 / fy + 0.5)))
            selection = set()
            for y in range(y0, y1):
                x0_ = max(0, int(x0 + 0.5 - (y % 2) / 2))
                x1_ = min(self.size_x - y % 2,
                          int(np.ceil(x1 + 0.5 - (y % 2) / 2)))
                selection |= {(x, y) for x in range(x0_, x1_)}
        else:
            selection = {(x, y) for x in range(x0, x1 + 1)
                         for y in range(y0, y1 + 1)}

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


N_ITERATIONS = 200


class OWSOM(OWWidget):
    name = "Self-organizing Map"
    description = "Computation of self-organizing map."
    icon = "icons/SOM.svg"
    keywords = ["SOM"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settingsHandler = DomainContextHandler()
    manual_dimension = Setting(False)
    size_x = Setting(10)
    size_y = Setting(10)
    hexagonal = Setting(1)
    initialization = Setting(0)

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
        self._optimizer = None
        self._optimizer_thread = None
        self._stop_optimization = False

        self.data = self.cont_x = None
        self.assignments = None
        self.sizes = None
        self.cells = self.member_data = None
        self.selection = set()

        box = gui.vBox(self.controlArea, box=True)
        hbox = gui.hBox(box)
        self.restart_button = gui.button(
            hbox, self, "Restart", callback=self.restart_som_pressed,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        gui.radioButtons(
            box, self, "initialization",
            ("Initialize with PCA", "Random initialization",
             "Replicable random"))

        self.grid_box = box = gui.vBox(self.controlArea, "Geometry")
        gui.comboBox(
            box, self, "hexagonal", items=("Square grid", "Hexagonal grid"),
            callback=self.on_geometry_change)
        box2 = gui.indentedBox(box, 10)
        gui.checkBox(
            box2, self, "manual_dimension", "Set dimensions manually",
            callback=self.on_manual_dimension_change)
        self.manual_box = box3 = gui.hBox(box2)
        spinargs = dict(
            widget=box3, master=self, minv=5, maxv=100, step=5,
            alignment=Qt.AlignRight, callback=self.on_geometry_change)
        gui.spin(value="size_x", **spinargs)
        gui.widgetLabel(box3, "×")
        gui.spin(value="size_y", **spinargs)
        self.manual_box.setDisabled(not self.manual_dimension)
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
        self.view.setMinimumWidth(400)
        self.view.setMinimumHeight(400)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.selection_changed.connect(self.on_selection_change)
        self.mainArea.layout().addWidget(self.view)

        self.elements = None
        self.grid = None
        self.grid_cells = None
        self.redraw_grid()

    @Inputs.data
    def set_data(self, data):
        self.stop_optimization_and_wait()

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
                if sp.issparse(x):
                    self.data = data
                    self.cont_x = x.tocsr()
                else:
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
            class_var = data.domain.class_var
            if class_var is not None and class_var.is_discrete:
                self.attr_color = class_var
            else:
                self.attr_color = None
            self.openContext(data)
        self.recompute_dimensions()
        self.replot()
        self._set_input_summary()

    def _set_input_summary(self):
        if self.data is None:
            self.info.set_input_summary(self.info.NoInput)
            return

        details = f"{len(self.data)} instances"
        ignored = self.cont_x.shape[0] - len(self.data)
        if ignored:
            details += f" {ignored} ignored because of missing values"
        details += f"\n{self.cont_x.shape[1]} numeric variables"
        self.info.set_input_summary(str(len(self.data)), details)

    def clear(self):
        self.data = self.cont_x = None
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
        self.selection.clear()
        self.redraw_selection()

    def recompute_dimensions(self):
        self.manual_box.setEnabled(self.manual_dimension)
        if not self.manual_dimension and self.cont_x is not None:
            self.size_x = self.size_y = \
                max(5, int(np.ceil(np.sqrt(5 * np.sqrt(self.cont_x.shape[0])))))
        else:
            self.size_x = int(5 * np.round(self.size_x / 5))
            self.size_y = int(5 * np.round(self.size_y / 5))
        self._resize()
        self.redraw_grid()

    def on_manual_dimension_change(self):
        self.recompute_dimensions()
        self.replot()

    def on_geometry_change(self):
        self._resize()
        self.redraw_grid()
        self.replot()

    def on_attr_color_change(self):
        self.controls.pie_charts.setEnabled(self.attr_color is not None)
        self._redraw()

    def on_attr_size_change(self):
        self._redraw()

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
        brushes = [QBrush(Qt.NoBrush), QBrush(QColor(240, 240, 255))]
        sel_pen = QPen(QBrush(QColor(128, 128, 128)), 2)
        sel_pen.setCosmetic(True)
        pens = [self._grid_pen, sel_pen]
        for y in range(self.size_y):
            for x in range(self.size_x - (y % 2) * self.hexagonal):
                cell = self.grid_cells[y, x]
                selected = (x, y) in self.selection
                cell.setBrush(brushes[selected])
                cell.setPen(pens[selected])
                cell.setZValue(selected)

    def replot(self):
        self.clear_selection()
        self._recompute_som()

    def restart_som_pressed(self):
        if self._optimizer_thread is not None:
            self._stop_optimization = True
        else:
            self.clear_selection()
            self._recompute_som()

    def _redraw(self):
        self.Warning.missing_colors.clear()
        if self.elements:
            self.scene.removeItem(self.elements)
        self.view.set_dimensions(self.size_x, self.size_y, self.hexagonal)

        if self.cells is None:
            return
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
            color_column = \
                self.data.get_column_view(self.attr_color)[0].astype(float)
            colors = [QColor(*color) for color in self.attr_color.colors]
            if self.pie_charts:
                self._redraw_pie_charts(color_column, colors)
            else:
                self._redraw_colored_circles(color_column, colors)

    def _grid_factors(self):
        return (0.5, np.sqrt(3 / 4)) if self.hexagonal else (0, 1)

    def _redraw_same_color(self):
        fx, fy = self._grid_factors()
        pen = QPen(QBrush(Qt.black), 4)
        pen.setCosmetic(True)
        brush = QBrush(QColor(192, 192, 192))
        for y in range(self.size_y):
            for x in range(self.size_x - self.hexagonal * (y % 2)):
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
            for x in range(self.size_x - self.hexagonal * (y % 2)):
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
            for x in range(self.size_x - self.hexagonal * (y % 2)):
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

    def redraw_grid(self):
        fy = np.sqrt(3) / 2
        if self.grid is not None:
            self.scene.removeItem(self.grid)
        self.grid = QGraphicsItemGroup()
        self.grid_cells = np.full((self.size_y, self.size_x), None)
        for y in range(self.size_y):
            for x in range(self.size_x - (y % 2) * self.hexagonal):
                if self.hexagonal:
                    cell = QGraphicsPathItem(_hexagon_path)
                    cell.setPos(x + (y % 2) / 2, y * fy)
                else:
                    cell = QGraphicsRectItem(x - 0.5, y - 0.5, 1, 1)
                self.grid_cells[y, x] = cell
                cell.setPen(self._grid_pen)
                self.grid.addToGroup(cell)
        self.scene.addItem(self.grid)

    def get_member_indices(self, x, y):
        i, j = self.cells[x, y]
        return self.member_data[i:j]

    def _recompute_som(self):
        if self.cont_x is None:
            return

        class Optimizer(QObject):
            update = Signal(float, SOM)
            done = Signal(SOM)
            stopped = Signal()

            def __init__(self, data, widget):
                super().__init__()
                self.som = SOM(
                    widget.size_x, widget.size_y,
                    hexagonal=widget.hexagonal,
                    pca_init=widget.initialization == 0,
                    random_seed=0 if widget.initialization == 2 else None)
                self.data = data
                self.widget = widget

            def callback(self, progress):
                self.update.emit(progress, self.som)
                return not self.widget._stop_optimization

            def run(self):
                self.som.fit(self.data, N_ITERATIONS, callback=self.callback)
                self.done.emit(self.som)
                self.stopped.emit()

        def update(progress, som):
            from AnyQt.QtWidgets import qApp
            progressbar.advance()
            qApp.processEvents()  # This is apparently needed to advance the bar
            self._assign_instances(som)
            self._redraw()

        def done(som):
            self.set_buttons(running=False)
            progressbar.finish()
            self._assign_instances(som)
            self._redraw()
            # This is the first time we know what was selected (assuming that
            # initialization is not set to random)
            if self.__pending_selection is not None:
                self.on_selection_change(self.__pending_selection)
                self.__pending_selection = None

        def thread_finished():
            self._optimizer = None
            self._optimizer_thread = None

        progressbar = gui.ProgressBar(self, N_ITERATIONS)
        self.set_buttons(running=True)

        self._optimizer = Optimizer(self.cont_x, self)
        self._optimizer_thread = QThread()
        self._optimizer.update.connect(update)
        self._optimizer.done.connect(done)
        self._optimizer.stopped.connect(self._optimizer_thread.quit)
        self._optimizer.moveToThread(self._optimizer_thread)
        self._optimizer_thread.started.connect(self._optimizer.run)
        self._optimizer_thread.finished.connect(thread_finished)
        self._stop_optimization = False
        self._optimizer_thread.start()

    def stop_optimization_and_wait(self):
        if self._optimizer_thread is not None:
            self._stop_optimization = True
            self._optimizer_thread.quit()
            self._optimizer_thread.wait()
            self._optimizer_thread = None

    def onDeleteWidget(self):
        self.stop_optimization_and_wait()
        super().onDeleteWidget()

    def set_buttons(self, running):
        self.restart_button.setText("Stop" if running else "Restart")
        self.grid_box.setDisabled(running)

    def _assign_instances(self, som):
        self.assignments = som.winners(self.cont_x)
        members = defaultdict(list)
        for i, (x, y) in enumerate(self.assignments):
            members[(x, y)].append(i)
        members.pop(None, None)
        self.cells = np.empty((self.size_x, self.size_y, 2), dtype=int)
        self.member_data = np.empty(self.cont_x.shape[0], dtype=int)
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
        if self.hexagonal:
            scale = min(vw / (self.size_x + 1),
                        vh / ((self.size_y + 1) * (np.sqrt(3) / 2)))
            self.scene.setSceneRect(
                0, -1, self.size_x -1, self.size_y - 0.5)
        else:
            scale = min(vw / (self.size_x + 1), vh / (self.size_y + 1))
            self.scene.setSceneRect(
                -0.25, -0.25, self.size_x - 0.5, self.size_y - 0.5)
        self.view.setTransform(QTransform.fromScale(scale, scale))

    def update_output(self):
        indices = []
        if self.data is not None:
            for (x, y) in self.selection:
                indices.extend(self.get_member_indices(x, y))
        if indices:
            self.Outputs.selected_data.send(self.data[indices])
            self.Outputs.annotated_data.send(
                create_annotated_table(self.data, indices))
            self.info.set_output_summary(str(len(indices)))
        else:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            self.info.set_output_summary(self.info.NoOutput)


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
    # If run on sparse data, the widget core dumps if the user tries resizing it?!
    # WidgetPreview(OWSOM).run(Table("/Users/janez/Downloads/deerwester.pkl"))
