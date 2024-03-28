from collections import defaultdict, namedtuple
from contextlib import contextmanager
from typing import Optional, Union
from xml.sax.saxutils import escape

import numpy as np

from AnyQt.QtCore import Qt, QRectF, pyqtSignal as Signal, QObject, QThread, \
    pyqtSlot as Slot
from AnyQt.QtGui import QTransform, QPen, QBrush, QColor, QPainter, \
    QPainterPath, QKeyEvent
from AnyQt.QtWidgets import \
    QGraphicsView, QGraphicsScene, \
    QGraphicsItem, QGraphicsRectItem, QGraphicsItemGroup, QSizePolicy, \
    QGraphicsPathItem

from Orange.widgets.utils.signals import lazy_table_transform

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.data.util import array_equal, SharedComputeValue, get_unique_names
from Orange.preprocess import decimal_binnings, time_binnings
from Orange.projection.som import SOM

from Orange.widgets import gui
from Orange.widgets.utils.localization import pl
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.settings import \
    DomainContextHandler, ContextSetting, Setting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.annotated_data import \
    add_columns, group_values, \
    ANNOTATED_DATA_SIGNAL_NAME, ANNOTATED_DATA_FEATURE_NAME
from Orange.widgets.utils.colorpalettes import \
    BinnedContinuousPalette, LimitedDiscretePalette, DiscretePalette
from Orange.widgets.visualize.utils import CanvasRectangle, CanvasText
from Orange.widgets.visualize.utils.plotutils import wrap_legend_items


sqrt3_2 = np.sqrt(3) / 2


class SomView(QGraphicsView):
    SelectionSet, SelectionNewGroup, SelectionAddToGroup, SelectionRemove \
        = range(4)
    selection_changed = Signal(np.ndarray, int)
    selection_moved = Signal(QKeyEvent)
    selection_mark_changed = Signal(np.ndarray)

    def __init__(self, scene):
        super().__init__(scene)
        self.__button_down_pos = None
        self.size_x = self.size_y = 1
        self.hexagonal = True

    def set_dimensions(self, size_x, size_y, hexagonal):
        self.size_x = size_x
        self.size_y = size_y
        self.hexagonal = hexagonal

    def _get_marked_cells(self, event):
        x0, y0 = self.__button_down_pos.x(), self.__button_down_pos.y()
        pos = self.mapToScene(event.pos())
        x1, y1 = pos.x(), pos.y()
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        selection = np.zeros((self.size_x, self.size_y), dtype=bool)
        if self.hexagonal:
            y0 = max(0, int(y0 / sqrt3_2 + 0.5))
            y1 = min(self.size_y, int(np.ceil(y1 / sqrt3_2 + 0.5)))
            for y in range(y0, y1):
                if x1 < 0:
                    continue
                x0_ = max(0, int(x0 + 0.5 - (y % 2) / 2))
                x1_ = min(self.size_x - y % 2,
                          int(np.ceil(x1 + 0.5 - (y % 2) / 2)))
                selection[x0_:x1_, y] = True
        elif not(x1 < -0.5 or x0 > self.size_x - 0.5
                 or y1 < -0.5 or y0 > self.size_y - 0.5):

            def roundclip(z, zmax):
                return int(np.clip(np.round(z), 0, zmax - 1))\

            x0 = roundclip(x0, self.size_x)
            y0 = roundclip(y0, self.size_y)
            x1 = roundclip(x1, self.size_x)
            y1 = roundclip(y1, self.size_y)
            selection[x0:x1 + 1, y0:y1 + 1] = True

        return selection

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        self.__button_down_pos = self.mapToScene(event.pos())
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() != Qt.LeftButton:
            return

        self.selection_mark_changed.emit(self._get_marked_cells(event))
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        if event.modifiers() & Qt.ShiftModifier:
            if event.modifiers() & Qt.ControlModifier:
                action = self.SelectionAddToGroup
            else:
                action = self.SelectionNewGroup
        elif event.modifiers() & Qt.AltModifier:
            action = self.SelectionRemove
        else:
            action = self.SelectionSet
        selection = self._get_marked_cells(event)
        self.selection_changed.emit(selection, action)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            self.selection_moved.emit(event)
        else:
            super().keyPressEvent(event)


class PieChart(QGraphicsItem):
    def __init__(self, dist, r, colors):
        super().__init__()
        self.dist = dist
        self.r = r
        self.colors = colors

    def boundingRect(self):
        return QRectF(- self.r, - self.r, 2 * self.r, 2 * self.r)

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


class ColoredCircle(QGraphicsItem):
    def __init__(self, r, color, proportion):
        super().__init__()
        self.r = r
        self.color = color
        self.proportion = proportion

    def boundingRect(self):
        return QRectF(-self.r, -self.r, 2 * self.r, 2 * self.r)

    def paint(self, painter, _option, _index):
        painter.save()
        pen = QPen(QBrush(self.color), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color.lighter(int(200 - 80 * self.proportion))))
        painter.drawEllipse(self.boundingRect())
        painter.restore()


@contextmanager
def disconnected_spin(spin):
    spin.blockSignals(True)
    try:
        yield
    finally:
        spin.blockSignals(False)


N_ITERATIONS = 200


class SomSharedValueCompute:
    def __init__(self, domain: Domain, model: SOM,
                 offsets: Union[np.ndarray, None],
                 scales: Union[np.ndarray, None]):
        # offsets and scales are made immutable so that they are hashable
        def immutable(a):
            if a is not None:
                a = a.copy()
                a.flags.writeable = False
            return a

        self.domain = domain
        self.model = model
        self.offsets = immutable(offsets)
        self.scales = immutable(scales)
        self.__hash = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["__hash"] = None
        return state

    def __call__(self, data):
        x = data.transform(self.domain).X
        cont_x, mask, _, _ = SOM.prepare_data(x, self.offsets, self.scales)
        winners = np.full((len(data), 2), np.nan)
        distances = np.full(len(data), np.nan)
        winners[mask], distances[mask] = self.model.winners(cont_x)
        winners += 1
        return winners, distances

    # `offsets` and `scales` are ndarray's (and `model` contains ndarray's)
    # Unless we __eq__ by identity, we can't properly define hash.
    def __eq__(self, other):
        return type(self) is type(other) and \
               self.domain == other.domain \
               and self.model == other.model \
               and np.array_equal(self.offsets, other.offsets) \
               and np.array_equal(self.scales, other.scales)

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash((
                self.domain, self.model,
                tuple(self.offsets), tuple(self.scales)))
        return self.__hash


class SomCellCompute(SharedComputeValue):
    def __init__(self, compute_shared, dim_x, hexagonal):
        super().__init__(compute_shared)
        self.dim_x = dim_x
        self.hexagonal = hexagonal

    def __eq__(self, other):
        return super().__eq__(other) \
            and self.dim_x == other.dim_x \
            and self.hexagonal == other.hexagonal

    def __hash__(self):
        return hash((super().__hash__(), self.dim_x, self.hexagonal))

    def compute(self, _, shared_data):
        coords = shared_data[0]
        # coords are 1-based, subtract 1
        col = (coords[:, 0] - 1) + (coords[:, 1] - 1) * self.dim_x
        if self.hexagonal:
            # 1 cell less after every two rows
            col -= col // (self.dim_x * 2 - 1)
        return col


class SomCoordsCompute(SharedComputeValue):
    def __init__(self, shared, column):
        super().__init__(shared)
        self.column = column

    def compute(self, _, shared_data):
        return shared_data[0][:, self.column]

    def __eq__(self, other):
        return self.column == other.column and super().__eq__(other)

    def __hash__(self):
        return hash((super().__hash__(), self.column))


class SomErrorCompute(SharedComputeValue):
    InheritEq = True

    def compute(self, _, shared_data):
        return shared_data[1]


class GetGroups:
    # This assigns instances to selection groups; two instances that are not
    # same cannot be considered equal, period.
    InheritEq = True

    def __init__(self, id_to_group, default, offset):
        self.id_to_group = id_to_group
        self.default = default + offset
        self.offset = offset

    def __call__(self, data, *args, **kwargs):
        return np.array([self.id_to_group.get(id, self.default) - self.offset
                        for id in data.ids])


class OWSOM(OWWidget):
    name = "Self-Organizing Map"
    description = "Computation of self-organizing map."
    icon = "icons/SOM.svg"
    keywords = "self-organizing map, som"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settingsHandler = DomainContextHandler()
    auto_dimension = Setting(True)
    size_x = Setting(10)
    size_y = Setting(10)
    hexagonal = Setting(1)
    initialization = Setting(0)

    attr_color = ContextSetting(None)
    size_by_instances = Setting(True)
    pie_charts = Setting(False)
    selection = Setting(None, schema_only=True)

    graph_name = "view"  # QGraphicsView

    _grid_pen = QPen(QBrush(QColor(224, 224, 224)), 2)
    _grid_pen.setCosmetic(True)

    OptControls = namedtuple(
        "OptControls",
        ("shape", "auto_dim", "spin_x", "spin_y", "initialization", "start")
    )

    class Information(OWWidget.Information):
        modified = Msg(
            'The parameter settings have been changed. Press "Start" to '
            "rerun with the new settings."
        )

    class Warning(OWWidget.Warning):
        ignoring_disc_variables = Msg("SOM ignores categorical variables.")
        missing_colors = \
            Msg("Some data instances have undefined value of '{}'.")
        no_defined_colors = \
            Msg("'{}' has no defined values.")
        missing_values = Msg("{}")
        single_attribute = Msg("Data contains a single numeric column.")

    class Error(OWWidget.Error):
        no_numeric_variables = Msg("Data contains no numeric columns.")
        not_enough_data = Msg("SOM needs at least two data rows without missing values.")

    def __init__(self):
        super().__init__()
        self.__pending_selection = self.selection
        self._optimizer = None
        self._optimizer_thread = None
        self.stop_optimization = False

        self.data = self.cont_x = None
        self.scales = self.offsets = None
        self.som = self.cells = self.member_data = None
        self.selection = None

        # self.colors holds a palette or None when we need to draw same-colored
        # circles. This happens by user's choice or when the color attribute
        # is numeric and has no defined values, so we can't construct bins
        self.colors: Optional[DiscretePalette] = None
        self.thresholds = self.bin_labels = None

        box = gui.vBox(self.controlArea, box="SOM")
        shape = gui.comboBox(
            box, self, "", items=("Hexagonal grid", "Square grid"))
        shape.setCurrentIndex(1 - self.hexagonal)
        shape.currentIndexChanged.connect(self.on_parameter_change)

        box2 = gui.indentedBox(box, 10)
        auto_dim = gui.checkBox(
            box2, self, "auto_dimension", "Set dimensions automatically",
            callback=self.on_auto_dimension_changed)
        self.manual_box = box3 = gui.hBox(box2)
        spinargs = dict(
            value="",
            widget=box3,
            master=self,
            minv=5,
            maxv=100,
            step=5,
            alignment=Qt.AlignRight,
            callback=self.on_parameter_change,
        )
        self.spin_x = gui.spin(**spinargs)
        with disconnected_spin(self.spin_x):
            self.spin_x.setValue(self.size_x)
        gui.widgetLabel(box3, "×")
        self.spin_y = gui.spin(**spinargs)
        with disconnected_spin(self.spin_y):
            self.spin_y.setValue(self.size_y)
        gui.rubber(box3)
        self.manual_box.setEnabled(not self.auto_dimension)

        initialization = gui.comboBox(
            box,
            self,
            "initialization",
            items=("Initialize with PCA", "Random initialization", "Replicable random"),
            callback=self.on_parameter_change,
        )

        start = gui.button(
            box, self, "Restart", callback=self.restart_som_pressed,
            sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        self.opt_controls = self.OptControls(
            shape, auto_dim, self.spin_x, self.spin_y, initialization, start
        )

        box = gui.vBox(self.controlArea, "Color")
        gui.comboBox(
            box, self, "attr_color", searchable=True,
            callback=self.on_attr_color_change,
            model=DomainModel(placeholder="(Same color)",
                              valid_types=DomainModel.PRIMITIVE))
        gui.checkBox(
            box, self, "pie_charts", label="Show pie charts",
            callback=self.on_pie_chart_change)
        gui.checkBox(
            box, self, "size_by_instances", label="Size by number of instances",
            callback=self.on_attr_size_change)

        gui.rubber(self.controlArea)

        self.scene = QGraphicsScene(self)

        self.view = SomView(self.scene)
        self.view.setMinimumWidth(400)
        self.view.setMinimumHeight(400)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.selection_changed.connect(self.on_selection_change)
        self.view.selection_moved.connect(self.on_selection_move)
        self.view.selection_mark_changed.connect(self.on_selection_mark_change)
        self.mainArea.layout().addWidget(self.view)

        self.elements = None
        self.grid = None
        self.grid_cells = None
        self.legend = None

    @staticmethod
    def _cont_domain(data):
        attrs = data.domain.attributes
        cont_attrs = [var for var in attrs if var.is_continuous]
        if not cont_attrs:
            return None
        return Domain(cont_attrs)

    @Inputs.data
    def set_data(self, data):
        def set_warnings():
            missing = len(data) - len(self.data)
            if missing:
                self.Warning.missing_values(
                    f'{missing} data {pl(missing, "instance")} with undefined value(s) {pl(missing, "is|are")} not shown.')

        cont_x = self.cont_x.copy() if self.cont_x is not None else None
        self.data = self.cont_x = None
        self.offsets = self.scales = None
        new_cont_x = None
        self.closeContext()
        self.clear_messages()

        if data is not None:
            cont_domain = self._cont_domain(data)
            if cont_domain is None:
                self.Error.no_numeric_variables()
            else:
                cont_attrs = cont_domain.attributes
                if len(cont_attrs) < len(data.domain.attributes):
                    self.Warning.ignoring_disc_variables()
                if len(cont_attrs) == 1:
                    self.Warning.single_attribute()

                new_cont_x, mask, self.offsets, self.scales \
                    = SOM.prepare_data(data.transform(cont_domain).X)
                rows = np.sum(mask)
                if rows == len(mask):
                    self.data = data
                elif rows > 1:
                    self.data = data[mask]
                else:
                    self.Error.not_enough_data()

        invalidated = cont_x is None or new_cont_x is None \
            or not array_equal(cont_x, new_cont_x)
        if invalidated:
            self.stop_optimization_and_wait()
            self.clear()

        if self.data is not None:
            self.cont_x = new_cont_x
            self.controls.attr_color.model().set_domain(data.domain)
            self.attr_color = data.domain.class_var
            set_warnings()

        self.openContext(self.data)
        self.set_color_bins()
        self.create_legend()
        if invalidated:
            with disconnected_spin(self.spin_x), disconnected_spin(self.spin_y):
                self.recompute_dimensions()
            self.start_som()
        else:
            self._redraw()

    def clear(self):
        self.som = self.cont_x = None
        self.cells = self.member_data = None
        self.attr_color = None
        self.colors = self.thresholds = self.bin_labels = None
        if self.elements is not None:
            self.scene.removeItem(self.elements)
            self.elements = None
        self.clear_selection()
        self.controls.attr_color.model().set_domain(None)

    def recompute_dimensions(self):
        if not self.auto_dimension or self.cont_x is None:
            return
        dim = max(5, int(np.ceil(np.sqrt(5 * np.sqrt(self.cont_x.shape[0])))))
        self.opt_controls.spin_x.setValue(dim)
        self.opt_controls.spin_y.setValue(dim)

    def on_auto_dimension_changed(self):
        self.manual_box.setEnabled(not self.auto_dimension)
        if self.auto_dimension:
            self.recompute_dimensions()
        else:
            spin_x = self.opt_controls.spin_x
            spin_y = self.opt_controls.spin_y
            dimx = int(5 * np.round(spin_x.value() / 5))
            dimy = int(5 * np.round(spin_y.value() / 5))
            spin_x.setValue(dimx)
            spin_y.setValue(dimy)
        self.on_parameter_change()

    def on_attr_color_change(self):
        self.controls.pie_charts.setEnabled(self.attr_color is not None)
        self.set_color_bins()
        self.create_legend()
        self.rescale()
        self._redraw()

    def on_attr_size_change(self):
        self._redraw()

    def on_pie_chart_change(self):
        self._redraw()

    def on_parameter_change(self):
        self.Information.modified()

    def clear_selection(self):
        self.selection = None
        self.redraw_selection()

    def on_selection_change(self, selection, action=SomView.SelectionSet):
        if self.data is None:  # clicks on empty canvas
            return
        if self.selection is None:
            self.selection = np.zeros(self.grid_cells.T.shape, dtype=np.int16)

        selection_np = np.array(self.selection)
        if action == SomView.SelectionSet:
            selection_np[:] = 0
            selection_np[selection] = 1
        elif action == SomView.SelectionAddToGroup:
            selection_np[selection] = max(1, np.max(selection_np))
        elif action == SomView.SelectionNewGroup:
            selection_np[selection] = 1 + np.max(selection_np)
        elif action & SomView.SelectionRemove:
            selection_np[selection] = 0
        self.selection = selection_np.tolist()
        self.redraw_selection()
        self.update_output()

    def on_selection_move(self, event: QKeyEvent):
        if self.selection is None or not np.any(self.selection):
            if event.key() in (Qt.Key_Right, Qt.Key_Down):
                x = y = 0
            else:
                x = self.size_x - 1
                y = self.size_y - 1
        else:
            x, y = np.nonzero(self.selection)
            if len(x) > 1:
                return
            x, y = x[0], y[0]
            if event.key() == Qt.Key_Up and y > 0:
                y -= 1
            if event.key() == Qt.Key_Down and y < self.size_y - 1:
                y += 1
            if event.key() == Qt.Key_Left and x:
                x -= 1
            if event.key() == Qt.Key_Right and x < self.size_x - 1:
                x += 1
            x -= self.hexagonal and x == self.size_x - 1 and y % 2

        if self.selection is not None and self.selection[x][y]:
            return
        selection = np.zeros(self.grid_cells.shape, dtype=bool)
        selection[x, y] = True
        self.on_selection_change(selection.tolist())

    def on_selection_mark_change(self, marks):
        self.redraw_selection(marks=marks)

    def redraw_selection(self, marks=None):
        if self.grid_cells is None:
            return

        sel_pen = QPen(QBrush(QColor(128, 128, 128)), 2)
        sel_pen.setCosmetic(True)
        mark_pen = QPen(QBrush(QColor(128, 128, 128)), 4)
        mark_pen.setCosmetic(True)
        pens = [self._grid_pen, sel_pen]

        mark_brush = QBrush(QColor(224, 255, 255))
        sels = self.selection is not None and np.max(self.selection)
        palette = LimitedDiscretePalette(number_of_colors=sels + 1)
        brushes = [QBrush(Qt.NoBrush)] + \
                  [QBrush(palette[i].lighter(165)) for i in range(sels)]

        for y in range(self.size_y):
            for x in range(self.size_x - (y % 2) * self.hexagonal):
                cell = self.grid_cells[y, x]
                marked = marks is not None and marks[x, y]
                sel_group = self.selection is not None and self.selection[x][y]
                if marked:
                    cell.setBrush(mark_brush)
                    cell.setPen(mark_pen)
                else:
                    cell.setBrush(brushes[sel_group])
                    cell.setPen(pens[bool(sel_group)])
                cell.setZValue(marked or sel_group)

    def restart_som_pressed(self):
        self.Information.modified.clear()
        if self._optimizer_thread is not None:
            self.stop_optimization = True
            self._optimizer.stop_optimization = True
        else:
            self.start_som()

    def start_som(self):
        self.read_controls()
        self.update_layout()
        self.clear_selection()
        if self.cont_x is not None:
            self.enable_controls(False)
            self._recompute_som()
        else:
            self.update_output()

    def read_controls(self):
        c = self.opt_controls
        self.hexagonal = c.shape.currentIndex() == 0
        self.size_x = c.spin_x.value()
        self.size_y = c.spin_y.value()

    def enable_controls(self, enable):
        c = self.opt_controls
        c.shape.setEnabled(enable)
        c.auto_dim.setEnabled(enable)
        c.start.setText("Start" if enable else "Stop")

    def update_layout(self):
        self.set_legend_pos()
        if self.elements:  # Prevent having redrawn grid but with old elements
            self.scene.removeItem(self.elements)
            self.elements = None
        self.redraw_grid()
        self.rescale()

    def _redraw(self):
        self.Warning.missing_colors.clear()
        if self.elements:
            self.scene.removeItem(self.elements)
            self.elements = None
        self.view.set_dimensions(self.size_x, self.size_y, self.hexagonal)

        if self.cells is None:
            return
        sizes = self.cells[:, :, 1] - self.cells[:, :, 0]
        sizes = sizes.astype(float)
        if not self.size_by_instances:
            sizes[sizes != 0] = 0.8
        else:
            sizes *= 0.8 / np.max(sizes)

        self.elements = QGraphicsItemGroup()
        self.scene.addItem(self.elements)
        if self.colors is None:
            self._draw_same_color(sizes)
        elif self.pie_charts:
            self._draw_pie_charts(sizes)
        else:
            self._draw_colored_circles(sizes)

    @property
    def _grid_factors(self):
        return (0.5, sqrt3_2) if self.hexagonal else (0, 1)

    def _draw_same_color(self, sizes):
        fx, fy = self._grid_factors
        color = QColor(64, 64, 64)
        for y in range(self.size_y):
            for x in range(self.size_x - self.hexagonal * (y % 2)):
                r = sizes[x, y]
                n = len(self.get_member_indices(x, y))
                if not r:
                    continue
                ellipse = ColoredCircle(r / 2, color, 0)
                ellipse.setPos(x + (y % 2) * fx, y * fy)
                ellipse.setToolTip(f"{n} instances")
                self.elements.addToGroup(ellipse)

    def _get_color_column(self):
        # if self.colors is None, we use _draw_same_color and don't call
        # this function
        assert self.colors is not None

        color_column = self.data.get_column(self.attr_color)
        if self.attr_color.is_discrete:
            with np.errstate(invalid="ignore"):
                int_col = color_column.astype(int)
            int_col[np.isnan(color_column)] = len(self.colors)
        else:
            int_col = np.zeros(len(color_column), dtype=int)
            int_col[np.isnan(color_column)] = len(self.colors)
            for i, thresh in enumerate(self.thresholds, start=1):
                int_col[color_column >= thresh] = i
        return int_col

    def _tooltip(self, colors, distribution):
        if self.attr_color.is_discrete:
            values = self.attr_color.values
        else:
            values = self._bin_names()
        values = list(values) + ["(N/A)"]
        tot = np.sum(distribution)
        nbhp = "\N{NON-BREAKING HYPHEN}"
        return '<table style="white-space: nowrap">' + "".join(f"""
            <tr>
                <td>
                    <font color={color.name()}>■</font>
                    <b>{escape(val).replace("-", nbhp)}</b>:
                </td>
                <td>
                    {n} ({n / tot * 100:.1f}&nbsp;%)
                </td>
            </tr>
            """ for color, val, n in zip(colors, values, distribution) if n) \
            + "</table>"

    def _draw_pie_charts(self, sizes):
        assert self.colors is not None  # if it were, we'd call _draw_same_color

        fx, fy = self._grid_factors
        color_column = self._get_color_column()
        colors = self.colors.qcolors_w_nan
        for y in range(self.size_y):
            for x in range(self.size_x - self.hexagonal * (y % 2)):
                r = sizes[x, y]
                if not r:
                    self.grid_cells[y, x].setToolTip("")
                    continue
                members = self.get_member_indices(x, y)
                color_dist = np.bincount(color_column[members],
                                         minlength=len(colors))
                rel_color_dist = color_dist.astype(float) / len(members)
                pie = PieChart(rel_color_dist, r / 2, colors)
                pie.setToolTip(self._tooltip(colors, color_dist))
                self.elements.addToGroup(pie)
                pie.setPos(x + (y % 2) * fx, y * fy)

    def _draw_colored_circles(self, sizes):
        assert self.colors is not None  # if it were, we'd call _draw_same_color

        fx, fy = self._grid_factors
        color_column = self._get_color_column()
        qcolors = self.colors.qcolors_w_nan
        for y in range(self.size_y):
            for x in range(self.size_x - self.hexagonal * (y % 2)):
                r = sizes[x, y]
                if not r:
                    continue
                members = self.get_member_indices(x, y)
                color_dist = color_column[members]
                color_dist = color_dist[color_dist < len(self.colors)]
                if len(color_dist) != len(members):
                    self.Warning.missing_colors(self.attr_color.name)
                bc = np.bincount(color_dist, minlength=len(self.colors))
                color = qcolors[np.argmax(bc)]
                ellipse = ColoredCircle(r / 2, color, np.max(bc) / len(members))
                ellipse.setPos(x + (y % 2) * fx, y * fy)
                ellipse.setToolTip(self._tooltip(qcolors, bc))
                self.elements.addToGroup(ellipse)

    def redraw_grid(self):
        if self.grid is not None:
            self.scene.removeItem(self.grid)
        self.grid = QGraphicsItemGroup()
        self.grid.setZValue(-200)
        self.grid_cells = np.full((self.size_y, self.size_x), None)
        for y in range(self.size_y):
            for x in range(self.size_x - (y % 2) * self.hexagonal):
                if self.hexagonal:
                    cell = QGraphicsPathItem(_hexagon_path)
                    cell.setPos(x + (y % 2) / 2, y * sqrt3_2)
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

        self.som = SOM(
            self.size_x, self.size_y,
            hexagonal=self.hexagonal,
            pca_init=self.initialization == 0,
            random_seed=0 if self.initialization == 2 else None
        )

        class Optimizer(QObject):
            update = Signal(float, np.ndarray, np.ndarray)
            done = Signal(SOM)
            stopped = Signal()
            stop_optimization = False

            def __init__(self, data, som):
                super().__init__()
                self.som = som
                self.data = data

            def callback(self, progress):
                self.update.emit(
                    progress,
                    self.som.weights.copy(), self.som.ssum_weights.copy())
                return not self.stop_optimization

            def run(self):
                try:
                    self.som.fit(self.data, N_ITERATIONS,
                                 callback=self.callback)
                    # Report an exception, but still remove the thread
                finally:
                    self.done.emit(self.som)
                    self.stopped.emit()

        def thread_finished():
            self._optimizer = None
            self._optimizer_thread = None

        self.progressBarInit()
        self.setInvalidated(True)

        self._optimizer = Optimizer(self.cont_x, self.som)
        self._optimizer_thread = QThread()
        self._optimizer_thread.setStackSize(5 * 2 ** 20)
        self._optimizer.update.connect(self.__update)
        self._optimizer.done.connect(self.__done)
        self._optimizer.stopped.connect(self._optimizer_thread.quit)
        self._optimizer.moveToThread(self._optimizer_thread)
        self._optimizer_thread.started.connect(self._optimizer.run)
        self._optimizer_thread.finished.connect(thread_finished)
        self.stop_optimization = False
        self._optimizer_thread.start()

    @Slot(float, object, object)
    def __update(self, _progress, weights, ssum_weights):
        self.progressBarSet(_progress)
        self._assign_instances(weights, ssum_weights)
        self._redraw()

    @Slot(object)
    def __done(self, som):
        self.enable_controls(True)
        self.progressBarFinished()
        self.setInvalidated(False)
        self._assign_instances(som.weights, som.ssum_weights)
        self._redraw()
        # This is the first time we know what was selected (assuming that
        # initialization is not set to random)
        if self.__pending_selection is not None:
            self.on_selection_change(self.__pending_selection)
            self.__pending_selection = None
        self.update_output()

    def stop_optimization_and_wait(self):
        if self._optimizer_thread is not None:
            self.stop_optimization = True
            self._optimizer.stop_optimization = True
            self._optimizer_thread.quit()
            self._optimizer_thread.wait()
            self._optimizer_thread = None

    def onDeleteWidget(self):
        self.stop_optimization_and_wait()
        self.clear()
        super().onDeleteWidget()

    def _assign_instances(self, weights, ssum_weights):
        if self.cont_x is None:
            return  # the widget is shutting down while signals still processed
        assignments, _ = SOM.winner_from_weights(
            self.cont_x, weights, ssum_weights, self.hexagonal)
        members = defaultdict(list)
        for i, (x, y) in enumerate(assignments):
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
        self.create_legend()  # re-wrap lines if necessary
        self.rescale()

    def rescale(self):
        if self.legend:
            leg_height = self.legend.boundingRect().height()
            leg_extra = 1.5
        else:
            leg_height = 0
            leg_extra = 1

        vw, vh = self.view.width(), self.view.height() - leg_height
        scale = min(vw / (self.size_x + 1),
                    vh / ((self.size_y + leg_extra) * self._grid_factors[1]))
        self.view.setTransform(QTransform.fromScale(scale, scale))
        if self.hexagonal:
            self.view.setSceneRect(
                # -1.5: 1 is necessary, 0.5 is to add some border
                -0.5, -1.5 / np.sqrt(3), self.size_x,
                (self.size_y + leg_extra) * sqrt3_2 + leg_height / scale - 1 / np.sqrt(3))
        else:
            self.view.setSceneRect(
                -0.25, -0.25, self.size_x - 0.5,
                self.size_y - 0.5 + leg_height / scale)

    def update_output(self):
        if self.som is None:
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(None)
            return

        ngroups = int(self.selection is not None) and np.max(self.selection)
        indices = np.zeros(len(self.data), dtype=int)
        id_to_group = {}
        if self.selection is not None and np.any(self.selection):
            for y in range(self.size_y):
                for x in range(self.size_x):
                    rows = self.get_member_indices(x, y)
                    group = self.selection[x][y]
                    indices[rows] = group
                    if group > 0:
                        for id_ in self.data.ids[rows]:
                            id_to_group[id_] = group

        cont_domain = self._cont_domain(self.data)
        shared_compute = SomSharedValueCompute(
            cont_domain, self.som, self.offsets, self.scales)
        cell = DiscreteVariable(
            "som_cell",
            values=tuple(
                f"r{row + 1}c{col + 1}"
                for row in range(self.size_y)
                for col in range(self.size_x - (self.hexagonal and row % 2))),
            compute_value=SomCellCompute(shared_compute,
                                         self.size_x, self.hexagonal))
        coordx = ContinuousVariable(
            "som_row",
            number_of_decimals=0,
            compute_value=SomCoordsCompute(shared_compute, 1))
        coordy = ContinuousVariable(
            "som_col",
            number_of_decimals=0,
            compute_value=SomCoordsCompute(shared_compute, 0))
        error = ContinuousVariable(
            "som_error",
            compute_value=SomErrorCompute(shared_compute)
        )
        som_attrs = (cell, coordx, coordy, error)

        grp_values, _ = group_values(
                indices, include_unselected=False, values=None)

        def make_domain(values, default_grp, offset):
            grp_var = DiscreteVariable(
                get_unique_names(self.data.domain, ANNOTATED_DATA_FEATURE_NAME),
                values,
                compute_value=GetGroups(id_to_group, default_grp, offset))

            if not self.data.domain.class_vars:
                class_vars, metas = (grp_var,), som_attrs
            else:
                class_vars, metas = (), (grp_var,) + som_attrs
            return add_columns(self.data.domain, (), class_vars, metas)

        if np.any(indices):
            sel_domain = make_domain(grp_values, np.nan, 1)
            mask = np.flatnonzero(indices)
            self.Outputs.selected_data.send(
                lazy_table_transform(sel_domain, self.data[mask]))
        else:
            self.Outputs.selected_data.send(None)

        if ngroups > 1:
            sel_domain = make_domain(grp_values + ["Unselected", ], ngroups, 1)
        else:
            sel_domain = make_domain(("No", "Yes"), 0, 0)
        self.Outputs.annotated_data.send(
            lazy_table_transform(sel_domain, self.data))

    def set_color_bins(self):
        self.Warning.no_defined_colors.clear()

        if self.attr_color is None:
            self.thresholds = self.bin_labels = self.colors = None
            return

        if self.attr_color.is_discrete:
            self.thresholds = self.bin_labels = None
            self.colors = self.attr_color.palette
            return

        col = self.data.get_column(self.attr_color)
        col = col[np.isfinite(col)]
        if not col.size:
            self.Warning.no_defined_colors(self.attr_color)
            self.thresholds = self.bin_labels = self.colors = None
            return

        if self.attr_color.is_time:
            binning = time_binnings(col, min_bins=4)[-1]
        else:
            binning = decimal_binnings(col, min_bins=4)[-1]
        self.thresholds = binning.thresholds[1:-1]
        self.bin_labels = (binning.labels[1:-1], binning.short_labels[1:-1])
        if not self.bin_labels[0] and binning.labels:
            # Nan's are already filtered out, but it doesn't hurt much
            # to use nanmax/nanmin
            if np.nanmin(col) == np.nanmax(col):
                # Handle a degenerate case with a single value
                # Use the second threshold (because value must be smaller),
                # but the first threshold as label (because that's the
                # actual value in the data.
                self.thresholds = binning.thresholds[1:]
                self.bin_labels = (binning.labels[:1],
                                   binning.short_labels[:1])
        palette = BinnedContinuousPalette.from_palette(
            self.attr_color.palette, binning.thresholds)
        self.colors = palette

    def create_legend(self):
        if self.legend is not None:
            self.scene.removeItem(self.legend)
            self.legend = None
        if self.colors is None:
            return

        if self.attr_color.is_discrete:
            names = self.attr_color.values
        else:
            names = self._bin_names()

        items = []
        size = 8
        for name, color in zip(names, self.colors.qcolors):
            item = QGraphicsItemGroup()
            item.addToGroup(
                CanvasRectangle(None, -size / 2, -size / 2, size, size,
                                Qt.gray, color))
            item.addToGroup(CanvasText(None, name, size, 0, Qt.AlignVCenter))
            items.append(item)

        self.legend = wrap_legend_items(
            items, hspacing=20, vspacing=16 + size,
            max_width=self.view.width() - 25)
        self.legend.setFlags(self.legend.ItemIgnoresTransformations)
        self.legend.setTransform(
            QTransform.fromTranslate(-self.legend.boundingRect().width() / 2,
                                     0))
        self.scene.addItem(self.legend)
        self.set_legend_pos()

    def _bin_names(self):
        labels, short_labels = self.bin_labels or ([], [])
        if len(labels) <= 1:
            return labels
        return \
            [f"< {labels[0]}"] \
            + [f"{x} - {y}" for x, y in zip(labels, short_labels[1:])] \
            + [f"≥ {labels[-1]}"]

    def set_legend_pos(self):
        if self.legend is None:
            return
        self.legend.setPos(
            self.size_x / 2,
            (self.size_y + 0.2 + 0.3 * self.hexagonal) * self._grid_factors[1])

    def send_report(self):
        self.report_plot()
        if self.colors:
            self.report_caption(
                f"Self-organizing map colored by '{self.attr_color.name}'")

    @classmethod
    def migrate_settings(cls, settings, _):
        # previously selection was saved as np.ndarray which is not supported
        # by widget-base, change selection to list
        selection = settings.get('selection')
        if selection is not None and isinstance(selection, np.ndarray):
            settings['selection'] = selection.tolist()


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
    WidgetPreview(OWSOM).run(Table("iris"))
