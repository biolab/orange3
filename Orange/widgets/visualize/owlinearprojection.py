"""
Linear projection widget
------------------------

"""

from functools import reduce
from operator import itemgetter
from types import SimpleNamespace as namespace
from xml.sax.saxutils import escape
import pkg_resources

import numpy

from AnyQt.QtWidgets import (
    QListView, QSlider, QToolButton, QFormLayout, QHBoxLayout,
    QSizePolicy, QAction, QActionGroup, QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsRectItem, QPinchGesture, QApplication
)
from AnyQt.QtGui import (
    QColor, QPen, QBrush, QKeySequence, QPainterPath, QPainter, QTransform,
    QCursor, QIcon
)
from AnyQt.QtCore import Qt, QObject, QEvent, QSize, QRectF, QLineF, QPointF, QMimeData
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import pyqtgraph.graphicsItems.ScatterPlotItem
import pyqtgraph as pg

from Orange.data import Table, Variable
from Orange.data.sql.table import SqlTable
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import colorpalette
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.visualize.owscatterplotgraph import LegendItem, \
    legend_anchor_pos
from Orange.widgets.utils import classdensity
from Orange.widgets.widget import Input, Output
from Orange.canvas import report


class ScatterPlotItem(pg.ScatterPlotItem):
    Symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paint(self, painter, option, widget=None):
        if self.opts["pxMode"]:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self.opts["antialias"]:
            painter.setRenderHint(QPainter.Antialiasing, True)

        super().paint(painter, option, widget)


class TextItem(pg.TextItem):
    if not hasattr(pg.TextItem, "setAnchor"):
        # Compatibility with pyqtgraph <= 0.9.10; in (as of yet unreleased)
        # 0.9.11 the TextItem has a `setAnchor`, but not `updateText`
        def setAnchor(self, anchor):
            self.anchor = pg.Point(anchor)
            self.updateText()


class AxisItem(pg.GraphicsObject):
    def __init__(self, parent=None, line=None, label=None, *args):
        super().__init__(parent, *args)
        self.setFlag(pg.GraphicsObject.ItemHasNoContents)

        if line is None:
            line = QLineF(0, 0, 1, 0)

        self._spine = QGraphicsLineItem(line, self)
        dx = line.x2() - line.x1()
        dy = line.y2() - line.y1()
        rad = numpy.arctan2(dy, dx)
        angle = (rad * 180 / numpy.pi) % 360

        self._arrow = pg.ArrowItem(parent=self, angle=180 - angle)
        self._arrow.setPos(self._spine.line().p2())

        self._label = TextItem(text=label, color=(10, 10, 10))
        self._label.setParentItem(self)
        self._label.setPos(self._spine.line().p2())

    def setLabel(self, label):
        if label != self._label:
            self._label = label
            self._label.setText(label)

    def setPen(self, pen):
        self._spine.setPen(pen)

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()

    def viewTransformChanged(self):
        self.__updateLabelPos()

    def __updateLabelPos(self):
        T = self.viewTransform()
        if T is not None:
            Tinv, ok = T.inverted()
        else:
            Tinv, ok = None, False
        if not ok:
            T = Tinv = QTransform()

        # map the axis spine to viewbox coord. system
        viewbox_line = Tinv.map(self._spine.line())
        angle = viewbox_line.angle()
        # note in Qt the y axis is inverted (90 degree angle 'points' down)
        left_quad = 270 <= angle <= 360 or -0.0 <= angle < 90

        # position the text label along the viewbox_line
        label_pos = viewbox_line.pointAt(0.90)

        if left_quad:
            anchor = (0.5, -0.1)
        else:
            anchor = (0.5, 1.1)

        pos = T.map(label_pos)
        self._label.setPos(pos)
        self._label.setAnchor(pg.Point(*anchor))
        self._label.setRotation(angle if left_quad else angle - 180)


class LegendItem(LegendItem):
    def __init__(self):
        super().__init__()
        self.items = []

    def clear(self):
        """
        Clear all legend items.
        """
        items = list(self.items)
        self.items = []
        for sample, label in items:
            # yes, the LegendItem shadows QGraphicsWidget.layout() with
            # an instance attribute.
            self.layout.removeItem(sample)
            self.layout.removeItem(label)
            sample.hide()
            label.hide()

        self.updateSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            event.accept()
            if self.parentItem() is not None:
                self.autoAnchor(
                    self.pos() + (event.pos() - event.lastPos()) / 2)
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            event.accept()
        else:
            event.ignore()


class OWLinearProjection(widget.OWWidget):
    name = "Linear Projection"
    description = "A multi-axis projection of data onto " \
                  "a two-dimensional plane."
    icon = "icons/LinearProjection.svg"
    priority = 240

    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

#              #TODO: Allow for axes to be supplied from an external source.
#               ("Projection", numpy.ndarray, "set_axes"),]

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settings_version = 2
    settingsHandler = settings.DomainContextHandler()

    variable_state = settings.ContextSetting({})

    attr_color = settings.ContextSetting(None, exclude_metas=False)
    attr_label = settings.ContextSetting(None, exclude_metas=False)
    attr_shape = settings.ContextSetting(None, exclude_metas=False)
    attr_size = settings.ContextSetting(None, exclude_metas=False)

    point_width = settings.Setting(10)
    alpha_value = settings.Setting(128)
    jitter_value = settings.Setting(0)

    class_density = settings.Setting(False)
    resolution = 256

    auto_commit = settings.Setting(True)

    legend_anchor = settings.Setting(((1, 0), (1, 0)))
    MinPointSize = 6

    ReplotRequest = QEvent.registerEventType()

    graph_name = "viewbox"

    def __init__(self):
        super().__init__()

        self.data = None
        self.subset_data = None
        self._subset_mask = None
        self._selection_mask = None
        self._item = None
        self._density_img = None
        self.__legend = None
        self.__selection_item = None
        self.__replot_requested = False

        box = gui.vBox(self.controlArea, "Axes")

        box1 = gui.vBox(box, "Displayed", margin=0)
        box1.setFlat(True)
        self.active_view = view = QListView(
            sizePolicy=QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored),
            selectionMode=QListView.ExtendedSelection,
            dragEnabled=True,
            defaultDropAction=Qt.MoveAction,
            dragDropOverwriteMode=False,
            dragDropMode=QListView.DragDrop,
            showDropIndicator=True,
            minimumHeight=100,
        )

        view.viewport().setAcceptDrops(True)
        movedown = QAction(
            "Move down", view,
            shortcut=QKeySequence(Qt.AltModifier | Qt.Key_Down),
            triggered=self.__deactivate_selection
        )
        view.addAction(movedown)

        self.varmodel_selected = model = VariableListModel(
            parent=self, enable_dnd=True)

        model.rowsInserted.connect(self._invalidate_plot)
        model.rowsRemoved.connect(self._invalidate_plot)
        model.rowsMoved.connect(self._invalidate_plot)

        view.setModel(model)

        box1.layout().addWidget(view)

        box1 = gui.vBox(box, "Other", margin=0)
        box1.setFlat(True)
        self.other_view = view = QListView(
            sizePolicy=QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored),
            selectionMode=QListView.ExtendedSelection,
            dragEnabled=True,
            defaultDropAction=Qt.MoveAction,
            dragDropOverwriteMode=False,
            dragDropMode=QListView.DragDrop,
            showDropIndicator=True,
            minimumHeight=100
        )
        view.viewport().setAcceptDrops(True)
        moveup = QAction(
            "Move up", view,
            shortcut=QKeySequence(Qt.AltModifier | Qt.Key_Up),
            triggered=self.__activate_selection
        )
        view.addAction(moveup)

        self.varmodel_other = model = VariableListModel(parent=self, enable_dnd=True)
        view.setModel(model)

        box1.layout().addWidget(view)

        # Main area plot
        self.view = pg.GraphicsView(background="w")
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setFrameStyle(pg.GraphicsView.StyledPanel)
        self.viewbox = pg.ViewBox(enableMouse=True, enableMenu=False)
        self.viewbox.setAspectLocked(True)
        self.viewbox.grabGesture(Qt.PinchGesture)
        self.view.setCentralItem(self.viewbox)
        self.mainArea.layout().addWidget(self.view)
        self.replot = None

        g = OWPlotGUI(self)
        g.point_properties_box(self.controlArea)
        self.controls.attr_label.parent().setVisible(False)
        self.models = g.points_models

        box = gui.widgetBox(self.controlArea, "Plot")
        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            spacing=8
        )
        box.layout().addLayout(form)

        self.jitter_combo = gui.comboBox(
            box, self, "jitter_value",
            items=["None", "0.01 %", "0.1 %", "0.5 %", "1 %", "2 %"],
            callback=self._invalidate_plot)
        form.addRow("Jittering:", self.jitter_combo)

        self.cb_class_density = gui.checkBox(
            box, self, "class_density", "", callback=self._update_density)
        form.addRow("Class density", self.cb_class_density)

        toolbox = gui.vBox(self.controlArea, "Zoom/Select")
        toollayout = QHBoxLayout()
        toolbox.layout().addLayout(toollayout)

        self.controlArea.layout().addStretch(1)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Send Selection",
                        auto_label="Send Automatically")

        self.selection = PlotSelectionTool(self)
        self.selection.setViewBox(self.viewbox)
        self.selection.selectionFinished.connect(self._selection_finish)

        self.zoomtool = PlotZoomTool(self)
        self.pantool = PlotPanTool(self)
        self.pinchtool = PlotPinchZoomTool(self)
        self.pinchtool.setViewBox(self.viewbox)

        def icon(name):
            path = "icons/Dlg_{}.png".format(name)
            path = pkg_resources.resource_filename(widget.__name__, path)
            return QIcon(path)

        actions = namespace(
            zoomtofit=QAction(
                "Zoom to fit", self, icon=icon("zoom_reset"),
                shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
                triggered=lambda:
                    self.viewbox.setRange(QRectF(-1.05, -1.05, 2.1, 2.1))),
            zoomin=QAction(
                "Zoom in", self,
                shortcut=QKeySequence(QKeySequence.ZoomIn),
                triggered=lambda: self.viewbox.scaleBy((1 / 1.25, 1 / 1.25))),
            zoomout=QAction(
                "Zoom out", self,
                shortcut=QKeySequence(QKeySequence.ZoomOut),
                triggered=lambda: self.viewbox.scaleBy((1.25, 1.25))),
            select=QAction(
                "Select", self, checkable=True, icon=icon("arrow"),
                shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_1)),
            zoom=QAction(
                "Zoom", self, checkable=True, icon=icon("zoom"),
                shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_2)),
            pan=QAction(
                "Pan", self, checkable=True, icon=icon("pan_hand"),
                shortcut=QKeySequence(Qt.ControlModifier + Qt.Key_3)),
        )
        self.addActions([actions.zoomtofit, actions.zoomin, actions.zoomout])

        group = QActionGroup(self, exclusive=True)
        group.addAction(actions.select)
        group.addAction(actions.zoom)
        group.addAction(actions.pan)

        actions.select.setChecked(True)

        currenttool = self.selection

        def activated(action):
            nonlocal currenttool
            if action is actions.select:
                tool, cursor = self.selection, Qt.ArrowCursor
            elif action is actions.zoom:
                tool, cursor = self.zoomtool, Qt.ArrowCursor
            elif action is actions.pan:
                tool, cursor = self.pantool, Qt.OpenHandCursor
            else:
                assert False
            currenttool.setViewBox(None)
            tool.setViewBox(self.viewbox)
            self.viewbox.setCursor(QCursor(cursor))
            currenttool = tool

        group.triggered[QAction].connect(activated)

        def button(action):
            b = QToolButton()
            b.setDefaultAction(action)
            return b

        toollayout.addWidget(button(actions.select))
        toollayout.addWidget(button(actions.zoom))
        toollayout.addWidget(button(actions.pan))

        toollayout.addSpacing(4)
        toollayout.addWidget(button(actions.zoomtofit))
        toollayout.addStretch()
        toolbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def sizeHint(self):
        return QSize(800, 500)

    def clear(self):
        self.data = None
        self._subset_mask = None
        self._selection_mask = None
        self.varmodel_selected[:] = []
        self.varmodel_other[:] = []
        self.clear_plot()

    def clear_item(self):
        if self._item is not None:
            self._item.setParentItem(None)
            self.viewbox.removeItem(self._item)
            self._item = None

    def clear_density_img(self):
        if self._density_img is not None:
            self._density_img.setParentItem(None)
            self.viewbox.removeItem(self._density_img)
            self._density_img = None

    def clear_legend(self):
        if self.__legend is not None:
            anchor = legend_anchor_pos(self.__legend)
            if anchor is not None:
                self.legend_anchor = anchor

            self.__legend.setParentItem(None)
            self.__legend.clear()
            self.__legend.setVisible(False)

    def clear_plot(self):
        self.clear_item()
        self.clear_density_img()
        self.clear_legend()
        self.viewbox.clear()

    def _invalidate_plot(self):
        """
        Schedule a delayed replot.
        """
        if not self.__replot_requested:
            self.__replot_requested = True
            QApplication.postEvent(self, QEvent(self.ReplotRequest),
                                   Qt.LowEventPriority - 10)

    def init_attr_values(self):
        domain = self.data and len(self.data) and self.data.domain or None
        for model in self.models:
            model.set_domain(domain)
        self.attr_color = domain and self.data.domain.class_var or None
        self.attr_shape = None
        self.attr_size = None
        self.attr_label = None

    @Inputs.data
    def set_data(self, data):
        """
        Set the input dataset.

        Args:
            data (Orange.data.table): data instances
        """
        self.closeContext()
        self.clear()
        self.information()
        if isinstance(data, SqlTable):
            if data.approx_len() < 4000:
                data = Table(data)
            else:
                self.information("Data has been sampled")
                data_sample = data.sample_time(1, no_cache=True)
                data_sample.download_data(2000, partial=True)
                data = Table(data_sample)
        self.data = data
        self.init_attr_values()
        if data is not None and len(data):
            self._initialize(data)
            # get the default encoded state, replacing the position with Inf
            state = self._encode_var_state(
                [list(self.varmodel_selected), list(self.varmodel_other)]
            )
            state = {key: (source_ind, numpy.inf) for key, (source_ind, _)
                     in state.items()}

            self.openContext(data.domain)
            selected_keys = [key
                             for key, (sind, _) in self.variable_state.items()
                             if sind == 0]

            if set(selected_keys).issubset(set(state.keys())):
                pass

            # update the defaults state (the encoded state must contain
            # all variables in the input domain)
            state.update(self.variable_state)
            # ... and restore it with saved positions taking precedence over
            # the defaults
            selected, other = self._decode_var_state(
                state, [list(self.varmodel_selected),
                        list(self.varmodel_other)]
            )
            self.varmodel_selected[:] = selected
            self.varmodel_other[:] = other
            self._invalidate_plot()

    @Inputs.data_subset
    def set_subset_data(self, subset):
        """
        Set the supplementary input subset dataset.

        Args:
            subset (Orange.data.table): subset of data instances
        """
        self.subset_data = subset
        self._subset_mask = None

    def handleNewSignals(self):
        if self.subset_data is not None and self._subset_mask is None:
            # Update the plot's highlight items
            if self.data is not None:
                dataids = self.data.ids.ravel()
                subsetids = numpy.unique(self.subset_data.ids)
                self._subset_mask = numpy.in1d(
                    dataids, subsetids, assume_unique=True)
                self._invalidate_plot()

        self.commit()

    def customEvent(self, event):
        if event.type() == OWLinearProjection.ReplotRequest:
            self.__replot_requested = False
            self._setup_plot()
        else:
            super().customEvent(event)

    def closeContext(self):
        self.variable_state = self._encode_var_state(
            [list(self.varmodel_selected), list(self.varmodel_other)]
        )
        super().closeContext()

    @staticmethod
    def _encode_var_state(lists):
        return {(type(var), var.name): (source_ind, pos)
                for source_ind, var_list in enumerate(lists)
                for pos, var in enumerate(var_list)
                if isinstance(var, Variable)}

    @staticmethod
    def _decode_var_state(state, lists):
        all_vars = reduce(list.__iadd__, lists, [])

        newlists = [[] for _ in lists]
        for var in all_vars:
            source, pos = state[(type(var), var.name)]
            newlists[source].append((pos, var))
        return [[var for _, var in sorted(newlist, key=itemgetter(0))]
                for newlist in newlists]

    def _initialize(self, data):
        # Initialize the GUI controls from data's domain.
        cont_vars = [var for var in data.domain.variables
                     if var.is_continuous]
        self.warning("Plotting requires continuous features.",
                     shown=not len(cont_vars))
        self.varmodel_selected[:] = cont_vars[:3]
        self.varmodel_other[:] = cont_vars[3:]

    def __activate_selection(self):
        view = self.other_view
        model = self.varmodel_other
        indices = view.selectionModel().selectedRows()

        variables = [model.data(ind, Qt.EditRole) for ind in indices]

        for i in sorted((ind.row() for ind in indices), reverse=True):
            del model[i]

        self.varmodel_selected.extend(variables)

    def __deactivate_selection(self):
        view = self.active_view
        model = self.varmodel_selected
        indices = view.selectionModel().selectedRows()

        variables = [model.data(ind, Qt.EditRole) for ind in indices]

        for i in sorted((ind.row() for ind in indices), reverse=True):
            del model[i]

        self.varmodel_other.extend(variables)

    def _get_data(self, var, dtype):
        """
        Return the column data and mask for variable `var`
        """
        X, _ = self.data.get_column_view(var)
        return column_data(self.data, var, dtype)

    def _setup_plot(self, reset_view=True):
        self.__replot_requested = False
        self.clear_plot()

        variables = list(self.varmodel_selected)
        if not variables:
            return

        coords = [self._get_data(var, dtype=float)[0] for var in variables]
        coords = numpy.vstack(coords)
        p, N = coords.shape
        assert N == len(self.data), p == len(variables)

        axes = linproj.defaultaxes(len(variables))

        assert axes.shape == (2, p)

        mask = ~numpy.logical_or.reduce(numpy.isnan(coords), axis=0)
        coords = coords[:, mask]

        X, Y = numpy.dot(axes, coords)
        if X.size and Y.size:
            X = plotutils.normalized(X)
            Y = plotutils.normalized(Y)

        pen_data, brush_data = self._color_data(mask)
        size_data = self._size_data(mask)
        shape_data = self._shape_data(mask)

        if self.jitter_value > 0:
            value = [0, 0.01, 0.1, 0.5, 1, 2][self.jitter_value]

            rstate = numpy.random.RandomState(0)
            jitter_x = (rstate.random_sample(X.shape) * 2 - 1) * value / 100
            rstate = numpy.random.RandomState(1)
            jitter_y = (rstate.random_sample(Y.shape) * 2 - 1) * value / 100
            X += jitter_x
            Y += jitter_y

        self._item = ScatterPlotItem(
            X, Y,
            pen=pen_data,
            brush=brush_data,
            size=size_data,
            symbol=shape_data,
            antialias=True,
            data=numpy.arange(len(self.data))[mask]
        )
        self._item._mask = mask

        self.viewbox.addItem(self._item)

        for i, axis in enumerate(axes.T):
            axis_item = AxisItem(line=QLineF(0, 0, axis[0], axis[1]),
                                 label=variables[i].name)
            axis_item.setPen(QPen(Qt.darkGray, 0))
            self.viewbox.addItem(axis_item)

        if reset_view:
            self.viewbox.setRange(QRectF(-1.05, -1.05, 2.1, 2.1))
        self._update_legend()

        if self.class_density and \
                self.attr_color is not None and self.attr_color.is_discrete:
            [min_x, max_x], [min_y, max_y] = self.viewbox.viewRange()
            rgb_data = [brush.color().getRgb()[:3] for brush in brush_data]
            self._density_img = classdensity.class_density_image(
                min_x, max_x, min_y, max_y, self.resolution, X, Y, rgb_data)
            self.viewbox.addItem(self._density_img)

    def _color_data(self, mask=None):
        if self.attr_color is not None:
            color_data, _ = self._get_data(self.attr_color, dtype=float)
            if self.attr_color.is_continuous:
                color_data = plotutils.continuous_colors(
                    color_data, None, *self.attr_color.colors)
            else:
                color_data = plotutils.discrete_colors(
                    color_data, len(self.attr_color.values),
                    color_index=self.attr_color.colors
                )
            if mask is not None:
                color_data = color_data[mask]

            pen_data = numpy.array(
                [pg.mkPen((r, g, b), width=1.5)
                 for r, g, b in color_data * 0.8],
                dtype=object)

            brush_data = numpy.array(
                [pg.mkBrush((r, g, b, self.alpha_value))
                 for r, g, b in color_data],
                dtype=object)
        else:
            color = QColor(Qt.darkGray)
            pen_data = QPen(color, 1.5)
            pen_data.setCosmetic(True)
            color = QColor(Qt.lightGray)
            color.setAlpha(self.alpha_value)
            brush_data = QBrush(color)

        if self._subset_mask is not None:
            assert self._subset_mask.shape == (len(self.data),)
            if mask is not None:
                subset_mask = self._subset_mask[mask]
            else:
                subset_mask = self._subset_mask

            if isinstance(brush_data, QBrush):
                brush_data = numpy.array([brush_data] * subset_mask.size,
                                         dtype=object)

            brush_data[~subset_mask] = QBrush(Qt.NoBrush)

        if self._selection_mask is not None:
            assert self._selection_mask.shape == (len(self.data),)
            if mask is not None:
                selection_mask = self._selection_mask[mask]
            else:
                selection_mask = self._selection_mask

            if isinstance(pen_data, QPen):
                pen_data = numpy.array([pen_data] * selection_mask.size,
                                       dtype=object)

            pen_data[selection_mask] = pg.mkPen((200, 200, 0, 150), width=4)
        return pen_data, brush_data

    def _on_color_change(self):
        if self.data is None or self._item is None:
            return

        pen, brush = self._color_data()

        if isinstance(pen, QPen):
            # Reset the brush for all points
            self._item.data["pen"] = None
            self._item.setPen(pen)
        else:
            self._item.setPen(pen[self._item._mask])

        if isinstance(brush, QBrush):
            # Reset the brush for all points
            self._item.data["brush"] = None
            self._item.setBrush(brush)
        else:
            self._item.setBrush(brush[self._item._mask])

        if self.attr_color is not None and self.attr_color.is_discrete:
            self.cb_class_density.setEnabled(True)
            if self.class_density:
                self._setup_plot(reset_view=False)
        else:
            self.clear_density_img()
            self.cb_class_density.setEnabled(False)

        self._update_legend()

    def _shape_data(self, mask):
        if self.attr_shape is None:
            shape_data = numpy.array(["o"] * len(self.data))
        else:
            assert self.attr_shape.is_discrete
            symbols = numpy.array(list(ScatterPlotItem.Symbols))
            max_symbol = symbols.size - 1
            shapeidx, shape_mask = column_data(self.data, self.attr_shape,
                                               dtype=int)
            shapeidx[shape_mask] = max_symbol
            shapeidx[~shape_mask] %= max_symbol -1
            shape_data = symbols[shapeidx]
        if mask is None:
            return shape_data
        else:
            return shape_data[mask]

    def _on_shape_change(self):
        if self.data is None:
            return

        self.set_shape(self._shape_data(mask=None))
        self._update_legend()

    def _size_data(self, mask=None):
        if self.attr_size is None:
            size_data = numpy.full((len(self.data),), self.point_width,
                                   dtype=float)
        else:
            nan_size = OWLinearProjection.MinPointSize - 2
            size_data, size_mask = self._get_data(self.attr_size, dtype=float)
            size_data_valid = size_data[~size_mask]
            if size_data_valid.size:
                smin, smax = numpy.min(size_data_valid), numpy.max(size_data_valid)
                sspan = smax - smin
            else:
                sspan = smin = 0
            size_data[~size_mask] -= smin
            if sspan > 0:
                size_data[~size_mask] /= sspan
            size_data = \
                size_data * self.point_width + OWLinearProjection.MinPointSize
            size_data[size_mask] = nan_size
        if mask is None:
            return size_data
        else:
            return size_data[mask]

    def _on_size_change(self):
        if self.data is None:
            return
        self.set_size(self._size_data(mask=None))

    update_point_size = update_sizes = _on_size_change
    update_alpha_value = update_colors = _on_color_change
    update_shapes = _on_shape_change

    def _update_density(self):
        self._setup_plot(reset_view=False)

    def _update_legend(self):
        if self.__legend is None:
            self.__legend = legend = LegendItem()
            legend.setParentItem(self.viewbox)
            legend.setZValue(self.viewbox.zValue() + 10)
            legend.restoreAnchor(self.legend_anchor)
        else:
            legend = self.__legend

        legend.clear()

        color_var, shape_var = self.attr_color, self.attr_shape
        if color_var is not None and not color_var.is_discrete:
            color_var = None
        assert shape_var is None or shape_var.is_discrete
        if color_var is None and shape_var is None:
            legend.setParentItem(None)
            legend.hide()
            return
        else:
            if legend.parentItem() is None:
                legend.setParentItem(self.viewbox)
            legend.setVisible(True)

        symbols = list(ScatterPlotItem.Symbols)

        if shape_var is color_var:
            items = [(QColor(*color_var.colors[i]), symbols[i], name)
                     for i, name in enumerate(color_var.values)]
        else:
            colors = shapes = []
            if color_var is not None:
                colors = [(QColor(*color_var.colors[i]), "o", name)
                          for i, name in enumerate(color_var.values)]
            if shape_var is not None:
                shapes = [(QColor(Qt.gray),
                           symbols[i % (len(symbols) - 1)], name)
                          for i, name in enumerate(shape_var.values)]
            items = colors + shapes

        for color, symbol, name in items:
            legend.addItem(
                ScatterPlotItem(pen=color, brush=color, symbol=symbol, size=10),
                escape(name)
            )

    def set_shape(self, shape):
        """
        Set (update) the current point shape map.
        """
        if self._item is not None:
            self._item.setSymbol(shape[self._item._mask])

    def set_size(self, size):
        """
        Set (update) the current point size.
        """
        if self._item is not None:
            self._item.setSize(size[self._item._mask])

    def _set_alpha(self, value):
        self.alpha_value = value
        self._on_color_change()

    def _set_size(self, value):
        self.point_size = value
        self._on_size_change()

    def _selection_finish(self, path):
        self.select(path)

    def select(self, selectionshape):
        item = self._item
        if item is None:
            return

        indices = [spot.data()
                   for spot in item.points()
                   if selectionshape.contains(spot.pos())]

        self.select_indices(indices, QApplication.keyboardModifiers())

    def select_indices(self, indices, modifiers=Qt.NoModifier):
        if self.data is None:
            return

        if self._selection_mask is None or \
                not modifiers & (Qt.ControlModifier | Qt.ShiftModifier |
                                 Qt.AltModifier):
            self._selection_mask = numpy.zeros(len(self.data), dtype=bool)

        if modifiers & Qt.AltModifier:
            self._selection_mask[indices] = False
        elif modifiers & Qt.ControlModifier:
            self._selection_mask[indices] = ~self._selection_mask[indices]
        else:
            self._selection_mask[indices] = True

        self._on_color_change()
        self.commit()

    def commit(self):
        subset = None
        indices = None
        if self.data is not None and self._selection_mask is not None:
            indices = numpy.flatnonzero(self._selection_mask)
            if len(indices) > 0:
                subset = self.data[indices]

        self.Outputs.selected_data.send(subset)
        self.Outputs.annotated_data.send(create_annotated_table(self.data, indices))

    def send_report(self):
        self.report_plot(name="", plot=self.viewbox.getViewBox())
        caption = report.render_items_vert((
            ("Colors", self.attr_color),
            ("Shape", self.attr_shape),
            ("Size", self.attr_size)
        ))
        jitter_caption = report.render_items_vert(
            (("Jittering",
              self.jitter_value > 0 and self.jitter_combo.currentText()),))
        caption = ";<br/>".join(x for x in (caption, jitter_caption) if x)
        self.report_caption(caption)

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            settings_["point_width"] = settings_["point_size"]

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            domain = context.ordered_domain
            c_domain = [t for t in context.ordered_domain if t[1] == 2]
            d_domain = [t for t in context.ordered_domain if t[1] == 1]
            for d, old_val, new_val in ((domain, "color_index", "attr_color"),
                                        (d_domain, "shape_index", "attr_shape"),
                                        (c_domain, "size_index", "attr_size")):
                index = context.values[old_val][0] - 1
                context.values[new_val] = (d[index][0], d[index][1] + 100) \
                    if 0 <= index < len(d) else None


class PlotTool(QObject):
    """
    An abstract tool operating on a pg.ViewBox.

    Subclasses of `PlotTool` implement various actions responding to
    user input events. For instance `PlotZoomTool` when active allows
    the user to select/draw a rectangular region on a plot in which to
    zoom.

    The tool works by installing itself as an `eventFilter` on to the
    `pg.ViewBox` instance and dispatching events to the appropriate
    event handlers `mousePressEvent`, ...

    When subclassing note that the event handlers (`mousePressEvent`, ...)
    are actually event filters and need to return a boolean value
    indicating if the event was handled (filtered) and should not propagate
    further to the view box.

    See Also
    --------
    QObject.eventFilter

    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__viewbox = None

    def setViewBox(self, viewbox):
        """
        Set the view box to operate on.

        Call ``setViewBox(None)`` to remove the tool from the current
        view box. If an existing view box is already set it is first
        removed.

        .. note::
            The PlotTool will install itself as an event filter on the
            view box.

        Parameters
        ----------
        viewbox : pg.ViewBox or None

        """
        if self.__viewbox is viewbox:
            return
        if self.__viewbox is not None:
            self.__viewbox.removeEventFilter(self)
            self.__viewbox.destroyed.disconnect(self.__viewdestroyed)

        self.__viewbox = viewbox

        if self.__viewbox is not None:
            self.__viewbox.installEventFilter(self)
            self.__viewbox.destroyed.connect(self.__viewdestroyed)

    def viewBox(self):
        """
        Return the view box.

        Returns
        -------
        viewbox : pg.ViewBox
        """
        return self.__viewbox

    @Slot(QObject)
    def __viewdestroyed(self, _):
        self.__viewbox = None

    def mousePressEvent(self, event):
        """
        Handle a mouse press event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseMoveEvent(self, event):
        """
        Handle a mouse move event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseReleaseEvent(self, event):
        """
        Handle a mouse release event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseDoubleClickEvent(self, event):
        """
        Handle a mouse double click event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def gestureEvent(self, event):
        """
        Handle a gesture event.

        Parameters
        ----------
        event : QGraphicsSceneGestureEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def eventFilter(self, obj, event):
        """
        Reimplemented from `QObject.eventFilter`.
        """
        if obj is self.__viewbox:
            if event.type() == QEvent.GraphicsSceneMousePress:
                return self.mousePressEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseMove:
                return self.mouseMoveEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseRelease:
                return self.mouseReleaseEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseDoubleClick:
                return self.mouseDoubleClickEvent(event)
            elif event.type() == QEvent.Gesture:
                return self.gestureEvent(event)
        return super().eventFilter(obj, event)


class PlotSelectionTool(PlotTool):
    """
    A tool for selecting a region on a plot.

    """
    #: Selection modes
    Rect, Lasso = 1, 2

    #: Selection was started by the user.
    selectionStarted = Signal(QPainterPath)
    #: The current selection has been updated
    selectionUpdated = Signal(QPainterPath)
    #: The selection has finished (user has released the mouse button)
    selectionFinished = Signal(QPainterPath)

    def __init__(self, parent=None, selectionMode=Rect, **kwargs):
        super().__init__(parent, **kwargs)
        self.__mode = selectionMode
        self.__path = None
        self.__item = None

    def setSelectionMode(self, mode):
        """
        Set the selection mode (rectangular or lasso selection).

        Parameters
        ----------
        mode : int
            PlotSelectionTool.Rect or PlotSelectionTool.Lasso

        """
        assert mode in {PlotSelectionTool.Rect, PlotSelectionTool.Lasso}
        if self.__mode != mode:
            if self.__path is not None:
                self.selectionFinished.emit()
            self.__mode = mode
            self.__path = None

    def selectionMode(self):
        """
        Return the current selection mode.
        """
        return self.__mode

    def selectionShape(self):
        """
        Return the current selection shape.

        This is the area selected/drawn by the user.

        Returns
        -------
        shape : QPainterPath
            The selection shape in view coordinates.
        """
        if self.__path is not None:
            shape = QPainterPath(self.__path)
            shape.closeSubpath()
        else:
            shape = QPainterPath()
        viewbox = self.viewBox()

        if viewbox is None:
            return QPainterPath()

        return viewbox.childGroup.mapFromParent(shape)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(pos, pos)
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path = QPainterPath()
                self.__path.moveTo(event.pos())
            self.selectionStarted.emit(self.selectionShape())
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path.lineTo(event.pos())
            self.selectionUpdated.emit(self.selectionShape())
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path.lineTo(event.pos())
            self.selectionFinished.emit(self.selectionShape())
            self.__path = QPainterPath()
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def __updategraphics(self):
        viewbox = self.viewBox()
        if viewbox is None:
            return

        if self.__path.isEmpty():
            if self.__item is not None:
                self.__item.setParentItem(None)
                viewbox.removeItem(self.__item)
                if self.__item.scene():
                    self.__item.scene().removeItem(self.__item)
                self.__item = None
        else:
            if self.__item is None:
                item = QGraphicsPathItem()
                color = QColor(Qt.yellow)
                item.setPen(QPen(color, 0))
                color.setAlpha(50)
                item.setBrush(QBrush(color))
                self.__item = item
                viewbox.addItem(item)

            self.__item.setPath(self.selectionShape())


class PlotZoomTool(PlotTool):
    """
    A zoom tool.

    Allows the user to draw a rectangular region to zoom in.
    """

    zoomStarted = Signal(QRectF)
    zoomUpdated = Signal(QRectF)
    zoomFinished = Signal(QRectF)

    def __init__(self, parent=None, autoZoom=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.__zoomrect = QRectF()
        self.__zoomitem = None
        self.__autozoom = autoZoom

    def zoomRect(self):
        """
        Return the current drawn rectangle (region of interest)

        Returns
        -------
        zoomrect : QRectF
        """
        view = self.viewBox()
        if view is None:
            return QRectF()
        return view.childGroup.mapRectFromParent(self.__zoomrect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__zoomrect = QRectF(event.pos(), event.pos())
            self.zoomStarted.emit(self.zoomRect())
            self.__updategraphics()
            event.accept()
            return True
        elif event.button() == Qt.RightButton:
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.__zoomrect = QRectF(
                event.buttonDownPos(Qt.LeftButton), event.pos()).normalized()
            self.zoomUpdated.emit(self.zoomRect())
            self.__updategraphics()
            event.accept()
            return True
        elif event.buttons() & Qt.RightButton:
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__zoomrect = QRectF(
                event.buttonDownPos(Qt.LeftButton), event.pos()).normalized()

            if self.__autozoom:
                PlotZoomTool.pushZoomRect(self.viewBox(), self.zoomRect())

            self.zoomFinished.emit(self.zoomRect())
            self.__zoomrect = QRectF()
            self.__updategraphics()
            event.accept()
            return True
        elif event.button() == Qt.RightButton:
            PlotZoomTool.popZoomStack(self.viewBox())
            event.accept()
            return True
        else:
            return False

    def __updategraphics(self):
        viewbox = self.viewBox()
        if viewbox is None:
            return
        if not self.__zoomrect.isValid():
            if self.__zoomitem is not None:
                self.__zoomitem.setParentItem(None)
                viewbox.removeItem(self.__zoomitem)
                if self.__zoomitem.scene() is not None:
                    self.__zoomitem.scene().removeItem(self.__zoomitem)
                self.__zoomitem = None
        else:
            if self.__zoomitem is None:
                self.__zoomitem = QGraphicsRectItem()
                color = QColor(Qt.yellow)
                self.__zoomitem.setPen(QPen(color, 0))
                color.setAlpha(50)
                self.__zoomitem.setBrush(QBrush(color))
                viewbox.addItem(self.__zoomitem)

            self.__zoomitem.setRect(self.zoomRect())

    @staticmethod
    def pushZoomRect(viewbox, rect):
        viewbox.showAxRect(rect)
        viewbox.axHistoryPointer += 1
        viewbox.axHistory[viewbox.axHistoryPointer:] = [rect]

    @staticmethod
    def popZoomStack(viewbox):
        if viewbox.axHistoryPointer == 0:
            viewbox.autoRange()
            viewbox.axHistory = []
            viewbox.axHistoryPointer = -1
        else:
            viewbox.scaleHistory(-1)


class PlotPanTool(PlotTool):
    """
    Pan/translate tool.
    """
    panStarted = Signal()
    translated = Signal(QPointF)
    panFinished = Signal()

    def __init__(self, parent=None, autoPan=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.__autopan = autoPan

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panStarted.emit()
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            viewbox = self.viewBox()
            delta = (viewbox.mapToView(event.pos()) -
                     viewbox.mapToView(event.lastPos()))
            if self.__autopan:
                viewbox.translateBy(-delta / 2)
            self.translated.emit(-delta / 2)
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panFinished.emit()
            event.accept()
            return True
        else:
            return False


class PlotPinchZoomTool(PlotTool):
    """
    A tool implementing a "Pinch to zoom".
    """

    def gestureEvent(self, event):
        gesture = event.gesture(Qt.PinchGesture)
        if gesture.state() == Qt.GestureStarted:
            event.accept(gesture)
            return True
        elif gesture.changeFlags() & QPinchGesture.ScaleFactorChanged:
            viewbox = self.viewBox()
            center = viewbox.mapSceneToView(gesture.centerPoint())
            scale_prev = gesture.lastScaleFactor()
            scale = gesture.scaleFactor()
            if scale_prev != 0:
                scale = scale / scale_prev
            if scale > 0:
                viewbox.scaleBy((1 / scale, 1 / scale), center)
            event.accept()
            return True
        elif gesture.state() == Qt.GestureFinished:
            viewbox = self.viewBox()
            PlotZoomTool.pushZoomRect(viewbox, viewbox.viewRect())
            event.accept()
            return True
        else:
            return False


def column_data(table, var, dtype):
    dtype = numpy.dtype(dtype)
    col, copy = table.get_column_view(var)
    if var.is_primitive() and not isinstance(col.dtype.type, numpy.inexact):
        # from mixes metas domain
        col = col.astype(float)
        copy = True
    mask = numpy.isnan(col)
    if dtype != col.dtype:
        col = col.astype(dtype)
        copy = True

    if not copy:
        col = col.copy()
    return col, mask


class plotutils:
    @staticmethod
    def continuous_colors(data, palette=None,
                          low=(220, 220, 220), high=(0,0,0),
                          through_black=False):
        if palette is None:
            palette = colorpalette.ContinuousPaletteGenerator(
                QColor(*low), QColor(*high), through_black)
        amin, amax = numpy.nanmin(data), numpy.nanmax(data)
        span = amax - amin
        data = (data - amin) / (span or 1)
        return palette.getRGB(data)

    @staticmethod
    def discrete_colors(data, nvalues, palette=None, color_index=None):
        if color_index is None:
            if palette is None or nvalues >= palette.number_of_colors:
                palette = colorpalette.ColorPaletteGenerator(nvalues)
            color_index = palette.getRGB(numpy.arange(nvalues))
        # Unknown values as gray
        # TODO: This should already be a part of palette
        color_index = numpy.vstack((color_index, [[128, 128, 128]]))

        data = numpy.where(numpy.isnan(data), nvalues, data)
        data = data.astype(int)
        return color_index[data]

    @staticmethod
    def normalized(a):
        if not a.size:
            return a.copy()
        amin, amax = numpy.nanmin(a), numpy.nanmax(a)
        if numpy.isnan(amin):
            return a.copy()
        span = amax - amin
        mean = numpy.nanmean(a)
        return (a - mean) / (span or 1)


class linproj:
    @staticmethod
    def defaultaxes(naxes):
        """
        Return the default axes for linear projection.
        """
        assert naxes > 0

        if naxes == 1:
            axes_angle = [0]
        elif naxes == 2:
            axes_angle = [0, numpy.pi / 2]
        else:
            axes_angle = numpy.linspace(0, 2 * numpy.pi, naxes, endpoint=False)

        axes = numpy.vstack(
            (numpy.cos(axes_angle),
             numpy.sin(axes_angle))
        )
        return axes

    @staticmethod
    def project(axes, X):
        return numpy.dot(axes, X)


def test_main(argv=None):
    import sys
    import sip

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        filename = argv[0]
    else:
        filename = "iris"

    data = Table(filename)

    app = QApplication([])
    w = OWLinearProjection()
    w.set_data(data)
    w.set_subset_data(data[::10])
    w.handleNewSignals()
    w.show()
    w.raise_()
    r = app.exec()
    w.set_data(None)
    w.saveSettings()
    sip.delete(w)
    del w
    return r


if __name__ == "__main__":
    import sys
    sys.exit(test_main())
