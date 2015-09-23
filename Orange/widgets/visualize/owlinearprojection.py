"""
Linear projection widget
------------------------

"""

from functools import reduce
from operator import itemgetter
from types import SimpleNamespace as namespace
from xml.sax.saxutils import escape

import pkg_resources

from PyQt4 import QtGui, QtCore

from PyQt4.QtGui import (
    QListView, QSizePolicy, QApplication, QAction, QKeySequence,
    QGraphicsLineItem, QSlider, QPainterPath
)
from PyQt4.QtCore import Qt, QObject, QEvent, QSize, QRectF, QLineF, QPointF
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import numpy

import pyqtgraph.graphicsItems.ScatterPlotItem
import pyqtgraph as pg
import Orange

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette
from .owscatterplotgraph import LegendItem, legend_anchor_pos
from Orange.widgets.io import FileFormats
from Orange.widgets.utils import classdensity


class DnDVariableListModel(itemmodels.VariableListModel):

    MimeType = "application/x-orange-variable-list"

    def supportedDropActions(self):
        return Qt.MoveAction

    def supportedDragActions(self):
        return Qt.MoveAction

    def mimeTypes(self):
        return [DnDVariableListModel.MimeType]

    def mimeData(self, indexlist):
        variables = []
        itemdata = []
        for index in indexlist:
            variables.append(self[index.row()])
            itemdata.append(self.itemData(index))

        mime = QtCore.QMimeData()
        mime.setData(self.MimeType, b"see properties")
        mime.setProperty("variables", variables)
        mime.setProperty("itemdata", itemdata)
        return mime

    def dropMimeData(self, mime, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True
        elif not mime.hasFormat(self.MimeType):
            return False

        variables = mime.property("variables")
        itemdata = mime.property("itemdata")

        if variables is None:
            return False

        if row == -1:
            row = len(self)

        # Insert variables at row and restore other item data
        self[row:row] = variables
        for i, data in enumerate(itemdata):
            self.setItemData(self.index(row + i), data)
        return True

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid():
            flags |= Qt.ItemIsDragEnabled
        else:
            flags |= Qt.ItemIsDropEnabled
        return flags


class ScatterPlotItem(pg.ScatterPlotItem):
    Symbols = pyqtgraph.graphicsItems.ScatterPlotItem.Symbols

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def paint(self, painter, option, widget=None):
        if self.opts["pxMode"]:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        if self.opts["antialias"]:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        super().paint(painter, option, widget)


class AxisItem(pg.GraphicsObject):
    def __init__(self, parent=None, line=None, label=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setFlag(pg.GraphicsObject.ItemHasNoContents)

        if line is None:
            line = QLineF(0, 0, 1, 0)

        self._spine = QGraphicsLineItem(line, self)
        angle = QLineF(0, 0, 1, 0).angleTo(line)
        angle = (180 - angle) % 360
        dx = line.x2() - line.x1()
        dy = line.y2() - line.y1()
        rad = numpy.arctan2(dy, dx)
        angle = (rad * 180 / numpy.pi) % 360

        self._arrow = pg.ArrowItem(parent=self, angle=180 - angle)
        self._arrow.setPos(self._spine.line().p2())

        self._label = pg.TextItem(text=label, color=(10, 10, 10))
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
            T = Tinv = QtGui.QTransform()

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
        self._label.anchor = pg.Point(*anchor)
        self._label.updateText()
        self._label.setRotation(angle if left_quad else angle - 180)


class LegendItem(LegendItem):
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
    description = "A multi-axes projection of data to a two-dimension plane."
    icon = "icons/LinearProjection.svg"
    priority = 2000

    inputs = [("Data", Orange.data.Table, "set_data", widget.Default),
              ("Data Subset", Orange.data.Table, "set_subset_data")]
#              #TODO: Allow for axes to be supplied from an external source.
#               ("Projection", numpy.ndarray, "set_axes"),]
    outputs = [("Selected Data", Orange.data.Table)]

    settingsHandler = settings.DomainContextHandler()

    variable_state = settings.ContextSetting({})

    color_index = settings.ContextSetting(0)
    shape_index = settings.ContextSetting(0)
    size_index = settings.ContextSetting(0)

    point_size = settings.Setting(10)
    alpha_value = settings.Setting(255)
    jitter_value = settings.Setting(0)

    class_density = settings.Setting(False)
    resolution = 256

    auto_commit = settings.Setting(True)

    legend_anchor = settings.Setting(((1, 0), (1, 0)))
    MinPointSize = 6

    ReplotRequest = QEvent.registerEventType()

    want_graph = True

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.subset_data = None
        self._subset_mask = None
        self._selection_mask = None
        self._item = None
        self._density_img = None
        self.__legend = None
        self.__selection_item = None
        self.__replot_requested = False

        box = gui.widgetBox(self.controlArea, "Axes")

        box1 = gui.widgetBox(box, "Displayed", margin=0)
        box1.setFlat(True)
        self.active_view = view = QListView(
            sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Ignored),
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

        self.varmodel_selected = model = DnDVariableListModel(
            parent=self)

        model.rowsInserted.connect(self._invalidate_plot)
        model.rowsRemoved.connect(self._invalidate_plot)
        model.rowsMoved.connect(self._invalidate_plot)

        view.setModel(model)

        box1.layout().addWidget(view)

        box1 = gui.widgetBox(box, "Other", margin=0)
        box1.setFlat(True)
        self.other_view = view = QListView(
            sizePolicy=QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Ignored),
            selectionMode=QListView.ExtendedSelection,
            dragEnabled=True,
            defaultDropAction=Qt.MoveAction,
            dragDropOverwriteMode=False,
            dragDropMode=QListView.DragDrop,
            showDropIndicator=True,
            minimumHeight=150
        )
        view.viewport().setAcceptDrops(True)
        moveup = QtGui.QAction(
            "Move up", view,
            shortcut=QKeySequence(Qt.AltModifier | Qt.Key_Up),
            triggered=self.__activate_selection
        )
        view.addAction(moveup)

        self.varmodel_other = model = DnDVariableListModel(parent=self)
        view.setModel(model)

        box1.layout().addWidget(view)

        box = gui.widgetBox(self.controlArea, "Plot Properties")
        box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.colorvar_model = itemmodels.VariableListModel(parent=self)
        self.shapevar_model = itemmodels.VariableListModel(parent=self)
        self.sizevar_model = itemmodels.VariableListModel(parent=self)

        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
            spacing=8
        )
        box.layout().addLayout(form)

        cb = gui.comboBox(box, self, "color_index",
                          callback=self._on_color_change,
                          contentsLength=12)
        cb.setModel(self.colorvar_model)
        form.addRow("Colors", cb)

        alpha_slider = QSlider(
            Qt.Horizontal, minimum=10, maximum=255, pageStep=25,
            tickPosition=QSlider.TicksBelow, value=self.alpha_value)
        alpha_slider.valueChanged.connect(self._set_alpha)
        form.addRow("Opacity", alpha_slider)

        cb = gui.comboBox(box, self, "shape_index",
                          callback=self._on_shape_change,
                          contentsLength=12)

        cb.setModel(self.shapevar_model)
        form.addRow("Shape", cb)

        cb = gui.comboBox(box, self, "size_index",
                          callback=self._on_size_change,
                          contentsLength=12)

        cb.setModel(self.sizevar_model)
        form.addRow("Size", cb)

        size_slider = QSlider(
            Qt.Horizontal,  minimum=3, maximum=30, value=self.point_size,
            pageStep=3,
            tickPosition=QSlider.TicksBelow)
        size_slider.valueChanged.connect(self._set_size)
        form.addRow("", size_slider)

        cb = gui.comboBox(box, self, "jitter_value",
                          items=["None", "0.01%", "0.1%", "0.5%", "1%", "2%"],
                          callback=self._invalidate_plot)
        form.addRow("Jittering", cb)

        self.cb_class_density = gui.checkBox(box, self, "class_density", label="",
                                             callback=self._update_density)
        form.addRow("Class density", self.cb_class_density)

        toolbox = gui.widgetBox(self.controlArea, "Zoom/Select")
        toollayout = QtGui.QHBoxLayout()
        toolbox.layout().addLayout(toollayout)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Commit")

        self.controlArea.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Main area plot
        self.view = pg.GraphicsView(background="w")
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.view.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.viewbox = pg.ViewBox(enableMouse=True, enableMenu=False)
        self.viewbox.grabGesture(Qt.PinchGesture)
        self.view.setCentralItem(self.viewbox)

        self.mainArea.layout().addWidget(self.view)

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
            return QtGui.QIcon(path)

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

        group = QtGui.QActionGroup(self, exclusive=True)
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
            self.viewbox.setCursor(QtGui.QCursor(cursor))
            currenttool = tool

        group.triggered[QAction].connect(activated)

        def button(action):
            b = QtGui.QToolButton()
            b.setDefaultAction(action)
            return b

        toollayout.addWidget(button(actions.select))
        toollayout.addWidget(button(actions.zoom))
        toollayout.addWidget(button(actions.pan))

        toollayout.addSpacing(4)
        toollayout.addWidget(button(actions.zoomtofit))
        toollayout.addStretch()
        toolbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.graphButton.clicked.connect(self.save_graph)

    def sizeHint(self):
        return QSize(800, 500)

    def clear(self):
        self.data = None
        self._subset_mask = None
        self._selection_mask = None

        self.varmodel_selected[:] = []
        self.varmodel_other[:] = []

        self.colorvar_model[:] = []
        self.sizevar_model[:] = []
        self.shapevar_model[:] = []

        self.color_index = 0
        self.size_index = 0
        self.shape_index = 0

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

    def set_data(self, data):
        """
        Set the input dataset.
        """
        self.closeContext()
        self.clear()
        self.data = data
        if data is not None:
            self._initialize(data)
            # get the default encoded state, replacing the position with Inf
            state = self._encode_var_state(
                [list(self.varmodel_selected), list(self.varmodel_other)]
            )
            state = {key: (source_ind, numpy.inf) for key, (source_ind, _)
                     in state.items()}

            self.openContext(data.domain)
            selected_keys = [key for key, (sind, _) in self.variable_state.items()
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

            def clip_index(value, maxv):
                return max(0, min(value, maxv))

            self.color_index = clip_index(
                self.color_index, len(self.colorvar_model) - 1)
            self.shape_index = clip_index(
                self.shape_index, len(self.shapevar_model) - 1)
            self.size_index = clip_index(
                self.size_index, len(self.sizevar_model) - 1)

            self._invalidate_plot()

    def set_subset_data(self, subset):
        """
        Set the supplementary input subset dataset.
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

    def _encode_var_state(self, lists):
        return {(type(var), var.name): (source_ind, pos)
                for source_ind, var_list in enumerate(lists)
                for pos, var in enumerate(var_list)
                if isinstance(var, Orange.data.Variable)}

    def _decode_var_state(self, state, lists):
        all_vars = reduce(list.__iadd__, lists, [])

        newlists = [[] for _ in lists]
        for var in all_vars:
            source, pos = state[(type(var), var.name)]
            newlists[source].append((pos, var))
        return [[var for _, var in sorted(newlist, key=itemgetter(0))]
                for newlist in newlists]

    def color_var(self):
        """
        Current selected color variable or None (if not selected).
        """
        if 1 <= self.color_index < len(self.colorvar_model):
            return self.colorvar_model[self.color_index]
        else:
            return None

    def size_var(self):
        """
        Current selected size variable or None (if not selected).
        """
        if 1 <= self.size_index < len(self.sizevar_model):
            return self.sizevar_model[self.size_index]
        else:
            return None

    def shape_var(self):
        """
        Current selected shape variable or None (if not selected).
        """
        if 1 <= self.shape_index < len(self.shapevar_model):
            return self.shapevar_model[self.shape_index]
        else:
            return None

    def _initialize(self, data):
        # Initialize the GUI controls from data's domain.
        all_vars = list(data.domain.variables)
        cont_vars = [var for var in data.domain.variables
                     if var.is_continuous]
        disc_vars = [var for var in data.domain.variables
                     if var.is_discrete]
        shape_vars = [var for var in disc_vars
                      if len(var.values) <= len(ScatterPlotItem.Symbols) - 1]

        self.all_vars = data.domain.variables
        self.varmodel_selected[:] = cont_vars[:3]
        self.varmodel_other[:] = cont_vars[3:]

        self.colorvar_model[:] = ["Same color"] + all_vars
        self.sizevar_model[:] = ["Same size"] + cont_vars
        self.shapevar_model[:] = ["Same shape"] + shape_vars

        if data.domain.has_discrete_class:
            self.color_index = all_vars.index(data.domain.class_var) + 1

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

    def _get_data(self, var):
        """Return the column data for variable `var`."""
        X, _ = self.data.get_column_view(var)
        return X.ravel()

    def _setup_plot(self, reset_view=True):
        self.__replot_requested = False
        self.clear_plot()

        variables = list(self.varmodel_selected)
        if not variables:
            return

        coords = [self._get_data(var) for var in variables]
        coords = numpy.vstack(coords)
        p, N = coords.shape
        assert N == len(self.data), p == len(variables)

        axes = linproj.defaultaxes(len(variables))

        assert axes.shape == (2, p)

        mask = ~numpy.logical_or.reduce(numpy.isnan(coords), axis=0)
        coords = coords[:, mask]

        X, Y = numpy.dot(axes, coords)
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
            shape=shape_data,
            antialias=True,
            data=numpy.arange(len(self.data))[mask]
        )
        self._item._mask = mask

        self.viewbox.addItem(self._item)

        for i, axis in enumerate(axes.T):
            axis_item = AxisItem(line=QLineF(0, 0, axis[0], axis[1]),
                                 label=variables[i].name)
            self.viewbox.addItem(axis_item)

        if reset_view:
            self.viewbox.setRange(QtCore.QRectF(-1.05, -1.05, 2.1, 2.1))
        self._update_legend()

        color_var = self.color_var()
        if self.class_density and color_var is not None and color_var.is_discrete:
            [min_x, max_x], [min_y, max_y] = self.viewbox.viewRange()
            rgb_data = [brush.color().getRgb()[:3] for brush in brush_data]
            self._density_img = classdensity.class_density_image(min_x, max_x, min_y, max_y, self.resolution,
                                                                X, Y, rgb_data)
            self.viewbox.addItem(self._density_img)

    def _color_data(self, mask=None):
        color_var = self.color_var()
        if color_var is not None:
            color_data = self._get_data(color_var)
            if color_var.is_continuous:
                color_data = plotutils.continuous_colors(color_data)
            else:
                palette = colorpalette.ColorPaletteGenerator(
                    len(color_var.values))
                color_data = plotutils.discrete_colors(
                    color_data, len(color_var.values), palette=palette
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
            color = QtGui.QColor(Qt.darkGray)
            pen_data = QtGui.QPen(color, 1.5)
            pen_data.setCosmetic(True)
            color = QtGui.QColor(Qt.lightGray)
            color.setAlpha(self.alpha_value)
            brush_data = QtGui.QBrush(color)

        if self._subset_mask is not None:
            assert self._subset_mask.shape == (len(self.data),)
            if mask is not None:
                subset_mask = self._subset_mask[mask]
            else:
                subset_mask = self._subset_mask

            if isinstance(brush_data, QtGui.QBrush):
                brush_data = numpy.array([brush_data] * subset_mask.size,
                                         dtype=object)

            brush_data[~subset_mask] = QtGui.QBrush(Qt.NoBrush)

        if self._selection_mask is not None:
            assert self._selection_mask.shape == (len(self.data),)
            if mask is not None:
                selection_mask = self._selection_mask[mask]
            else:
                selection_mask = self._selection_mask

            if isinstance(pen_data, QtGui.QPen):
                pen_data = numpy.array([pen_data] * selection_mask.size,
                                       dtype=object)

            pen_data[selection_mask] = pg.mkPen((200, 200, 0, 150), width=4)
        return pen_data, brush_data

    def _on_color_change(self):
        if self.data is None or self._item is None:
            return

        pen, brush = self._color_data()

        if isinstance(pen, QtGui.QPen):
            # Reset the brush for all points
            self._item.data["pen"] = None
            self._item.setPen(pen)
        else:
            self._item.setPen(pen[self._item._mask])

        if isinstance(brush, QtGui.QBrush):
            # Reset the brush for all points
            self._item.data["brush"] = None
            self._item.setBrush(brush)
        else:
            self._item.setBrush(brush[self._item._mask])

        color_var = self.color_var()
        if color_var is not None and color_var.is_discrete:
            self.cb_class_density.setEnabled(True)
            if self.class_density:
                self._setup_plot(reset_view=False)
        else:
            self.clear_density_img()
            self.cb_class_density.setEnabled(False)


        self._update_legend()

    def _shape_data(self, mask):
        shape_var = self.shape_var()
        if shape_var is None:
            shape_data = numpy.array(["o"] * len(self.data))
        else:
            assert shape_var.is_discrete
            max_symbol = len(ScatterPlotItem.Symbols) - 1
            shape = self._get_data(shape_var)
            shape_mask = numpy.isnan(shape)
            shape = shape % (max_symbol - 1)
            shape[shape_mask] = max_symbol

            symbols = numpy.array(list(ScatterPlotItem.Symbols))
            shape_data = symbols[numpy.asarray(shape, dtype=int)]
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
        size_var = self.size_var()
        if size_var is None:
            size_data = numpy.full((len(self.data),), self.point_size)
        else:
            size_data = plotutils.normalized(self._get_data(size_var))
            size_data -= numpy.nanmin(size_data)
            size_mask = numpy.isnan(size_data)
            size_data = \
                size_data * self.point_size + OWLinearProjection.MinPointSize
            size_data[size_mask] = OWLinearProjection.MinPointSize - 2
        if mask is None:
            return size_data
        else:
            return size_data[mask]

    def _on_size_change(self):
        if self.data is None:
            return
        self.set_size(self._size_data(mask=None))

    def _update_density(self):
        self._setup_plot(reset_view=False)

    def _update_legend(self):
        if self.__legend is None:
            self.__legend = legend = LegendItem()
            legend.setParentItem(self.viewbox)
            legend.setZValue(self.viewbox.zValue() + 10)
            legend.anchor(*self.legend_anchor)
        else:
            legend = self.__legend

        legend.clear()

        color_var, shape_var = self.color_var(), self.shape_var()
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

        palette = colorpalette.ColorPaletteGenerator(
            len(color_var.values) if color_var is not None else 0)
        symbols = list(ScatterPlotItem.Symbols)

        if shape_var is color_var:
            items = [(palette[i], symbols[i], name)
                     for i, name in enumerate(color_var.values)]
        else:
            colors = shapes = []
            if color_var is not None:
                colors = [(palette[i], "o", name)
                          for i, name in enumerate(color_var.values)]
            if shape_var is not None:
                shapes = [(QtGui.QColor(Qt.gray),
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
        if self.data is not None and self._selection_mask is not None:
            indices = numpy.flatnonzero(self._selection_mask)
            if len(indices) > 0:
                subset = self.data[indices]

        self.send("Selected Data", subset)

    def save_graph(self):
        from Orange.widgets.data.owsave import OWSave

        save_img = OWSave(parent=self, data=self.viewbox,
                          file_formats=FileFormats.img_writers)
        save_img.exec_()


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
    PyQt4.QtCore.QObject.eventFilter

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
    def __viewdestroyed(self, obj):
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
                item = QtGui.QGraphicsPathItem()
                color = QtGui.QColor(Qt.yellow)
                item.setPen(QtGui.QPen(color, 0))
                color.setAlpha(50)
                item.setBrush(QtGui.QBrush(color))
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
                self.__zoomitem = QtGui.QGraphicsRectItem()
                color = QtGui.QColor(Qt.yellow)
                self.__zoomitem.setPen(QtGui.QPen(color, 0))
                color.setAlpha(50)
                self.__zoomitem.setBrush(QtGui.QBrush(color))
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
        elif gesture.changeFlags() & QtGui.QPinchGesture.ScaleFactorChanged:
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


class plotutils:
    @ staticmethod
    def continuous_colors(data, palette=None):
        if palette is None:
            palette = colorpalette.ContinuousPaletteGenerator(
                QtGui.QColor(220, 220, 220),
                QtGui.QColor(0, 0, 0),
                False
            )
        amin, amax = numpy.nanmin(data), numpy.nanmax(data)
        span = amax - amin
        data = (data - amin) / (span or 1)

        mask = numpy.isnan(data)
        # Unknown values as gray
        # TODO: This should already be a part of palette
        colors = numpy.empty((len(data), 3))
        colors[mask] = (128, 128, 128)
        colors[~mask] = [palette.getRGB(v) for v in data[~mask]]
        return colors

    @staticmethod
    def discrete_colors(data, nvalues, palette=None):
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
        amin, amax = numpy.nanmin(a), numpy.nanmax(a)
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

    data = Orange.data.Table(filename)

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
