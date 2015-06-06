"""
Linear projection widget
------------------------

"""

from functools import reduce
from operator import itemgetter

from PyQt4 import QtGui, QtCore

from PyQt4.QtGui import (
    QListView, QSizePolicy, QApplication, QAction, QKeySequence,
    QGraphicsLineItem, QSlider, QPainterPath
)
from PyQt4.QtCore import Qt, QObject, QEvent, QSize, QRectF, QLineF
from PyQt4.QtCore import pyqtSignal as Signal

import numpy

import pyqtgraph.graphicsItems.ScatterPlotItem
import pyqtgraph as pg
import Orange

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette


is_continuous = lambda var: isinstance(var, Orange.data.ContinuousVariable)
is_discrete = lambda var: isinstance(var, Orange.data.DiscreteVariable)
is_string = lambda var: isinstance(var, Orange.data.StringVariable)


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
        label_pos = viewbox_line.pointAt(0.95)

        if left_quad:
            anchor = (0, -0.1)
        else:
            anchor = (1, 1.1)

        pos = T.map(label_pos)
        self._label.setPos(pos)
        self._label.anchor = pg.Point(*anchor)
        self._label.updateText()
        self._label.setRotation(angle if left_quad else angle - 180)


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

    selected_variables = settings.ContextSetting(
        [], required=settings.ContextSetting.REQUIRED
    )
    variable_state = settings.ContextSetting({})

    color_index = settings.ContextSetting(0)
    shape_index = settings.ContextSetting(0)
    size_index = settings.ContextSetting(0)
    label_index = settings.ContextSetting(0)

    point_size = settings.Setting(10)
    alpha_value = settings.Setting(255)
    jitter_value = settings.Setting(0)

    auto_commit = settings.Setting(True)

    MinPointSize = 6

    ReplotRequest = QEvent.registerEventType()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.subset_data = None
        self._subset_mask = None
        self._selection_mask = None
        self._item = None
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
            minimumHeight=50,
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
            minimumHeight=50
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

        box = gui.widgetBox(self.controlArea, "Jittering")
        gui.comboBox(box, self, "jitter_value",
                     items=["None", "0.01%", "0.1%", "0.5%", "1%", "2%"],
                     callback=self._invalidate_plot)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        box = gui.widgetBox(self.controlArea, "Points")
        box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.colorvar_model = itemmodels.VariableListModel(parent=self)
        self.shapevar_model = itemmodels.VariableListModel(parent=self)
        self.sizevar_model = itemmodels.VariableListModel(parent=self)
        self.labelvar_model = itemmodels.VariableListModel(parent=self)

        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow,
            spacing=8
        )
        box.layout().addLayout(form)

        cb = gui.comboBox(box, self, "color_index",
                          callback=self._on_color_change)
        cb.setModel(self.colorvar_model)

        form.addRow("Colors", cb)
        alpha_slider = QSlider(
            Qt.Horizontal, minimum=10, maximum=255, pageStep=25,
            tickPosition=QSlider.TicksBelow, value=self.alpha_value)
        alpha_slider.valueChanged.connect(self._set_alpha)

        form.addRow("Opacity", alpha_slider)

        cb = gui.comboBox(box, self, "shape_index",
                          callback=self._on_shape_change)
        cb.setModel(self.shapevar_model)

        form.addRow("Shape", cb)

        cb = gui.comboBox(box, self, "size_index",
                          callback=self._on_size_change)
        cb.setModel(self.sizevar_model)

        form.addRow("Size", cb)
        size_slider = QSlider(
            Qt.Horizontal,  minimum=3, maximum=30, value=self.point_size,
            pageStep=3,
            tickPosition=QSlider.TicksBelow)

        size_slider.valueChanged.connect(self._set_size)
        form.addRow("", size_slider)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Commit")

        # Main area plot
        self.view = pg.GraphicsView(background="w")
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.view.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.viewbox = pg.ViewBox(enableMouse=True, enableMenu=False)
        self.view.setCentralItem(self.viewbox)

        self.mainArea.layout().addWidget(self.view)
        self.selection = PlotSelectionTool(
            self.viewbox, selectionMode=PlotSelectionTool.Lasso)

        self.selection.setViewBox(self.viewbox)

        self.selection.selectionStarted.connect(self._selection_start)
        self.selection.selectionUpdated.connect(self._selection_update)
        self.selection.selectionFinished.connect(self._selection_finish)

        self.continuous_palette = colorpalette.ContinuousPaletteGenerator(
            QtGui.QColor(220, 220, 220),
            QtGui.QColor(0, 0, 0),
            False
        )
        self.discrete_palette = colorpalette.ColorPaletteGenerator(13)

        zoomtofit = QAction(
            "Zoom to fit", self,
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
            triggered=lambda:
                self.viewbox.setRange(QtCore.QRectF(-1, -1, 2, 2))
        )
        zoomin = QAction(
            "Zoom in", self,
            shortcut=QKeySequence(QKeySequence.ZoomIn),
            triggered=lambda: self.viewbox.scaleBy((1 / 1.25, 1 / 1.25))
        )
        zoomout = QAction(
            "Zoom out", self,
            shortcut=QKeySequence(QKeySequence.ZoomOut),
            triggered=lambda: self.viewbox.scaleBy((1.25, 1.25))
        )
        self.addActions([zoomtofit, zoomin, zoomout])

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
        self.labelvar_model[:] = []

        self.clear_plot()

    def clear_plot(self):
        if self._item is not None:
            self._item.setParentItem(None)
            self.viewbox.removeItem(self._item)
            self._item = None

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
                     if is_continuous(var)]
        disc_vars = [var for var in data.domain.variables
                     if is_discrete(var)]
        string_vars = [var for var in data.domain.variables
                       if is_string(var)]

        self.all_vars = data.domain.variables
        self.varmodel_selected[:] = cont_vars[:3]
        self.varmodel_other[:] = cont_vars[3:]

        self.colorvar_model[:] = ["Same color"] + all_vars
        self.sizevar_model[:] = ["Same size"] + cont_vars
        self.shapevar_model[:] = ["Same shape"] + disc_vars
        self.labelvar_model[:] = ["No label"] + string_vars

        if is_discrete(data.domain.class_var):
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

    def _setup_plot(self):
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

        self.viewbox.setRange(QtCore.QRectF(-1, -1, 2, 2))

    def _color_data(self, mask=None):
        color_var = self.color_var()
        if color_var is not None:
            color_data = self._get_data(color_var)
            if is_continuous(color_var):
                color_data = plotutils.continuous_colors(color_data)
            else:
                color_data = plotutils.discrete_colors(
                    color_data, len(color_var.values)
                )
            if mask is not None:
                color_data = color_data[mask]

            pen_data = numpy.array(
                [pg.mkPen((r, g, b, self.alpha_value / 2))
                 for r, g, b in color_data],
                dtype=object)

            brush_data = numpy.array(
                [pg.mkBrush((r, g, b, self.alpha_value))
                 for r, g, b in color_data],
                dtype=object)
        else:
            color = QtGui.QColor(Qt.lightGray)
            color.setAlpha(self.alpha_value)
            pen_data = QtGui.QPen(color)
            pen_data.setCosmetic(True)
            color = QtGui.QColor(Qt.darkGray)
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

    def _shape_data(self, mask):
        shape_var = self.shape_var()
        if shape_var is None:
            shape_data = numpy.array(["o"] * len(self.data))
        else:
            assert is_discrete(shape_var)
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

    def _selection_start(self):
        item = QtGui.QGraphicsPathItem()
        color = QtGui.QColor(Qt.yellow)
        item.setPen(QtGui.QPen(color, 0))
        color.setAlpha(50)
        item.setBrush(QtGui.QBrush(color))
        self.__selection_item = item

        self._selection_update()

    def _selection_update(self):
        if self.__selection_item is None:
            return
        item = self.__selection_item

        T, ok = self.viewbox.childTransform().inverted()

        if ok:
            path = self.selection.selectionShape()
            path = T.map(path)
            item.setPath(path)
            if item.parentItem() is None:
                self.viewbox.addItem(item)

    def _selection_finish(self):
        self._selection_update()
        path = self.__selection_item.path()
        self.__selection_item.setParentItem(None)
        self.viewbox.removeItem(self.__selection_item)

        self.select(path)

    def select(self, selectionshape):
        item = self._item
        if item is None:
            return

        indices = [spot.data()
                   for spot in item.points()
                   if selectionshape.contains(spot.pos())]

        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            self.select_indices(indices)
        else:
            self._selection_mask = None
            self.select_indices(indices)

    def select_indices(self, indices):
        if self.data is None:
            return

        if self._selection_mask is None:
            self._selection_mask = numpy.zeros(len(self.data), dtype=bool)

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


class PlotSelectionTool(QObject):
    #: Selection modes
    Rect, Lasso = 1, 2

    selectionStarted = Signal()
    selectionUpdated = Signal()
    selectionFinished = Signal()

    def __init__(self, parent=None, selectionMode=Rect, **kwargs):
        super().__init__(parent, **kwargs)
        self.__viewbox = None
        self.__mode = selectionMode
        self._selection = None

    def setSelectionMode(self, mode):
        assert mode in {PlotSelectionTool.Rect, PlotSelectionTool.Lasso}
        if self.__mode != mode:
            if self._selection is not None:
                self.selectionFinished.emit()
            self.__mode = mode
            self._selection = None

    def selectionMode(self):
        return self.__mode

    def setViewBox(self, viewbox):
        if self.__viewbox is viewbox:
            return
        if self.__viewbox is not None:
            self.__viewbox.removeEventFilter(self)

        self.__viewbox = viewbox

        if self.__viewbox is not None:
            self.__viewbox.installEventFilter(self)

    def viewBox(self):
        return self.__viewbox

    def selectionShape(self):
        if self._selection is not None:
            shape = QPainterPath(self._selection)
            shape.closeSubpath()
        else:
            shape = QPainterPath()
        return shape

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(pos, pos)
                self._selection = QPainterPath()
                self._selection.addRect(rect)
            else:
                self._selection = QPainterPath()
                self._selection.moveTo(event.pos())
            self.selectionStarted.emit()
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self._selection = QPainterPath()
                self._selection.addRect(rect)
            else:
                self._selection.lineTo(event.pos())
            self.selectionUpdated.emit()
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self._selection = QPainterPath()
                self._selection.addRect(rect)
            else:
                self._selection.lineTo(event.pos())

            self.selectionFinished.emit()
            event.accept()
            return True
        else:
            return False

    def eventFilter(self, obj, event):
        if event.type() == QEvent.GraphicsSceneMousePress:
            return self.mousePressEvent(event)
        elif event.type() == QEvent.GraphicsSceneMouseMove:
            return self.mouseMoveEvent(event)
        elif event.type() == QEvent.GraphicsSceneMouseRelease:
            return self.mouseReleaseEvent(event)
        else:
            return super().eventFilter(obj, event)


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
        if palette is None:
            palette = colorpalette.ColorPaletteGenerator(nvalues)

        color_index = palette.getRGB(numpy.arange(nvalues + 1))
        # Unknown values as gray
        # TODO: This should already be a part of palette
        color_index[nvalues] = (128, 128, 128)

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
