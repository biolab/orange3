
import os
import sys
import unicodedata
import itertools
from functools import partial
from itertools import count

import numpy

from PyQt4 import QtCore
from PyQt4 import QtGui

from PyQt4.QtGui import (
    QListView, QAction, QIcon, QSizePolicy, QPen
)

from PyQt4.QtCore import Qt, QObject, QTimer, QSize, QSizeF, QPointF, QRectF
from PyQt4.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

import Orange.data

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import itemmodels, colorpalette


def indices_to_mask(indices, size):
    """
    Convert an array of integer indices into a boolean mask index.
    The elements in indices must be unique.

    :param ndarray[int] indices: Integer indices.
    :param int size: Size of the resulting mask.

    """
    mask = numpy.zeros(size, dtype=bool)
    mask[indices] = True
    return mask


def split_on_condition(array, condition):
    """
    Split an array in two parts based on a boolean mask array `condition`.
    """
    return array[condition], array[~condition]


def stack_on_condition(a, b, condition):
    """
    Inverse of `split_on_condition`.
    """
    axis = 0
    N = condition.size
    shape = list(a.shape)
    shape[axis] = N
    shape = tuple(shape)
    arr = numpy.empty(shape, dtype=a.dtype)
    arr[condition] = a
    arr[~condition] = b
    return arr


# ###########################
# Data manipulation operators
# ###########################

from collections import namedtuple
if sys.version_info < (3, 4):
    # use singledispatch backports from pypi
    from singledispatch import singledispatch
else:
    from functools import singledispatch


# Base commands
Append = namedtuple("Append", ["points"])
Insert = namedtuple("Insert", ["indices", "points"])
Move = namedtuple("Move", ["indices", "delta"])
DeleteIndices = namedtuple("DeleteIndices", ["indices"])

# A composite of two operators
Composite = namedtuple("Composite", ["f", "g"])

# Non-base commands
# These should be `normalized` (expressed) using base commands
AirBrush = namedtuple("AirBrush", ["pos", "radius", "intensity", "rstate"])
Jitter = namedtuple("Jitter", ["pos", "radius", "intensity", "rstate"])
Magnet = namedtuple("Magnet", ["pos", "radius", "density"])
SelectRegion = namedtuple("SelectRegion", ["region"])
DeleteSelection = namedtuple("DeleteSelection", [])
MoveSelection = namedtuple("MoveSelection", ["delta"])


# Transforms functions for base commands
@singledispatch
def transform(command, data):
    """
    Generic transform for base commands

    :param command: An instance of base command
    :param ndarray data: Input data array
    :rval:
        A (transformed_data, command) tuple of the transformed input data
        and a base command expressing the inverse operation.
    """
    raise NotImplementedError


@transform.register(Append)
def append(command, data):
    return (numpy.vstack([data, command.points]),
            DeleteIndices(slice(len(data),
                                len(data) + len(command.points))))


@transform.register(Insert)
def insert(command, data):
    indices = indices_to_mask(command.indices, len(data) + len(command.points))
    return (stack_on_condition(command.points, data, indices),
            DeleteIndices(indices))


@transform.register(DeleteIndices)
def delete(command, data, ):
    if isinstance(command.indices, slice):
        condition = indices_to_mask(command.indices, len(data))
    else:
        indices = numpy.asarray(command.indices)
        if indices.dtype == numpy.bool:
            condition = indices
        else:
            condition = indices_to_mask(indices, len(data))
    data, deleted = split_on_condition(data, ~condition)
    return data, Insert(command.indices, deleted)


@transform.register(Move)
def move(command, data):
    data[command.indices] += command.delta
    return data, Move(command.indices, -command.delta)


@transform.register(Composite)
def compositum(command, data):
    data, ginv = command.g(data)
    data, finv = command.f(data)
    return data, Composite(ginv, finv)


class PaintViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.tool = None

    def mousePressEvent(self, event):
        if self.tool is not None and self.tool.mousePressEvent(event):
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.tool is not None and self.tool.mouseMoveEvent(event):
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.tool is not None and self.tool.mouseReleaseEvent(event):
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseClickEvent(self, event):
        if self.tool is not None and self.tool.mouseClickEvent(event):
            event.accept()
        else:
            super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if self.tool is not None and self.tool.mouseDragEvent(event):
            event.accept()
        else:
            super().mouseDragEvent(event)

    def hoverEnterEvent(self, event):
        if self.tool is not None and self.tool.hoverEnterEvent(event):
            event.accept()
        else:
            super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if self.tool is not None and self.tool.hoverLeaveEvent(event):
            event.accept()
        else:
            super().hoverLeaveEvent(event)


def crosshairs(color, radius=24, circle=False):
    radius = max(radius, 16)
    pixmap = QtGui.QPixmap(radius, radius)
    pixmap.fill(Qt.transparent)
    painter = QtGui.QPainter()
    painter.begin(pixmap)
    painter.setRenderHints(painter.Antialiasing)
    pen = QtGui.QPen(QtGui.QBrush(color), 1)
    pen.setWidthF(1.5)
    painter.setPen(pen)
    if circle:
        painter.drawEllipse(2, 2, radius - 2, radius - 2)
    painter.drawLine(radius / 2, 7, radius / 2, radius / 2 - 7)
    painter.drawLine(7, radius / 2, radius / 2 - 7, radius / 2)
    painter.end()
    return pixmap


class DataTool(QObject):
    """
    A base class for data tools that operate on PaintViewBox.
    """
    #: Tool mouse cursor has changed
    cursorChanged = Signal(QtGui.QCursor)
    #: User started an editing operation.
    editingStarted = Signal()
    #: User ended an editing operation.
    editingFinished = Signal()
    #: Emits a data transformation command
    issueCommand = Signal(object)

    def __init__(self, parent, plot):
        super().__init__(parent)
        self._cursor = Qt.ArrowCursor
        self._plot = plot

    def cursor(self):
        return QtGui.QCursor(self._cursor)

    def setCursor(self, cursor):
        if self._cursor != cursor:
            self._cursor = QtGui.QCursor(cursor)
            self.cursorChanged.emit()

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def mouseClickEvent(self, event):
        return False

    def mouseDragEvent(self, event):
        return False

    def hoverEnterEvent(self, event):
        return False

    def hoverLeaveEvent(self, event):
        return False

    def mapToPlot(self, point):
        """Map a point in ViewBox local coordinates into plot coordinates.
        """
        box = self._plot.getViewBox()
        return box.mapToView(point)

    def activate(self, ):
        """Activate the tool"""
        pass

    def deactivate(self, ):
        """Deactivate a tool"""
        pass


class PutInstanceTool(DataTool):
    """
    Add a single data instance with a mouse click.
    """
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingStarted.emit()
            pos = self.mapToPlot(event.pos())
            self.issueCommand.emit(Append([pos]))
            event.accept()
            self.editingFinished.emit()
            return True
        else:
            return super().mousePressEvent(event)


class PenTool(DataTool):
    """
    Add points on a path specified with a mouse drag.
    """
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingStarted.emit()
            self.__handleEvent(event)
            return True
        else:
            return super().mousePressEvent()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.__handleEvent(event)
            return True
        else:
            return super().mouseMoveEvent()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingFinished.emit()
            return True
        else:
            return super().mouseReleaseEvent()

    def __handleEvent(self, event):
        pos = self.mapToPlot(event.pos())
        self.issueCommand.emit(Append([pos]))
        event.accept()


class AirBrushTool(DataTool):
    """
    Add points with an 'air brush'.
    """
    def __init__(self, parent, plot):
        super().__init__(parent, plot)
        self.__timer = QTimer(self, interval=50)
        self.__timer.timeout.connect(self.__timout)
        self.__count = itertools.count()
        self.__pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingStarted.emit()
            self.__pos = self.mapToPlot(event.pos())
            self.__timer.start()
            return True
        else:
            return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.__pos = self.mapToPlot(event.pos())
            return True
        else:
            return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__timer.stop()
            self.editingFinished.emit()
            return True
        else:
            return super().mouseReleaseEvent(event)

    def __timout(self):
        self.issueCommand.emit(
            AirBrush(self.__pos, None, None, next(self.__count))
        )


def random_state(rstate):
    if isinstance(rstate, numpy.random.RandomState):
        return rstate
    else:
        return numpy.random.RandomState(rstate)


def create_data(x, y, radius, size, rstate):
    random = random_state(rstate)
    x = random.normal(x, radius / 2, size=size)
    y = random.normal(y, radius / 2, size=size)
    return numpy.c_[x, y]


class MagnetTool(DataTool):
    """
    Draw points closer to the mouse position.
    """
    def __init__(self, parent, plot):
        super().__init__(parent, plot)
        self.__timer = QTimer(self, interval=50)
        self.__timer.timeout.connect(self.__timeout)
        self._radius = 20.0
        self._density = 4.0
        self._pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingStarted.emit()
            self._pos = self.mapToPlot(event.pos())
            self.__timer.start()
            return True
        else:
            return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._pos = self.mapToPlot(event.pos())
            return True
        else:
            return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__timer.stop()
            self.editingFinished.emit()
            return True
        else:
            return super().mouseReleaseEvent(event)

    def __timeout(self):
        self.issueCommand.emit(
            Magnet(self._pos, self._radius, self._density)
        )


class JitterTool(DataTool):
    """
    Jitter points around the mouse position.
    """
    def __init__(self, parent, plot):
        super().__init__(parent, plot)
        self.__timer = QTimer(self, interval=50)
        self.__timer.timeout.connect(self._do)
        self._pos = None
        self._radius = 20.0
        self._intensity = 5.0
        self.__count = itertools.count()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingStarted.emit()
            self._pos = self.mapToPlot(event.pos())
            self.__timer.start()
            return True
        else:
            return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._pos = self.mapToPlot(event.pos())
            return True
        else:
            return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__timer.stop()
            self.editingFinished.emit()
            return True
        else:
            return super().mouseReleaseEvent(event)

    def _do(self):
        self.issueCommand.emit(
            Jitter(self._pos, self._radius, self._intensity,
                   next(self.__count))
        )


class _RectROI(pg.ROI):
    def __init__(self, pos, size, **kwargs):
        super().__init__(pos, size, **kwargs)

    def setRect(self, rect):
        self.setPos(rect.topLeft(), finish=False)
        self.setSize(rect.size(), finish=False)

    def rect(self):
        return QRectF(self.pos(), QSizeF(*self.size()))


class SelectTool(DataTool):
    cursor = Qt.ArrowCursor

    def __init__(self, parent, plot):
        super().__init__(parent, plot)
        self._item = None
        self._start_pos = None
        self._selection_rect = None
        self._mouse_dragging = False
        self._delete_action = QAction(
            "Delete", self,
            shortcut=QtGui.QKeySequence.Delete,
            shortcutContext=Qt.WindowShortcut
        )
        self._delete_action.triggered.connect(self.delete)

    def setSelectionRect(self, rect):
        if self._selection_rect != rect:
            self._selection_rect = QRectF(rect)
            self._item.setRect(self._selection_rect)

    def selectionRect(self):
        return self._item.rect()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToPlot(event.pos())
            if self._item.isVisible():
                if self.selectionRect().contains(pos):
                    # Allow the event to propagate to the item.
                    event.setAccepted(False)
                    self._item.setCursor(Qt.ClosedHandCursor)
                    return False

            self._mouse_dragging = True

            self._start_pos = pos
            self._item.setVisible(True)
            self._plot.addItem(self._item)

            self.setSelectionRect(QRectF(pos, pos))
            event.accept()
            return True
        else:
            return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            pos = self.mapToPlot(event.pos())
            self.setSelectionRect(QRectF(self._start_pos, pos).normalized())
            event.accept()
            return True
        else:
            return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToPlot(event.pos())
            self.setSelectionRect(QRectF(self._start_pos, pos).normalized())
            event.accept()
            self.issueCommand.emit(SelectRegion(self.selectionRect()))
            self._item.setCursor(Qt.OpenHandCursor)
            self._mouse_dragging = False
            return True
        else:
            return super().mouseReleaseEvent(event)

    def activate(self):
        if self._item is None:
            self._item = _RectROI((0, 0), (0, 0), pen=(25, 25, 25))
            self._item.setAcceptedMouseButtons(Qt.LeftButton)
            self._item.setVisible(False)
            self._item.setCursor(Qt.OpenHandCursor)
            self._item.sigRegionChanged.connect(self._on_region_changed)
            self._item.sigRegionChangeStarted.connect(
                self._on_region_change_started)
            self._item.sigRegionChangeFinished.connect(
                self._on_region_change_finished)
            self._plot.addItem(self._item)
            self._mouse_dragging = False

        self._plot.addAction(self._delete_action)

    def deactivate(self):
        self._reset()
        self._plot.removeAction(self._delete_action)

    def _reset(self):
        self.setSelectionRect(QRectF())
        self._item.setVisible(False)
        self._mouse_dragging = False

    def delete(self):
        if not self._mouse_dragging and self._item.isVisible():
            self.issueCommand.emit(DeleteSelection())
            self._reset()

    def _on_region_changed(self):
        if not self._mouse_dragging:
            newrect = self._item.rect()
            delta = newrect.topLeft() - self._selection_rect.topLeft()
            self._selection_rect = newrect
            self.issueCommand.emit(MoveSelection(delta))

    def _on_region_change_started(self):
        if not self._mouse_dragging:
            self.editingStarted.emit()

    def _on_region_change_finished(self):
        if not self._mouse_dragging:
            self.editingFinished.emit()


class ZoomTool(DataTool):

    cursor = None

    def __init__(self, parent, plot):
        super().__init__(parent, plot)

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def activate(self):
        pass

    def deactivate(self):
        pass


class SimpleUndoCommand(QtGui.QUndoCommand):
    """
    :param function redo: A function expressing a redo action.
    :param function undo: A function expressing a undo action.
    """
    def __init__(self, redo, undo, parent=None):
        super().__init__(parent)
        self._redo = redo
        self._undo = undo

    def redo(self):
        self._redo()

    def undo(self):
        self._undo()


class UndoCommand(QtGui.QUndoCommand):
    """An QUndoCommand applying a data transformation operation
    """
    def __init__(self, command, model, parent=None, text=None):
        super().__init__(parent,)
        self._command = command
        self._model = model
        self._undo = None
        if text is not None:
            self.setText(text)

    def redo(self):
        self._undo = self._model.execute(self._command)

    def undo(self):
        self._model.execute(self._undo)

    def mergeWith(self, other):
        if self.id() != other.id():
            return False

        composit = Composite(self._command, other._command)
        merged_command = merge_cmd(composit)

        if merged_command is composit:
            return False

        assert other._undo is not None

        composit = Composite(other._undo, self._undo)
        merged_undo = merge_cmd(composit)

        if merged_undo is composit:
            return False

        self._command = merged_command
        self._undo = merged_undo
        return True

    def id(self):
        return 1


def indices_eq(ind1, ind2):
    if isinstance(ind1, tuple) and isinstance(ind2, tuple):
        if len(ind1) != len(ind2):
            return False
        else:
            return all(indices_eq(i1, i2) for i1, i2 in zip(ind1, ind2))
    elif isinstance(ind1, slice) and isinstance(ind2, slice):
        return ind1 == ind2
    elif ind1 is ... and ind2 is ...:
        return True

    ind1, ind1 = numpy.array(ind1), numpy.array(ind2)

    if ind1.shape != ind2.shape or ind1.dtype != ind2.dtype:
        return False
    else:
        return (ind1 == ind2).all()


def merge_cmd(composit):
    f = composit.f
    g = composit.g

    if isinstance(g, Composite):
        g = merge_cmd(g)

    if isinstance(f, Append) and isinstance(g, Append):
        return Append(numpy.vstack((f.points, g.points)))
    elif isinstance(f, Move) and isinstance(g, Move):
        if indices_eq(f.indices, g.indices):
            return Move(f.indices, f.delta + g.delta)
        else:
            # TODO: union of indices, ...
            return composit
#     elif isinstance(f, DeleteIndices) and isinstance(g, DeleteIndices):
#         indices = numpy.array(g.indices)
#         return DeleteIndices(indices)
    else:
        return composit


def apply_attractor(data, point, density, radius):
    delta = data - point
    dist_sq = numpy.sum(delta ** 2, axis=1)
    dist = numpy.sqrt(dist_sq)

    dist[dist < radius] = 0
    dist_sq = dist ** 2
    valid = (dist_sq > 100 * numpy.finfo(dist.dtype).eps)
    assert valid.shape == (dist.shape[0],)

    df = 0.05 * density / dist_sq[valid]

    df_bound = 1 - radius / dist[valid]

    df = numpy.clip(df, 0, df_bound)

    dx = numpy.zeros_like(delta)
    dx[valid] = df.reshape(-1, 1) * delta[valid]
    return dx


def apply_jitter(data, point, density, radius, rstate=None):
    random = random_state(rstate)

    delta = data - point
    dist_sq = numpy.sum(delta ** 2, axis=1)
    dist = numpy.sqrt(dist_sq)
    valid = dist_sq > 100 * numpy.finfo(dist_sq.dtype).eps

    df = 0.05 * density / dist_sq[valid]
    df_bound = 1 - radius / dist[valid]
    df = numpy.clip(df, 0, df_bound)

    dx = numpy.zeros_like(delta)
    jitter = random.normal(0, 0.1, size=(df.size, data.shape[1]))

    dx[valid, :] = df.reshape(-1, 1) * jitter
    return dx


class ColoredListModel(itemmodels.PyListModel):
    def __init__(self, iterable, parent, flags,
                 list_item_role=QtCore.Qt.DisplayRole,
                 supportedDropActions=QtCore.Qt.MoveAction):

        super().__init__(iterable, parent, flags, list_item_role,
                         supportedDropActions)
        self.colors = colorpalette.ColorPaletteGenerator(10)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if self._is_index_valid_for(index, self) and \
                role == QtCore.Qt.DecorationRole:
            return gui.createAttributePixmap("", self.colors[index.row()])
        else:
            return super().data(index, role)


def _i(name, icon_path="icons/paintdata",
       widg_path=os.path.dirname(os.path.abspath(__file__))):
    return os.path.join(widg_path, icon_path, name)


class OWPaintData(widget.OWWidget):
    TOOLS = [
        ("Brush", "Create multiple instances", AirBrushTool, _i("brush.svg")),
        ("Put", "Put individual instances", PutInstanceTool, _i("put.svg")),
        ("Select", "Select and move instances", SelectTool,
            _i("select-transparent_42px.png")),
        ("Jitter", "Jitter instances", JitterTool, _i("jitter.svg")),
        ("Magnet", "Attract multiple instances", MagnetTool, _i("magnet.svg")),
        ("Zoom", "Zoom", ZoomTool, _i("Dlg_zoom2.png"))
    ]

    name = "Paint Data"
    description = "Create data by painting data points in the plane."
    long_description = ""
    icon = "icons/PaintData.svg"
    priority = 10
    keywords = ["data", "paint", "create"]

    outputs = [("Data", Orange.data.Table)]

    autocommit = Setting(False)
    attr1 = Setting("x")
    attr2 = Setting("y")

    brushRadius = Setting(75)
    density = Setting(7)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.current_tool = None
        self._selected_indices = None
        self._scatter_item = None

        self.labels = ["Class-1", "Class-2"]

        self.undo_stack = QtGui.QUndoStack(self)

        self.class_model = ColoredListModel(
            self.labels, self,
            flags=QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled |
                  QtCore.Qt.ItemIsEditable)

        self.class_model.dataChanged.connect(self._class_value_changed)
        self.class_model.rowsInserted.connect(self._class_count_changed)
        self.class_model.rowsRemoved.connect(self._class_count_changed)

        self.tools_cache = {}

        self._init_ui()

        self.data = numpy.zeros((0, 3))
        self.colors = colorpalette.ColorPaletteGenerator(10)

    def _init_ui(self):
        namesBox = gui.widgetBox(self.controlArea, "Names")

        gui.lineEdit(namesBox, self, "attr1", "Variable X ",
                     controlWidth=80, orientation="horizontal",
                     enterPlaceholder=True, callback=self._attr_name_changed)
        gui.lineEdit(namesBox, self, "attr2", "Variable Y ",
                     controlWidth=80, orientation="horizontal",
                     enterPlaceholder=True, callback=self._attr_name_changed)
        gui.separator(namesBox)

        gui.widgetLabel(namesBox, "Class labels")
        self.classValuesView = listView = QListView(
            selectionMode=QListView.SingleSelection,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored,
                                   QSizePolicy.Maximum)
        )
        listView.setModel(self.class_model)
        itemmodels.select_row(listView, 0)
        namesBox.layout().addWidget(listView)

        addClassLabel = QAction(
            "+", self,
            toolTip="Add new class label",
            triggered=self.add_new_class_label
        )

        self.removeClassLabel = QAction(
            unicodedata.lookup("MINUS SIGN"), self,
            toolTip="Remove selected class label",
            triggered=self.remove_selected_class_label
        )

        actionsWidget = itemmodels.ModelActionsWidget(
            [addClassLabel, self.removeClassLabel], self
        )
        actionsWidget.layout().addStretch(10)
        actionsWidget.layout().setSpacing(1)
        namesBox.layout().addWidget(actionsWidget)

        tBox = gui.widgetBox(self.controlArea, "Tools", addSpace=True)
        buttonBox = gui.widgetBox(tBox, orientation="horizontal")
        toolsBox = gui.widgetBox(buttonBox, orientation=QtGui.QGridLayout())

        self.toolActions = QtGui.QActionGroup(self)
        self.toolActions.setExclusive(True)

        for i, (name, tooltip, tool, icon) in enumerate(self.TOOLS):
            action = QAction(
                name, self,
                toolTip=tooltip,
                checkable=True,
                icon=QIcon(icon),
            )
            action.triggered.connect(partial(self.set_current_tool, tool))

            button = QtGui.QToolButton(
                iconSize=QSize(24, 24),
                toolButtonStyle=Qt.ToolButtonTextUnderIcon,
                sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                       QSizePolicy.Fixed)
            )
            button.setDefaultAction(action)

            toolsBox.layout().addWidget(button, i / 3, i % 3)
            self.toolActions.addAction(action)

        for column in range(3):
            toolsBox.layout().setColumnMinimumWidth(column, 10)
            toolsBox.layout().setColumnStretch(column, 1)

        undo = self.undo_stack.createUndoAction(self)
        redo = self.undo_stack.createRedoAction(self)

        undo.setShortcut(QtGui.QKeySequence.Undo)
        redo.setShortcut(QtGui.QKeySequence.Redo)

        self.addActions([undo, redo])

        gui.separator(tBox)
        indBox = gui.indentedBox(tBox, sep=8)
        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow
        )
        indBox.layout().addLayout(form)
        slider = gui.hSlider(
            indBox, self, "brushRadius", minValue=1, maxValue=100,
            createLabel=False
        )
        form.addRow("Radius", slider)

        slider = gui.hSlider(
            indBox, self, "density", None, minValue=1, maxValue=100,
            createLabel=False
        )

        form.addRow("Intensity", slider)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.controlArea, self, "autocommit",
                        "Send", "Send on change")

        # main area GUI
        viewbox = PaintViewBox()
        self.plotview = pg.PlotWidget(background="w", viewBox=viewbox)
        self.plot = self.plotview.getPlotItem()

        axis_color = self.palette().color(QtGui.QPalette.Text)
        axis_pen = QtGui.QPen(axis_color)

        tickfont = QtGui.QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        axis = self.plot.getAxis("bottom")
        axis.setLabel(self.attr1)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)

        axis = self.plot.getAxis("left")
        axis.setLabel(self.attr2)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)

        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0),
                           disableAutoRange=True)

        self.mainArea.layout().addWidget(self.plotview)

        # enable brush tool
        self.toolActions.actions()[0].setChecked(True)
        self.set_current_tool(self.TOOLS[0][2])

    def add_new_class_label(self):

        labels = ("Class-%i" % i for i in count(1))
        labels = filter(lambda label: label not in self.class_model,
                        labels)
        newlabel = next(labels)

        command = SimpleUndoCommand(
            lambda: self.class_model.append(newlabel),
            lambda: self.class_model.__delitem__(-1)
        )
        self.undo_stack.push(command)

    def remove_selected_class_label(self):
        index = self.selected_class_label()

        if index is None:
            return

        label = self.class_model[index]
        mask = self.data[:, 2] == index
        move_mask = self.data[~mask][:, 2] > index

        self.undo_stack.beginMacro("Delete class label")
        self.undo_stack.push(UndoCommand(DeleteIndices(mask), self))
        self.undo_stack.push(UndoCommand(Move((move_mask, 2), -1), self))
        self.undo_stack.push(
            SimpleUndoCommand(lambda: self.class_model.__delitem__(index),
                              lambda: self.class_model.insert(index, label)))
        self.undo_stack.endMacro()

        newindex = min(max(index - 1, 0), len(self.class_model) - 1)
        itemmodels.select_row(self.classValuesView, newindex)

    def _class_count_changed(self):
        self.labels = list(self.class_model)
        self.removeClassLabel.setEnabled(len(self.class_model) > 1)

        if self.selected_class_label() is None:
            itemmodels.select_row(self.classValuesView, 0)

    def _class_value_changed(self, index, _):
        index = index.row()
        newvalue = self.class_model[index]
        oldvalue = self.labels[index]
        if newvalue != oldvalue:
            self.labels[index] = newvalue
#             command = Command(
#                 lambda: self.class_model.__setitem__(index, newvalue),
#                 lambda: self.class_model.__setitem__(index, oldvalue),
#             )
#             self.undo_stack.push(command)

    def selected_class_label(self):
        rows = self.classValuesView.selectedIndexes()
        if rows:
            return rows[0].row()
        else:
            return None

    def set_current_tool(self, tool):
        if self.current_tool is not None:
            self.current_tool.deactivate()
            self.current_tool.editingStarted.disconnect(
                self._on_editing_started)
            self.current_tool.editingFinished.disconnect(
                self._on_editing_finished)
            self.current_tool = None
            self.plot.getViewBox().tool = None

        if tool not in self.tools_cache:
            newtool = tool(self, self.plot)
            newtool.editingFinished.connect(self.invalidate)
            self.tools_cache[tool] = newtool
            newtool.issueCommand.connect(self._add_command)

        self._selected_region = QRectF()
        self.current_tool = tool = self.tools_cache[tool]
        self.plot.getViewBox().tool = tool
        tool.editingStarted.connect(self._on_editing_started)
        tool.editingFinished.connect(self._on_editing_finished)
        tool.activate()

    def _on_editing_started(self):
        self.undo_stack.beginMacro("macro")

    def _on_editing_finished(self):
        self.undo_stack.endMacro()

    def execute(self, command):
        if isinstance(command, (Append, DeleteIndices, Insert, Move)):
            if isinstance(command, (DeleteIndices, Insert)):
                self._selected_indices = None

                if isinstance(self.current_tool, SelectTool):
                    self.current_tool._reset()

            self.data, undo = transform(command, self.data)
            self._replot()
            return undo
        else:
            assert False, "Non normalized command"

    def _add_command(self, cmd):
        name = "Name"

        if isinstance(cmd, Append):
            cls = self.selected_class_label()
            points = numpy.array([[p.x(), p.y(), cls] for p in cmd.points])
            self.undo_stack.push(UndoCommand(Append(points), self, text=name))
        elif isinstance(cmd, Move):
            self.undo_stack.push(UndoCommand(cmd, self, text=name))
        elif isinstance(cmd, SelectRegion):
            indices = [i for i, (x, y) in enumerate(self.data[:, :2])
                       if cmd.region.contains(QPointF(x, y))]
            indices = numpy.array(indices, dtype=int)
            self._selected_indices = indices
        elif isinstance(cmd, DeleteSelection):
            indices = self._selected_indices
            if indices is not None and indices.size:
                self.undo_stack.push(
                    UndoCommand(DeleteIndices(indices), self, text="Delete")
                )
        elif isinstance(cmd, MoveSelection):
            indices = self._selected_indices
            if indices is not None and indices.size:
                self.undo_stack.push(
                    UndoCommand(
                        Move((self._selected_indices, slice(0, 2)),
                             numpy.array([cmd.delta.x(), cmd.delta.y()])),
                        self, text="Move")
                )
        elif isinstance(cmd, DeleteIndices):
            self.undo_stack.push(UndoCommand(cmd, self, text="Delete"))
        elif isinstance(cmd, Insert):
            self.undo_stack.push(UndoCommand(cmd, self))
        elif isinstance(cmd, AirBrush):
            data = create_data(cmd.pos.x(), cmd.pos.y(),
                               self.brushRadius / 1000,
                               1 + self.density / 20, cmd.rstate)
            self._add_command(Append([QPointF(*p) for p in zip(*data.T)]))
        elif isinstance(cmd, Jitter):
            point = numpy.array([cmd.pos.x(), cmd.pos.y()])
            delta = - apply_jitter(self.data[:, :2], point,
                                   self.density / 100.0, 0, cmd.rstate)
            self._add_command(Move((..., slice(0, 2)), delta))
        elif isinstance(cmd, Magnet):
            point = numpy.array([cmd.pos.x(), cmd.pos.y()])
            delta = - apply_attractor(self.data[:, :2], point,
                                      self.density / 100.0, 0)
            self._add_command(Move((..., slice(0, 2)), delta))
        else:
            assert False, "unreachable"

    def _replot(self):
        def pen(color):
            pen = QPen(color, 1)
            pen.setCosmetic(True)
            return pen

        if self._scatter_item is not None:
            self.plot.removeItem(self._scatter_item)
            self._scatter_item = None

        nclasses = len(self.class_model)
        pens = [pen(self.colors[i]) for i in range(nclasses)]

        self._scatter_item = pg.ScatterPlotItem(
            self.data[:, 0], self.data[:, 1],
            symbol="+",
            pen=[pens[int(ci)] for ci in self.data[:, 2]]
        )

        self.plot.addItem(self._scatter_item)

    def _attr_name_changed(self):
        self.plot.getAxis("bottom").setLabel(self.attr1)
        self.plot.getAxis("left").setLabel(self.attr2)
        self.invalidate()

    def invalidate(self):
        self.commit()

    def commit(self):
        X, Y = self.data[:, :2], self.data[:, 2]
        attrs = (Orange.data.ContinuousVariable(self.attr1),
                 Orange.data.ContinuousVariable(self.attr2))
        if len(self.class_model) > 1:
            domain = Orange.data.Domain(
                attrs,
                Orange.data.DiscreteVariable(
                    "Class", values=list(self.class_model))
            )
            data = Orange.data.Table.from_numpy(domain, X, Y)
        else:
            domain = Orange.data.Domain(attrs)
            data = Orange.data.Table.from_numpy(domain, X)

        self.send("Data", data)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(1200, 800))

    def onDeleteWidget(self):
        self.plot.clear()


def test():
    import gc
    import sip
    app = QtGui.QApplication([])
    ow = OWPaintData()
    ow.show()
    ow.raise_()
    rval = app.exec_()
    ow.saveSettings()
    ow.onDeleteWidget()
    sip.delete(ow)
    del ow
    gc.collect()
    app.processEvents()
    return rval


if __name__ == "__main__":
    sys.exit(test())
