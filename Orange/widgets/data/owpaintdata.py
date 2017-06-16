
import os
import sys
import unicodedata
import itertools
from functools import partial
from collections import namedtuple

import numpy as np

from AnyQt.QtGui import (
    QIcon, QPen, QBrush, QPainter, QPixmap, QCursor, QFont, QKeySequence,
    QPalette
)
from AnyQt.QtWidgets import (
    QSizePolicy, QAction, QUndoCommand, QUndoStack, QGridLayout,
    QFormLayout, QToolButton, QActionGroup
)

from AnyQt.QtCore import Qt, QObject, QTimer, QSize, QSizeF, QPointF, QRectF
from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

import Orange.data

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import itemmodels, colorpalette

from Orange.util import scale, namegen
from Orange.widgets.widget import OWWidget, Msg


def indices_to_mask(indices, size):
    """
    Convert an array of integer indices into a boolean mask index.
    The elements in indices must be unique.

    :param ndarray[int] indices: Integer indices.
    :param int size: Size of the resulting mask.

    """
    mask = np.zeros(size, dtype=bool)
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
    arr = np.empty(shape, dtype=a.dtype)
    arr[condition] = a
    arr[~condition] = b
    return arr


# ###########################
# Data manipulation operators
# ###########################

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
    np.clip(command.points[:, :2], 0, 1, out=command.points[:, :2])
    return (np.vstack([data, command.points]),
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
        indices = np.asarray(command.indices)
        if indices.dtype == np.bool:
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

        def handle(event, eventType):
            if self.tool is not None and getattr(self.tool, eventType)(event):
                event.accept()
            else:
                getattr(super(self.__class__, self), eventType)(event)

        for eventType in ('mousePressEvent', 'mouseMoveEvent', 'mouseReleaseEvent',
                          'mouseClickEvent', 'mouseDragEvent',
                          'mouseEnterEvent', 'mouseLeaveEvent'):
            setattr(self, eventType, partial(handle, eventType=eventType))


def crosshairs(color, radius=24, circle=False):
    radius = max(radius, 16)
    pixmap = QPixmap(radius, radius)
    pixmap.fill(Qt.transparent)
    painter = QPainter()
    painter.begin(pixmap)
    painter.setRenderHints(QPainter.Antialiasing)
    pen = QPen(QBrush(color), 1)
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
    cursorChanged = Signal(QCursor)
    #: User started an editing operation.
    editingStarted = Signal()
    #: User ended an editing operation.
    editingFinished = Signal()
    #: Emits a data transformation command
    issueCommand = Signal(object)

    # Makes for a checkable push-button
    checkable = True

    # The tool only works if (at least) two dimensions
    only2d = True

    def __init__(self, parent, plot):
        super().__init__(parent)
        self._cursor = Qt.ArrowCursor
        self._plot = plot

    def cursor(self):
        return QCursor(self._cursor)

    def setCursor(self, cursor):
        if self._cursor != cursor:
            self._cursor = QCursor(cursor)
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
    only2d = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingStarted.emit()
            pos = self.mapToPlot(event.pos())
            self.issueCommand.emit(Append([pos]))
            event.accept()
            self.editingFinished.emit()
            return True
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
        return super().mousePressEvent()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.__handleEvent(event)
            return True
        return super().mouseMoveEvent()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editingFinished.emit()
            return True
        return super().mouseReleaseEvent()

    def __handleEvent(self, event):
        pos = self.mapToPlot(event.pos())
        self.issueCommand.emit(Append([pos]))
        event.accept()


class AirBrushTool(DataTool):
    """
    Add points with an 'air brush'.
    """
    only2d = False

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
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.__pos = self.mapToPlot(event.pos())
            return True
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__timer.stop()
            self.editingFinished.emit()
            return True
        return super().mouseReleaseEvent(event)

    def __timout(self):
        self.issueCommand.emit(
            AirBrush(self.__pos, None, None, next(self.__count))
        )


def random_state(rstate):
    if isinstance(rstate, np.random.RandomState):
        return rstate
    return np.random.RandomState(rstate)


def create_data(x, y, radius, size, rstate):
    random = random_state(rstate)
    x = random.normal(x, radius / 2, size=size)
    y = random.normal(y, radius / 2, size=size)
    return np.c_[x, y]


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
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._pos = self.mapToPlot(event.pos())
            return True
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__timer.stop()
            self.editingFinished.emit()
            return True
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
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._pos = self.mapToPlot(event.pos())
            return True
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__timer.stop()
            self.editingFinished.emit()
            return True
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
            "Delete", self, shortcutContext=Qt.WindowShortcut
        )
        self._delete_action.setShortcuts([QKeySequence.Delete,
                                          QKeySequence("Backspace")])
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
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            pos = self.mapToPlot(event.pos())
            self.setSelectionRect(QRectF(self._start_pos, pos).normalized())
            event.accept()
            return True
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


class ClearTool(DataTool):
    cursor = None
    checkable = False
    only2d = False

    def activate(self):
        self.editingStarted.emit()
        self.issueCommand.emit(SelectRegion(self._plot.rect()))
        self.issueCommand.emit(DeleteSelection())
        self.editingFinished.emit()


class SimpleUndoCommand(QUndoCommand):
    """
    :param function redo: A function expressing a redo action.
    :param function undo: A function expressing a undo action.
    """
    def __init__(self, redo, undo, parent=None):
        super().__init__(parent)
        self.redo = redo
        self.undo = undo


class UndoCommand(QUndoCommand):
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
        return all(indices_eq(i1, i2) for i1, i2 in zip(ind1, ind2))
    elif isinstance(ind1, slice) and isinstance(ind2, slice):
        return ind1 == ind2
    elif ind1 is ... and ind2 is ...:
        return True

    ind1, ind1 = np.array(ind1), np.array(ind2)

    if ind1.shape != ind2.shape or ind1.dtype != ind2.dtype:
        return False
    return (ind1 == ind2).all()


def merge_cmd(composit):
    f = composit.f
    g = composit.g

    if isinstance(g, Composite):
        g = merge_cmd(g)

    if isinstance(f, Append) and isinstance(g, Append):
        return Append(np.vstack((f.points, g.points)))
    elif isinstance(f, Move) and isinstance(g, Move):
        if indices_eq(f.indices, g.indices):
            return Move(f.indices, f.delta + g.delta)
        else:
            # TODO: union of indices, ...
            return composit
#     elif isinstance(f, DeleteIndices) and isinstance(g, DeleteIndices):
#         indices = np.array(g.indices)
#         return DeleteIndices(indices)
    else:
        return composit


def apply_attractor(data, point, density, radius):
    delta = data - point
    dist_sq = np.sum(delta ** 2, axis=1)
    dist = np.sqrt(dist_sq)

    dist[dist < radius] = 0
    dist_sq = dist ** 2
    valid = (dist_sq > 100 * np.finfo(dist.dtype).eps)
    assert valid.shape == (dist.shape[0],)

    df = 0.05 * density / dist_sq[valid]

    df_bound = 1 - radius / dist[valid]

    df = np.clip(df, 0, df_bound)

    dx = np.zeros_like(delta)
    dx[valid] = df.reshape(-1, 1) * delta[valid]
    return dx


def apply_jitter(data, point, density, radius, rstate=None):
    random = random_state(rstate)

    delta = data - point
    dist_sq = np.sum(delta ** 2, axis=1)
    dist = np.sqrt(dist_sq)
    valid = dist_sq > 100 * np.finfo(dist_sq.dtype).eps

    df = 0.05 * density / dist_sq[valid]
    df_bound = 1 - radius / dist[valid]
    df = np.clip(df, 0, df_bound)

    dx = np.zeros_like(delta)
    jitter = random.normal(0, 0.1, size=(df.size, data.shape[1]))

    dx[valid, :] = df.reshape(-1, 1) * jitter
    return dx


class ColoredListModel(itemmodels.PyListModel):
    def __init__(self, iterable, parent, flags,
                 list_item_role=Qt.DisplayRole,
                 supportedDropActions=Qt.MoveAction):

        super().__init__(iterable, parent, flags, list_item_role,
                         supportedDropActions)
        self.colors = colorpalette.ColorPaletteGenerator(
            len(colorpalette.DefaultRGBColors))

    def data(self, index, role=Qt.DisplayRole):
        if self._is_index_valid_for(index, self) and \
                role == Qt.DecorationRole and \
                0 <= index.row() < self.colors.number_of_colors:
            return gui.createAttributePixmap("", self.colors[index.row()])
        return super().data(index, role)


def _icon(name, icon_path="icons/paintdata",
        widg_path=os.path.dirname(os.path.abspath(__file__))):
    return os.path.join(widg_path, icon_path, name)


class OWPaintData(OWWidget):
    TOOLS = [
        ("Brush", "Create multiple instances", AirBrushTool, _icon("brush.svg")),
        ("Put", "Put individual instances", PutInstanceTool, _icon("put.svg")),
        ("Select", "Select and move instances", SelectTool,
         _icon("select-transparent_42px.png")),
        ("Jitter", "Jitter instances", JitterTool, _icon("jitter.svg")),
        ("Magnet", "Attract multiple instances", MagnetTool, _icon("magnet.svg")),
        ("Clear", "Clear the plot", ClearTool, _icon("../../../icons/Dlg_clear.png"))
    ]

    name = "Paint Data"
    description = "Create data by painting data points on a plane."
    icon = "icons/PaintData.svg"
    priority = 15
    keywords = ["data", "paint", "create"]

    outputs = [("Data", Orange.data.Table)]
    inputs = [("Data", Orange.data.Table, "set_data")]

    autocommit = Setting(True)
    table_name = Setting("Painted data")
    attr1 = Setting("x")
    attr2 = Setting("y")
    hasAttr2 = Setting(True)

    brushRadius = Setting(75)
    density = Setting(7)
    #: current data array (shape=(N, 3)) as presented on the output
    data = Setting(None, schema_only=True)
    labels = Setting(["C1", "C2"], schema_only=True)

    graph_name = "plot"

    class Warning(OWWidget.Warning):
        no_input_variables = Msg("Input data has no variables")
        continuous_target = Msg("Continuous target value can not be used.")
        sparse_not_supported = Msg("Sparse data is ignored.")

    class Information(OWWidget.Information):
        use_first_two = \
            Msg("Paint Data uses data from the first two attributes.")

    def __init__(self):
        super().__init__()

        self.input_data = None
        self.input_classes = []
        self.input_colors = None
        self.input_has_attr2 = True
        self.current_tool = None
        self._selected_indices = None
        self._scatter_item = None
        #: A private data buffer (can be modified in place). `self.data` is
        #: a copy of this array (as seen when the `invalidate` method is
        #: called
        self.__buffer = None

        self.undo_stack = QUndoStack(self)

        self.class_model = ColoredListModel(
            self.labels, self,
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled |
            Qt.ItemIsEditable)

        self.class_model.dataChanged.connect(self._class_value_changed)
        self.class_model.rowsInserted.connect(self._class_count_changed)
        self.class_model.rowsRemoved.connect(self._class_count_changed)

        if not self.data:
            self.data = []
            self.__buffer = np.zeros((0, 3))
        elif isinstance(self.data, np.ndarray):
            self.__buffer = self.data.copy()
            self.data = self.data.tolist()
        else:
            self.__buffer = np.array(self.data)

        self.colors = colorpalette.ColorPaletteGenerator(
            len(colorpalette.DefaultRGBColors))
        self.tools_cache = {}

        self._init_ui()
        self.commit()

    def _init_ui(self):
        namesBox = gui.vBox(self.controlArea, "Names")

        hbox = gui.hBox(namesBox, margin=0, spacing=0)
        gui.lineEdit(hbox, self, "attr1", "Variable X: ",
                     controlWidth=80, orientation=Qt.Horizontal,
                     callback=self._attr_name_changed)
        gui.separator(hbox, 21)
        hbox = gui.hBox(namesBox, margin=0, spacing=0)
        attr2 = gui.lineEdit(hbox, self, "attr2", "Variable Y: ",
                             controlWidth=80, orientation=Qt.Horizontal,
                             callback=self._attr_name_changed)
        gui.separator(hbox)
        gui.checkBox(hbox, self, "hasAttr2", '', disables=attr2,
                     labelWidth=0,
                     callback=self.set_dimensions)
        gui.separator(namesBox)

        gui.widgetLabel(namesBox, "Labels")
        self.classValuesView = listView = gui.ListViewWithSizeHint(
            preferred_size=(-1, 30))
        listView.setModel(self.class_model)
        itemmodels.select_row(listView, 0)
        namesBox.layout().addWidget(listView)

        self.addClassLabel = QAction(
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
            [self.addClassLabel, self.removeClassLabel], self
        )
        actionsWidget.layout().addStretch(10)
        actionsWidget.layout().setSpacing(1)
        namesBox.layout().addWidget(actionsWidget)

        tBox = gui.vBox(self.controlArea, "Tools", addSpace=True)
        buttonBox = gui.hBox(tBox)
        toolsBox = gui.widgetBox(buttonBox, orientation=QGridLayout())

        self.toolActions = QActionGroup(self)
        self.toolActions.setExclusive(True)
        self.toolButtons = []

        for i, (name, tooltip, tool, icon) in enumerate(self.TOOLS):
            action = QAction(
                name, self,
                toolTip=tooltip,
                checkable=tool.checkable,
                icon=QIcon(icon),
            )
            action.triggered.connect(partial(self.set_current_tool, tool))

            button = QToolButton(
                iconSize=QSize(24, 24),
                toolButtonStyle=Qt.ToolButtonTextUnderIcon,
                sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                       QSizePolicy.Fixed)
            )
            button.setDefaultAction(action)
            self.toolButtons.append((button, tool))

            toolsBox.layout().addWidget(button, i / 3, i % 3)
            self.toolActions.addAction(action)

        for column in range(3):
            toolsBox.layout().setColumnMinimumWidth(column, 10)
            toolsBox.layout().setColumnStretch(column, 1)

        undo = self.undo_stack.createUndoAction(self)
        redo = self.undo_stack.createRedoAction(self)

        undo.setShortcut(QKeySequence.Undo)
        redo.setShortcut(QKeySequence.Redo)

        self.addActions([undo, redo])
        self.undo_stack.indexChanged.connect(lambda _: self.invalidate())

        gui.separator(tBox)
        indBox = gui.indentedBox(tBox, sep=8)
        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )
        indBox.layout().addLayout(form)
        slider = gui.hSlider(
            indBox, self, "brushRadius", minValue=1, maxValue=100,
            createLabel=False
        )
        form.addRow("Radius:", slider)

        slider = gui.hSlider(
            indBox, self, "density", None, minValue=1, maxValue=100,
            createLabel=False
        )

        form.addRow("Intensity:", slider)
        self.btResetToInput = gui.button(
            tBox, self, "Reset to Input Data", self.reset_to_input)
        self.btResetToInput.setDisabled(True)

        gui.auto_commit(self.left_side, self, "autocommit",
                        "Send")

        # main area GUI
        viewbox = PaintViewBox(enableMouse=False)
        self.plotview = pg.PlotWidget(background="w", viewBox=viewbox)
        self.plotview.sizeHint = lambda: QSize(200, 100)  # Minimum size for 1-d painting
        self.plot = self.plotview.getPlotItem()

        axis_color = self.palette().color(QPalette.Text)
        axis_pen = QPen(axis_color)

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        axis = self.plot.getAxis("bottom")
        axis.setLabel(self.attr1)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)

        axis = self.plot.getAxis("left")
        axis.setLabel(self.attr2)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)
        if not self.hasAttr2:
            self.plot.hideAxis('left')

        self.plot.hideButtons()
        self.plot.setXRange(0, 1, padding=0.01)

        self.mainArea.layout().addWidget(self.plotview)

        # enable brush tool
        self.toolActions.actions()[0].setChecked(True)
        self.set_current_tool(self.TOOLS[0][2])

        self.set_dimensions()

    def set_dimensions(self):
        if self.hasAttr2:
            self.plot.setYRange(0, 1, padding=0.01)
            self.plot.showAxis('left')
            self.plotview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        else:
            self.plot.setYRange(-.5, .5, padding=0.01)
            self.plot.hideAxis('left')
            self.plotview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self._replot()
        for button, tool in self.toolButtons:
            if tool.only2d:
                button.setDisabled(not self.hasAttr2)

    def set_data(self, data):
        """Set the input_data and call reset_to_input"""
        def _check_and_set_data(data):
            self.clear_messages()
            if data and data.is_sparse():
                self.Warning.sparse_not_supported()
                return False
            if data is not None and len(data):
                if not data.domain.attributes:
                    self.Warning.no_input_variables()
                    data = None
                elif len(data.domain.attributes) > 2:
                    self.Information.use_first_two()
            self.input_data = data
            self.btResetToInput.setDisabled(data is None)
            return data is not None and len(data)

        if not _check_and_set_data(data):
            return

        X = np.array([scale(vals) for vals in data.X[:, :2].T]).T
        try:
            y = next(cls for cls in data.domain.class_vars if cls.is_discrete)
        except StopIteration:
            if data.domain.class_vars:
                self.Warning.continuous_target()
            self.input_classes = ["C1"]
            self.input_colors = None
            y = np.zeros(len(data))
        else:
            self.input_classes = y.values
            self.input_colors = y.colors

            y = data[:, y].Y

        self.input_has_attr2 = len(data.domain.attributes) >= 2
        if not self.input_has_attr2:
            self.input_data = np.column_stack((X, np.zeros(len(data)), y))
        else:
            self.input_data = np.column_stack((X, y))
        self.reset_to_input()
        self.unconditional_commit()

    def reset_to_input(self):
        """Reset the painting to input data if present."""
        if self.input_data is None:
            return
        self.undo_stack.clear()

        index = self.selected_class_label()
        if self.input_colors is not None:
            colors = self.input_colors
        else:
            colors = colorpalette.DefaultRGBColors
        palette = colorpalette.ColorPaletteGenerator(
            number_of_colors=len(colors), rgb_colors=colors)
        self.colors = palette
        self.class_model.colors = palette
        self.class_model[:] = self.input_classes

        newindex = min(max(index, 0), len(self.class_model) - 1)
        itemmodels.select_row(self.classValuesView, newindex)

        self.data = self.input_data.tolist()
        self.__buffer = self.input_data.copy()

        prev_attr2 = self.hasAttr2
        self.hasAttr2 = self.input_has_attr2
        if prev_attr2 != self.hasAttr2:
            self.set_dimensions()
        else:  # set_dimensions already calls _replot, no need to call it again
            self._replot()

    def add_new_class_label(self, undoable=True):

        newlabel = next(label for label in namegen('C', 1)
                        if label not in self.class_model)

        command = SimpleUndoCommand(
            lambda: self.class_model.append(newlabel),
            lambda: self.class_model.__delitem__(-1)
        )
        if undoable:
            self.undo_stack.push(command)
        else:
            command.redo()

    def remove_selected_class_label(self):
        index = self.selected_class_label()

        if index is None:
            return

        label = self.class_model[index]
        mask = self.__buffer[:, 2] == index
        move_mask = self.__buffer[~mask][:, 2] > index

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
        self.addClassLabel.setEnabled(
            len(self.class_model) < self.colors.number_of_colors)
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
        return None

    def set_current_tool(self, tool):
        prev_tool = self.current_tool.__class__

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
            self.tools_cache[tool] = newtool
            newtool.issueCommand.connect(self._add_command)

        self._selected_region = QRectF()
        self.current_tool = tool = self.tools_cache[tool]
        self.plot.getViewBox().tool = tool
        tool.editingStarted.connect(self._on_editing_started)
        tool.editingFinished.connect(self._on_editing_finished)
        tool.activate()

        if not tool.checkable:
            self.set_current_tool(prev_tool)

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

            self.__buffer, undo = transform(command, self.__buffer)
            self._replot()
            return undo
        else:
            assert False, "Non normalized command"

    def _add_command(self, cmd):
        name = "Name"

        if (not self.hasAttr2 and
                isinstance(cmd, (Move, MoveSelection, Jitter, Magnet))):
            # tool only supported if both x and y are enabled
            return

        if isinstance(cmd, Append):
            cls = self.selected_class_label()
            points = np.array([(p.x(), p.y() if self.hasAttr2 else 0, cls)
                               for p in cmd.points])
            self.undo_stack.push(UndoCommand(Append(points), self, text=name))
        elif isinstance(cmd, Move):
            self.undo_stack.push(UndoCommand(cmd, self, text=name))
        elif isinstance(cmd, SelectRegion):
            indices = [i for i, (x, y) in enumerate(self.__buffer[:, :2])
                       if cmd.region.contains(QPointF(x, y))]
            indices = np.array(indices, dtype=int)
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
                             np.array([cmd.delta.x(), cmd.delta.y()])),
                        self, text="Move")
                )
        elif isinstance(cmd, DeleteIndices):
            self.undo_stack.push(UndoCommand(cmd, self, text="Delete"))
        elif isinstance(cmd, Insert):
            self.undo_stack.push(UndoCommand(cmd, self))
        elif isinstance(cmd, AirBrush):
            data = create_data(cmd.pos.x(), cmd.pos.y(),
                               self.brushRadius / 1000,
                               int(1 + self.density / 20), cmd.rstate)
            self._add_command(Append([QPointF(*p) for p in zip(*data.T)]))
        elif isinstance(cmd, Jitter):
            point = np.array([cmd.pos.x(), cmd.pos.y()])
            delta = - apply_jitter(self.__buffer[:, :2], point,
                                   self.density / 100.0, 0, cmd.rstate)
            self._add_command(Move((..., slice(0, 2)), delta))
        elif isinstance(cmd, Magnet):
            point = np.array([cmd.pos.x(), cmd.pos.y()])
            delta = - apply_attractor(self.__buffer[:, :2], point,
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

        x = self.__buffer[:, 0].copy()
        if self.hasAttr2:
            y = self.__buffer[:, 1].copy()
        else:
            y = np.zeros(self.__buffer.shape[0])

        colors = self.colors[self.__buffer[:, 2]]
        pens = [pen(c) for c in colors]
        self._scatter_item = pg.ScatterPlotItem(
            x, y, symbol="+", pen=pens
        )
        self.plot.addItem(self._scatter_item)

    def _attr_name_changed(self):
        self.plot.getAxis("bottom").setLabel(self.attr1)
        self.plot.getAxis("left").setLabel(self.attr2)
        self.invalidate()

    def invalidate(self):
        self.data = self.__buffer.tolist()
        self.commit()

    def commit(self):
        data = np.array(self.data)
        if len(data) == 0:
            self.send("Data", None)
            return
        if self.hasAttr2:
            X, Y = data[:, :2], data[:, 2]
            attrs = (Orange.data.ContinuousVariable(self.attr1),
                     Orange.data.ContinuousVariable(self.attr2))
        else:
            X, Y = data[:, np.newaxis, 0], data[:, 2]
            attrs = (Orange.data.ContinuousVariable(self.attr1),)
        if len(np.unique(Y)) >= 2:
            domain = Orange.data.Domain(
                attrs,
                Orange.data.DiscreteVariable(
                    "Class", values=list(self.class_model))
            )
            data = Orange.data.Table.from_numpy(domain, X, Y)
        else:
            domain = Orange.data.Domain(attrs)
            data = Orange.data.Table.from_numpy(domain, X)
        data.name = self.table_name
        self.send("Data", data)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(570, 690))

    def onDeleteWidget(self):
        self.plot.clear()

    def send_report(self):
        if self.data is None:
            return
        settings = []
        if self.attr1 != "x" or self.attr2 != "y":
            settings += [("Axis x", self.attr1), ("Axis y", self.attr2)]
        settings += [("Number of points", len(self.data))]
        self.report_items("Painted data", settings)
        self.report_plot()

def main():
    from AnyQt.QtWidgets import QApplication
    import gc
    import sip
    app = QApplication([])
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
    sys.exit(main())
