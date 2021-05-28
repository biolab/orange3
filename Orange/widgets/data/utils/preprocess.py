# Preprocess widget helpers

import bisect
import contextlib
import warnings

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QSpacerItem, QSizePolicy,
    QStyle, QAction, QApplication,
    QStylePainter, QStyleOptionFrame, QDockWidget,
    QFocusFrame
)

from AnyQt.QtCore import (
    Qt, QObject, QEvent, QSize, QModelIndex, QMimeData
)

from AnyQt.QtGui import (
    QCursor, QIcon, QPainter, QPixmap, QStandardItemModel,
    QDrag, QKeySequence
)

from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


@contextlib.contextmanager
def blocked(qobj):
    state = qobj.signalsBlocked()
    qobj.blockSignals(True)
    try:
        yield qobj
    finally:
        qobj.blockSignals(state)


class BaseEditor(QWidget):
    """
    Base widget for editing preprocessor's parameters.
    """
    #: Emitted when parameters have changed.
    changed = Signal()
    #: Emitted when parameters were edited/changed  as a result of
    #: user interaction.
    edited = Signal()

    def setParameters(self, params):
        """
        Set parameters.

        Parameters
        ----------
        params : dict
            Parameters as a dictionary. It is up to subclasses to
            properly parse the contents.

        """
        raise NotImplementedError

    def parameters(self):
        """Return the parameters as a dictionary.
        """
        raise NotImplementedError

    @staticmethod
    def createinstance(params):
        """
        Create the Preprocessor instance given the stored parameters dict.

        Parameters
        ----------
        params : dict
            Parameters as returned by `parameters`.
        """
        raise NotImplementedError


def list_model_move_row_helper(model, parent, src, dst):
    assert src != dst and src != dst - 1
    data = model.itemData(model.index(src, 0, parent))
    removed = model.removeRow(src, parent)
    if not removed:
        return False

    realdst = dst - 1 if dst > src else dst
    inserted = model.insertRow(realdst, parent)
    if not inserted:
        return False

    dataset = model.setItemData(model.index(realdst, 0, parent), data)

    return removed and inserted and dataset


def list_model_move_rows_helper(model, parent, src, count, dst):
    assert not (src <= dst < src + count + 1)
    rowdata = [model.itemData(model.index(src + i, 0, parent))
               for i in range(count)]
    removed = model.removeRows(src, count, parent)
    if not removed:
        return False

    realdst = dst - count if dst > src else dst
    inserted = model.insertRows(realdst, count, parent)
    if not inserted:
        return False

    setdata = True
    for i, data in enumerate(rowdata):
        didset = model.setItemData(model.index(realdst + i, 0, parent), data)
        setdata = setdata and didset
    return setdata


class StandardItemModel(QStandardItemModel):
    """
    A QStandardItemModel improving support for internal row moves.

    The QStandardItemModel is missing support for explicitly moving
    rows internally. Therefore to move a row it is first removed
    reinserted as an empty row and it's data repopulated.
    This triggers rowsRemoved/rowsInserted and dataChanged signals.
    If an observer is monitoring the model state it would see all the model
    changes. By using moveRow[s] only one `rowsMoved` signal is emitted
    coalescing all the updates.

    .. note:: The semantics follow Qt5's QAbstractItemModel.moveRow[s]

    """

    def moveRow(self, sourceParent, sourceRow, destParent, destRow):
        """
        Move sourceRow from sourceParent to destinationRow under destParent.

        Returns True if the row was successfully moved; otherwise
        returns false.

        .. note:: Only moves within the same parent are currently supported

        """
        if not sourceParent == destParent:
            return False

        if not self.beginMoveRows(sourceParent, sourceRow, sourceRow,
                                  destParent, destRow):
            return False

        # block so rowsRemoved/Inserted and dataChanged signals
        # are not emitted during the move. Instead the rowsMoved
        # signal will be emitted from self.endMoveRows().
        # I am mostly sure this is safe (a possible problem would be if the
        # base class itself would connect to the rowsInserted, ... to monitor
        # ensure internal invariants)
        with blocked(self):
            didmove = list_model_move_row_helper(
                self, sourceParent, sourceRow, destRow)
        self.endMoveRows()

        if not didmove:
            warnings.warn(
                "`moveRow` did not succeed! Data model might be "
                "in an inconsistent state.",
                RuntimeWarning)
        return didmove

    def moveRows(self, sourceParent, sourceRow, count,
                 destParent, destRow):
        """
        Move count rows starting with the given sourceRow under parent
        sourceParent to row destRow under parent destParent.

        Return true if the rows were successfully moved; otherwise
        returns false.

        .. note:: Only moves within the same parent are currently supported

        """
        if not self.beginMoveRows(sourceParent, sourceRow, sourceRow + count,
                                  destParent, destRow):
            return False

        # block so rowsRemoved/Inserted and dataChanged signals
        # are not emitted during the move. Instead the rowsMoved
        # signal will be emitted from self.endMoveRows().
        with blocked(self):
            didmove = list_model_move_rows_helper(
                self, sourceParent, sourceRow, count, destRow)
        self.endMoveRows()

        if not didmove:
            warnings.warn(
                "`moveRows` did not succeed! Data model might be "
                "in an inconsistent state.",
                RuntimeWarning)
        return didmove


#: Qt.ItemRole holding the PreprocessAction instance
DescriptionRole = Qt.UserRole
#: Qt.ItemRole storing the preprocess parameters
ParametersRole = Qt.UserRole + 1


class Controller(QObject):
    """
    Controller for displaying/editing QAbstractItemModel using SequenceFlow.

    It creates/deletes updates the widgets in the view when the model
    changes, as well as interprets drop events (with appropriate mime data)
    onto the view, modifying the model appropriately.

    Parameters
    ----------
    view : SeqeunceFlow
        The view to control (required).
    model : QAbstarctItemModel
        A list model
    parent : QObject
        The controller's parent.
    """
    MimeType = "application/x-qwidget-ref"

    def __init__(self, view, model=None, parent=None):
        super().__init__(parent)
        self._model = None

        self.view = view
        view.installEventFilter(self)
        view.widgetCloseRequested.connect(self._closeRequested)
        view.widgetMoved.connect(self._widgetMoved)

        # gruesome
        self._setDropIndicatorAt = view._SequenceFlow__setDropIndicatorAt
        self._insertIndexAt = view._SequenceFlow__insertIndexAt

        if model is not None:
            self.setModel(model)

    def __connect(self, model):
        model.dataChanged.connect(self._dataChanged)
        model.rowsInserted.connect(self._rowsInserted)
        model.rowsRemoved.connect(self._rowsRemoved)
        model.rowsMoved.connect(self._rowsMoved)

    def __disconnect(self, model):
        model.dataChanged.disconnect(self._dataChanged)
        model.rowsInserted.disconnect(self._rowsInserted)
        model.rowsRemoved.disconnect(self._rowsRemoved)
        model.rowsMoved.disconnect(self._rowsMoved)

    def setModel(self, model):
        """Set the model for the view.

        :type model: QAbstarctItemModel.
        """
        if self._model is model:
            return

        if self._model is not None:
            self.__disconnect(self._model)

        self._clear()
        self._model = model

        if self._model is not None:
            self._initialize(model)
            self.__connect(model)

    def model(self):
        """Return the model.
        """
        return self._model

    def _initialize(self, model):
        for i in range(model.rowCount()):
            index = model.index(i, 0)
            self._insertWidgetFor(i, index)

    def _clear(self):
        self.view.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            return True
        else:
            return False

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            event.accept()
            self._setDropIndicatorAt(event.pos())
            return True
        else:
            return False

    def dragLeaveEvent(self, event):
        return False
        # TODO: Remember if we have seen enter with the proper data
        # (leave event does not have mimeData)
#         if event.mimeData().hasFormat(self.MimeType) and \
#                 event.proposedAction() == Qt.CopyAction:
#             event.accept()
#             self._setDropIndicatorAt(None)
#             return True
#         else:
#             return False

    def dropEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            # Create and insert appropriate widget.
            self._setDropIndicatorAt(None)
            row = self._insertIndexAt(event.pos())
            model = self.model()

            diddrop = model.dropMimeData(
                event.mimeData(), Qt.CopyAction, row, 0, QModelIndex())

            if diddrop:
                event.accept()
            return True
        else:
            return False

    def eventFilter(self, view, event):
        if view is not self.view:
            return False

        if event.type() == QEvent.DragEnter:
            return self.dragEnterEvent(event)
        elif event.type() == QEvent.DragMove:
            return self.dragMoveEvent(event)
        elif event.type() == QEvent.DragLeave:
            return self.dragLeaveEvent(event)
        elif event.type() == QEvent.Drop:
            return self.dropEvent(event)
        else:
            return super().eventFilter(view, event)

    def _dataChanged(self, topleft, bottomright):
        model = self.model()
        widgets = self.view.widgets()

        top, left = topleft.row(), topleft.column()
        bottom, right = bottomright.row(), bottomright.column()
        assert left == 0 and right == 0

        for row in range(top, bottom + 1):
            self.setWidgetData(widgets[row], model.index(row, 0))

    def _rowsInserted(self, parent, start, end):
        model = self.model()
        for row in range(start, end + 1):
            index = model.index(row, 0, parent)
            self._insertWidgetFor(row, index)

    def _rowsRemoved(self, parent, start, end):
        for row in reversed(range(start, end + 1)):
            self._removeWidgetFor(row, None)

    def _rowsMoved(self, srcparetn, srcstart, srcend,
                   dstparent, dststart, dstend):
        raise NotImplementedError

    def _closeRequested(self, row):
        model = self.model()
        assert 0 <= row < model.rowCount()
        model.removeRows(row, 1, QModelIndex())

    def _widgetMoved(self, from_, to):
        # The widget in the view were already swapped, so
        # we must disconnect from the model when moving the rows.
        # It would be better if this class would also filter and
        # handle internal widget moves.
        model = self.model()
        self.__disconnect(model)
        try:
            model.moveRow
        except AttributeError:
            data = model.itemData(model.index(from_, 0))
            removed = model.removeRow(from_, QModelIndex())
            inserted = model.insertRow(to, QModelIndex())
            model.setItemData(model.index(to, 0), data)
            assert removed and inserted
            assert model.rowCount() == len(self.view.widgets())
        else:
            if to > from_:
                to = to + 1
            didmove = model.moveRow(QModelIndex(), from_, QModelIndex(), to)
            assert didmove
        finally:
            self.__connect(model)

    def _insertWidgetFor(self, row, index):
        widget = self.createWidgetFor(index)
        self.view.insertWidget(row, widget, title=index.data(Qt.DisplayRole))
        self.view.setIcon(row, index.data(Qt.DecorationRole))
        self.setWidgetData(widget, index)
        widget.edited.connect(self.__edited)

    def _removeWidgetFor(self, row, index):
        widget = self.view.widgets()[row]
        self.view.removeWidget(widget)
        widget.edited.disconnect(self.__edited)
        widget.deleteLater()

    def createWidgetFor(self, index):
        """
        Create a QWidget instance for the index (:class:`QModelIndex`)
        """
        definition = index.data(DescriptionRole)
        widget = definition.viewclass()
        return widget

    def setWidgetData(self, widget, index):
        """
        Set/update the widget state from the model at index.
        """
        params = index.data(ParametersRole)
        if not isinstance(params, dict):
            params = {}
        widget.setParameters(params)

    def setModelData(self, widget, index):
        """
        Get the data from the widget state and set/update the model at index.
        """
        params = widget.parameters()
        assert isinstance(params, dict)
        self._model.setData(index, params, ParametersRole)

    @Slot()
    def __edited(self,):
        widget = self.sender()
        row = self.view.indexOf(widget)
        index = self.model().index(row, 0)
        self.setModelData(widget, index)


class SequenceFlow(QWidget):
    """
    A re-orderable list of widgets.
    """
    #: Emitted when the user clicks the Close button in the header
    widgetCloseRequested = Signal(int)
    #: Emitted when the user moves/drags a widget to a new location.
    widgetMoved = Signal(int, int)

    class Frame(QDockWidget):
        """
        Widget frame with a handle.
        """
        closeRequested = Signal()

        def __init__(self, parent=None, widget=None, title=None, **kwargs):

            super().__init__(parent, **kwargs)
            self.setFeatures(QDockWidget.DockWidgetClosable)
            self.setAllowedAreas(Qt.NoDockWidgetArea)

            self.__title = ""
            self.__icon = ""
            self.__focusframe = None

            self.__deleteaction = QAction(
                "Remove", self, shortcut=QKeySequence.Delete,
                enabled=False, triggered=self.closeRequested
            )
            self.addAction(self.__deleteaction)

            if widget is not None:
                self.setWidget(widget)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

            if title:
                self.setTitle(title)

            self.setFocusPolicy(Qt.StrongFocus)

        def setTitle(self, title):
            if self.__title != title:
                self.__title = title
                self.setWindowTitle(title)
                self.update()

        def setIcon(self, icon):
            icon = QIcon(icon)
            if self.__icon != icon:
                self.__icon = icon
                self.setWindowIcon(icon)
                self.update()

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QStylePainter(self)
            opt = QStyleOptionFrame()
            opt.initFrom(self)
            painter.drawPrimitive(QStyle.PE_FrameDockWidget, opt)
            painter.end()

        def focusInEvent(self, event):
            event.accept()
            self.__focusframe = QFocusFrame(self)
            self.__focusframe.setWidget(self)
            self.__deleteaction.setEnabled(True)

        def focusOutEvent(self, event):
            event.accept()
            if self.__focusframe is not None:
                self.__focusframe.deleteLater()
                self.__focusframe = None
            self.__deleteaction.setEnabled(False)

        def closeEvent(self, event):
            super().closeEvent(event)
            event.ignore()
            self.closeRequested.emit()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__dropindicator = QSpacerItem(
            16, 16, QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.__dragstart = (None, None, None)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.__flowlayout = QVBoxLayout()
        layout.addLayout(self.__flowlayout)
        layout.addSpacerItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.setLayout(layout)
        self.setAcceptDrops(True)

    def sizeHint(self):
        """Reimplemented."""
        if self.widgets():
            return super().sizeHint()
        else:
            return QSize(250, 350)

    def addWidget(self, widget, title):
        """Add `widget` with `title` to list of widgets (in the last position).

        Parameters
        ----------
        widget : QWidget
            Widget instance.
        title : str
            Widget title.
        """
        index = len(self.widgets())
        self.insertWidget(index, widget, title)

    def insertWidget(self, index, widget, title):
        """Insert `widget` with `title` at `index`.

        Parameters
        ----------
        index : int
            Position at which the widget should be inserted.
        widget : QWidget
            Widget instance.
        title : str
            Widget title.
        """
        # TODO: Check if widget is already inserted.
        frame = SequenceFlow.Frame(widget=widget, title=title)
        frame.closeRequested.connect(self.__closeRequested)

        layout = self.__flowlayout

        frames = [item.widget() for item in self.layout_iter(layout)
                  if item.widget()]

        if 0 < index < len(frames):
            # find the layout index of a widget occupying the current
            # index'th slot.
            insert_index = layout.indexOf(frames[index])
        elif index == 0:
            insert_index = 0
        elif index < 0 or index >= len(frames):
            insert_index = layout.count()
        else:
            assert False

        layout.insertWidget(insert_index, frame)

        frame.installEventFilter(self)

    def removeWidget(self, widget):
        """Remove widget from the list.

        Parameters
        ----------
        widget : QWidget
            Widget instance to remove.
        """
        layout = self.__flowlayout
        frame = self.__widgetFrame(widget)
        if frame is not None:
            frame.setWidget(None)
            widget.setVisible(False)
            widget.setParent(None)
            layout.takeAt(layout.indexOf(frame))
            frame.hide()
            frame.deleteLater()

    def clear(self):
        """Clear the list (remove all widgets).
        """
        for w in reversed(self.widgets()):
            self.removeWidget(w)

    def widgets(self):
        """Return a list of all `widgets`.
        """
        layout = self.__flowlayout
        items = (layout.itemAt(i) for i in range(layout.count()))
        return [item.widget().widget()
                for item in items if item.widget() is not None]

    def indexOf(self, widget):
        """Return the index (logical position) of `widget`
        """
        widgets = self.widgets()
        return widgets.index(widget)

    def setTitle(self, index, title):
        """Set title for `widget` at `index`.
        """
        widget = self.widgets()[index]
        frame = self.__widgetFrame(widget)
        frame.setTitle(title)

    def setIcon(self, index, icon):
        widget = self.widgets()[index]
        frame = self.__widgetFrame(widget)
        frame.setIcon(icon)

    def dropEvent(self, event):
        """Reimplemented."""
        layout = self.__flowlayout
        index = self.__insertIndexAt(self.mapFromGlobal(QCursor.pos()))

        if event.mimeData().hasFormat("application/x-internal-move") and \
                event.source() is self:
            # Complete the internal move
            frame, oldindex, _ = self.__dragstart
            # Remove the drop indicator spacer item before re-inserting
            # the frame
            self.__setDropIndicatorAt(None)

            if index > oldindex:
                index = index - 1

            if index != oldindex:
                item = layout.takeAt(oldindex)
                assert item.widget() is frame
                layout.insertWidget(index, frame)
                self.widgetMoved.emit(oldindex, index)
                event.accept()

            self.__dragstart = None, None, None

    def dragEnterEvent(self, event):
        """Reimplemented."""
        if event.mimeData().hasFormat("application/x-internal-move") and \
                event.source() is self:
            assert self.__dragstart[0] is not None
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Reimplemented."""
        pos = self.mapFromGlobal(QCursor.pos())
        self.__setDropIndicatorAt(pos)

    def dragLeaveEvent(self, event):
        """Reimplemented."""
        self.__setDropIndicatorAt(None)

    def eventFilter(self, obj, event):
        """Reimplemented."""
        if isinstance(obj, SequenceFlow.Frame) and obj.parent() is self:
            etype = event.type()
            if etype == QEvent.MouseButtonPress and \
                    event.button() == Qt.LeftButton:
                # Is the mouse press on the dock title bar
                # (assume everything above obj.widget is a title bar)
                # TODO: Get the proper title bar geometry.
                if event.pos().y() < obj.widget().y():
                    index = self.indexOf(obj.widget())
                    self.__dragstart = (obj, index, event.pos())
            elif etype == QEvent.MouseMove and \
                    event.buttons() & Qt.LeftButton and \
                    obj is self.__dragstart[0]:
                _, _, down = self.__dragstart
                if (down - event.pos()).manhattanLength() >= \
                        QApplication.startDragDistance():
                    self.__startInternalDrag(obj, event.pos())
                    self.__dragstart = None, None, None
                    return True
            elif etype == QEvent.MouseButtonRelease and \
                    event.button() == Qt.LeftButton and \
                    self.__dragstart[0] is obj:
                self.__dragstart = None, None, None

        return super().eventFilter(obj, event)

    def __setDropIndicatorAt(self, pos):
        # find the index where drop at pos would insert.
        index = -1
        layout = self.__flowlayout
        if pos is not None:
            index = self.__insertIndexAt(pos)
        spacer = self.__dropindicator
        currentindex = self.layout_index_of(layout, spacer)

        if currentindex != -1:
            item = layout.takeAt(currentindex)
            assert item is spacer
            if currentindex < index:
                index -= 1

        if index != -1:
            layout.insertItem(index, spacer)

    def __insertIndexAt(self, pos):
        y = pos.y()
        midpoints = [item.widget().geometry().center().y()
                     for item in self.layout_iter(self.__flowlayout)
                     if item.widget() is not None]
        index = bisect.bisect_left(midpoints, y)
        return index

    def __startInternalDrag(self, frame, hotSpot=None):
        drag = QDrag(self)
        pixmap = QPixmap(frame.size())
        frame.render(pixmap)

        transparent = QPixmap(pixmap.size())
        transparent.fill(Qt.transparent)
        painter = QPainter(transparent)
        painter.setOpacity(0.35)
        painter.drawPixmap(0, 0, pixmap.width(), pixmap.height(), pixmap)
        painter.end()

        drag.setPixmap(transparent)
        if hotSpot is not None:
            drag.setHotSpot(hotSpot)
        mime = QMimeData()
        mime.setData("application/x-internal-move", b"")
        drag.setMimeData(mime)
        return drag.exec(Qt.MoveAction)

    def __widgetFrame(self, widget):
        layout = self.__flowlayout
        for item in self.layout_iter(layout):
            if item.widget() is not None and \
                    isinstance(item.widget(), SequenceFlow.Frame) and \
                    item.widget().widget() is widget:
                return item.widget()
        else:
            return None

    def __closeRequested(self):
        frame = self.sender()
        index = self.indexOf(frame.widget())
        self.widgetCloseRequested.emit(index)

    @staticmethod
    def layout_iter(layout):
        return (layout.itemAt(i) for i in range(layout.count()))

    @staticmethod
    def layout_index_of(layout, item):
        for i, item1 in enumerate(SequenceFlow.layout_iter(layout)):
            if item == item1:
                return i
        return -1
