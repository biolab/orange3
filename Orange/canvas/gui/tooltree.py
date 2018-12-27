"""
=========
Tool Tree
=========

A ToolTree widget presenting the user with a set of actions
organized in a tree structure.

"""

import logging

from AnyQt.QtWidgets import (
    QTreeView, QWidget, QVBoxLayout, QSizePolicy, QStyledItemDelegate,
    QStyle, QAction
)
from AnyQt.QtGui import QStandardItemModel
from AnyQt.QtCore import Qt, QEvent, QModelIndex, QAbstractProxyModel
from AnyQt.QtCore import pyqtSignal as Signal

log = logging.getLogger(__name__)


class ToolTree(QWidget):
    """
    A ListView like presentation of a list of actions.
    """
    triggered = Signal(QAction)
    hovered = Signal(QAction)

    def __init__(self, parent=None, **kwargs):
        QTreeView.__init__(self, parent, **kwargs)
        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.Expanding)

        self.__model = QStandardItemModel()
        self.__flattened = False
        self.__actionRole = Qt.UserRole
        self.__view = None

        self.__setupUi()

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        view = QTreeView(objectName="tool-tree-view")
        view.setUniformRowHeights(True)
        view.setFrameStyle(QTreeView.NoFrame)
        view.setModel(self.__model)
        view.setRootIsDecorated(False)
        view.setHeaderHidden(True)
        view.setItemsExpandable(True)
        view.setEditTriggers(QTreeView.NoEditTriggers)
        view.setItemDelegate(ToolTreeItemDelegate(self))

        view.activated.connect(self.__onActivated)
        view.clicked.connect(self.__onActivated)
        view.entered.connect(self.__onEntered)

        view.installEventFilter(self)

        self.__view = view

        layout.addWidget(view)

        self.setLayout(layout)

    def setFlattened(self, flatten):
        """
        Show the actions in a flattened view.
        """
        if self.__flattened != flatten:
            self.__flattened = flatten
            if flatten:
                model = FlattenedTreeItemModel()
                model.setSourceModel(self.__model)
            else:
                model = self.__model

            self.__view.setModel(model)

    def flattened(self):
        """
        Are actions shown in a flattened tree (a list).
        """
        return self.__flattened

    def setModel(self, model):
        if self.__model is not model:
            self.__model = model

            if self.__flattened:
                model = FlattenedTreeItemModel()
                model.setSourceModel(self.__model)

            self.__view.setModel(model)

    def model(self):
        return self.__model

    def setRootIndex(self, index):
        """Set the root index
        """
        self.__view.setRootIndex(index)

    def rootIndex(self):
        """Return the root index.
        """
        return self.__view.rootIndex()

    def view(self):
        """Return the QTreeView instance used.
        """
        return self.__view

    def setActionRole(self, role):
        """Set the action role. By default this is UserRole
        """
        self.__actionRole = role

    def actionRole(self):
        return self.__actionRole

    def __actionForIndex(self, index):
        val = index.data(self.__actionRole)
        if val is not None:
            action = val
            if isinstance(action, QAction):
                return action
            else:
                log.debug("index does not have an QAction")
        else:
            log.debug("index does not have a value for action role")

    def __onActivated(self, index):
        """The item was activated, if index has an action we
        need to trigger it.

        """
        if index.isValid():
            action = self.__actionForIndex(index)
            if action is not None:
                action.trigger()
                self.triggered.emit(action)

    def __onEntered(self, index):
        if index.isValid():
            action = self.__actionForIndex(index)
            if action is not None:
                action.hover()
                self.hovered.emit(action)

    def ensureCurrent(self):
        """Ensure the view has a current item if one is available.
        """
        model = self.__view.model()
        curr = self.__view.currentIndex()
        if not curr.isValid():
            for i in range(model.rowCount()):
                index = model.index(i, 0)
                if index.flags() & Qt.ItemIsEnabled:
                    self.__view.setCurrentIndex(index)
                    break

    def eventFilter(self, obj, event):
        if obj is self.__view and event.type() == QEvent.KeyPress:
            key = event.key()

            space_activates = \
                self.style().styleHint(
                    QStyle.SH_Menu_SpaceActivatesItem,
                    None, None)

            if key in [Qt.Key_Enter, Qt.Key_Return, Qt.Key_Select] or \
                    (key == Qt.Key_Space and space_activates):
                index = self.__view.currentIndex()
                if index.isValid() and index.flags() & Qt.ItemIsEnabled:
                    # Emit activated on behalf of QTreeView.
                    self.__view.activated.emit(index)
                return True

        return QWidget.eventFilter(self, obj, event)


class ToolTreeItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        QStyledItemDelegate.paint(self, painter, option, index)


class FlattenedTreeItemModel(QAbstractProxyModel):
    """An Proxy Item model containing a flattened view of a column in a tree
    like item model.

    """
    Default = 1
    InternalNodesDisabled = 2
    LeavesOnly = 4

    def __init__(self, parent=None):
        QAbstractProxyModel.__init__(self, parent)
        self.__sourceColumn = 0
        self.__flatteningMode = 1
        self.__sourceRootIndex = QModelIndex()

    def setSourceModel(self, model):
        self.beginResetModel()

        curr_model = self.sourceModel()

        if curr_model is not None:
            curr_model.dataChanged.disconnect(self._sourceDataChanged)
            curr_model.rowsInserted.disconnect(self._sourceRowsInserted)
            curr_model.rowsRemoved.disconnect(self._sourceRowsRemoved)
            curr_model.rowsMoved.disconnect(self._sourceRowsMoved)

        QAbstractProxyModel.setSourceModel(self, model)
        self._updateRowMapping()

        model.dataChanged.connect(self._sourceDataChanged)
        model.rowsInserted.connect(self._sourceRowsInserted)
        model.rowsRemoved.connect(self._sourceRowsRemoved)
        model.rowsMoved.connect(self._sourceRowsMoved)

        self.endResetModel()

    def setSourceColumn(self, column):
        raise NotImplementedError

        self.beginResetModel()
        self.__sourceColumn = column
        self._updateRowMapping()
        self.endResetModel()

    def sourceColumn(self):
        return self.__sourceColumn

    def setSourceRootIndex(self, rootIndex):
        """Set the source root index.
        """
        self.beginResetModel()
        self.__sourceRootIndex = rootIndex
        self._updateRowMapping()
        self.endResetModel()

    def sourceRootIndex(self):
        """Return the source root index.
        """
        return self.__sourceRootIndex

    def setFlatteningMode(self, mode):
        """Set the flattening mode.
        """
        if mode != self.__flatteningMode:
            self.beginResetModel()
            self.__flatteningMode = mode
            self._updateRowMapping()
            self.endResetModel()

    def flatteningMode(self):
        """Return the flattening mode.
        """
        return self.__flatteningMode

    def mapFromSource(self, sourceIndex):
        if sourceIndex.isValid():
            key = self._indexKey(sourceIndex)
            offset = self._source_offset[key]
            row = offset + sourceIndex.row()
            return self.index(row, 0)
        else:
            return sourceIndex

    def mapToSource(self, index):
        if index.isValid():
            row = index.row()
            source_key_path = self._source_key[row]
            return self._indexFromKey(source_key_path)
        else:
            return index

    def index(self, row, column=0, parent=QModelIndex()):
        if not parent.isValid():
            return self.createIndex(row, column, object=row)
        else:
            return QModelIndex()

    def parent(self, child):
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return len(self._source_key)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return 1

    def flags(self, index):
        flags = QAbstractProxyModel.flags(self, index)
        if self.__flatteningMode == self.InternalNodesDisabled:
            sourceIndex = self.mapToSource(index)
            sourceModel = self.sourceModel()
            if sourceModel.rowCount(sourceIndex) > 0 and \
                    flags & Qt.ItemIsEnabled:
                # Internal node, enabled in the source model, disable it
                flags ^= Qt.ItemIsEnabled
        return flags

    def _indexKey(self, index):
        """Return a key for `index` from the source model into
        the _source_offset map. The key is a tuple of row indices on
        the path from the top if the model to the `index`.

        """
        key_path = []
        parent = index
        while parent.isValid():
            key_path.append(parent.row())
            parent = parent.parent()
        return tuple(reversed(key_path))

    def _indexFromKey(self, key_path):
        """Return an source QModelIndex for the given key.
        """
        index = self.sourceModel().index(key_path[0], 0)
        for row in key_path[1:]:
            index = index.child(row, 0)
        return index

    def _updateRowMapping(self):
        source = self.sourceModel()

        source_key = []
        source_offset_map = {}

        def create_mapping(index, key_path):
            if source.rowCount(index) > 0:
                if self.__flatteningMode != self.LeavesOnly:
                    source_offset_map[key_path] = len(source_offset_map)
                    source_key.append(key_path)

                for i in range(source.rowCount(index)):
                    create_mapping(index.child(i, 0), key_path + (i, ))

            else:
                source_offset_map[key_path] = len(source_offset_map)
                source_key.append(key_path)

        for i in range(source.rowCount()):
            create_mapping(source.index(i, 0), (i,))

        self._source_key = source_key
        self._source_offset = source_offset_map

    def _sourceDataChanged(self, top, bottom):
        changed_indexes = []
        for i in range(top.row(), bottom.row() + 1):
            source_ind = top.sibling(i, 0)
            changed_indexes.append(source_ind)

        for ind in changed_indexes:
            self.dataChanged.emit(ind, ind)

    def _sourceRowsInserted(self, parent, start, end):
        self.beginResetModel()
        self._updateRowMapping()
        self.endResetModel()

    def _sourceRowsRemoved(self, parent, start, end):
        self.beginResetModel()
        self._updateRowMapping()
        self.endResetModel()

    def _sourceRowsMoved(self, sourceParent, sourceStart, sourceEnd,
                         destParent, destRow):
        self.beginResetModel()
        self._updateRowMapping()
        self.endResetModel()
