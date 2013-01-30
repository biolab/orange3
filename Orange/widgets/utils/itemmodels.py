import pickle

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from contextlib import contextmanager
from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable
from Orange.widgets import gui


class _store(dict):
    pass


def _argsort(seq, cmp=None, key=None, reverse=False):
    if key is not None:
        items = sorted(enumerate(seq), key=lambda i, v: key(v))
    elif cmp is not None:
        items = sorted(enumerate(seq), cmp=lambda a, b: cmp(a[1], b[1]))
    else:
        items = sorted(enumerate(seq), key=seq.__getitem__)
    if reverse:
        items = reversed(items)
    return items


@contextmanager
def signal_blocking(obj):
    blocked = obj.signalsBlocked()
    obj.blockSignals(True)
    yield
    obj.blockSignals(blocked)


class PyListModel(QAbstractListModel):
    """ A model for displaying python list like objects in Qt item view classes
    """
    MIME_TYPES = ["application/x-Orange-PyListModelData"]

    def __init__(self, iterable=None, parent=None,
                 flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled,
                 list_item_role=Qt.DisplayRole,
                 supportedDropActions=Qt.MoveAction):
        super().__init__(parent)
        self._list = []
        self._other_data = []
        self._flags = flags
        self.list_item_role = list_item_role

        self._supportedDropActions = supportedDropActions
        if iterable is not None:
            self.extend(iterable)

    def _is_index_valid_for(self, index, list_like):
        if isinstance(index, QModelIndex) and index.isValid():
            row, column = index.row(), index.column()
            return 0 <= row < len(list_like) and not column
        elif isinstance(index, int):
            return -len(self) < index < len(list_like)
        else:
            return False

    def wrap(self, lst):
        """ Wrap the list with this model. All changes to the model
        are done in place on the passed list
        """
        self._list = lst
        self._other_data = [_store() for _ in lst]
        self.reset()


    # noinspection PyMethodOverriding
    def index(self, row, column=0, parent=QModelIndex()):
        if self._is_index_valid_for(row, self) and column == 0:
            return QAbstractListModel.createIndex(self, row, column, parent)
        else:
            return QModelIndex()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(section)


    # noinspection PyMethodOverriding
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 1

    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role in [self.list_item_role, Qt.EditRole] \
                and self._is_index_valid_for(index, self):
            return self[row]
        elif self._is_index_valid_for(row, self._other_data):
            return self._other_data[row].get(role, None)

    def itemData(self, index):
        mapping = QAbstractListModel.itemData(self, index)
        if self._is_index_valid_for(index, self._other_data):
            items = list(self._other_data[index.row()].items())
        else:
            items = []
        for key, value in items:
            mapping[key] = value
        return mapping

    def parent(self, index=QModelIndex()):
        return QModelIndex()

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and self._is_index_valid_for(index, self):
            self[index.row()] = value  # Will emit proper dataChanged signal
            return True
        elif self._is_index_valid_for(index, self._other_data):
            self._other_data[index.row()][role] = value
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    def setItemData(self, index, data):
        data = dict(data)
        with signal_blocking(self):
            for role, value in data.items():
                if role == Qt.EditRole and \
                        self._is_index_valid_for(index, self):
                    self[index.row()] = value
                elif self._is_index_valid_for(index, self._other_data):
                    self._other_data[index.row()][role] = value

        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        if self._is_index_valid_for(index, self._other_data):
            return self._other_data[index.row()].get("flags", self._flags)
        else:
            return self._flags | Qt.ItemIsDropEnabled


    # noinspection PyMethodOverriding
    def insertRows(self, row, count, parent=QModelIndex()):
        """ Insert ``count`` rows at ``row``, the list fill be filled
        with ``None``
        """
        if not parent.isValid():
            self[row:row] = [None] * count
            return True
        else:
            return False


    # noinspection PyMethodOverriding
    def removeRows(self, row, count, parent=QModelIndex()):
        """Remove ``count`` rows starting at ``row``
        """
        if not parent.isValid():
            del self[row:row + count]
            return True
        else:
            return False

    def extend(self, iterable):
        list_ = list(iterable)
        self.beginInsertRows(QModelIndex(),
                             len(self), len(self) + len(list_) - 1)
        self._list.extend(list_)
        self._other_data.extend([_store() for _ in list_])
        self.endInsertRows()

    def append(self, item):
        self.extend([item])

    def insert(self, i, val):
        self.beginInsertRows(QModelIndex(), i, i)
        self._list.insert(i, val)
        self._other_data.insert(i, _store())
        self.endInsertRows()

    def remove(self, val):
        i = self._list.index(val)
        self.__delitem__(i)

    def pop(self, i):
        item = self._list[i]
        self.__delitem__(i)
        return item

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __add__(self, iterable):
        new_list = PyListModel(list(self._list),
                               self.parent(),
                               flags=self._flags,
                               list_item_role=self.list_item_role,
                               supportedDropActions=self.supportedDropActions()
                               )
        new_list._other_data = list(self._other_data)
        new_list.extend(iterable)
        return new_list

    def __iadd__(self, iterable):
        self.extend(iterable)

    def __delitem__(self, s):
        if isinstance(s, slice):
            i, j, _ = s.indices(len(self))
            self.beginRemoveRows(QModelIndex(), i, j - 1)
        else:
            self.beginRemoveRows(QModelIndex(), s, s)
        del self._list[s]
        del self._other_data[s]
        self.endRemoveRows()

    def __setitem__(self, s, value):
        if isinstance(s, slice):
            self.beginResetModel()
            if not isinstance(value, list):
                value = list(value)
            self._list[s] = value
            self._other_data[s] = (_store() for _ in value)
            self.endResetModel()
        else:
            self._list[s] = value
            self._other_data[s] = _store()
            self.dataChanged.emit(self.index(s), self.index(s))

    def reverse(self):
        self._list.reverse()
        self._other_data.reverse()
        self.dataChanged.emit(self.index(0), self.index(len(self) - 1))

    def sort(self, *args, **kwargs):
        indices = _argsort(self._list, *args, **kwargs)
        lst = [self._list[i] for i in indices]
        other = [self._other_data[i] for i in indices]
        for i, new_l, new_o in enumerate(zip(lst, other)):
            self._list[i] = new_l
            self._other_data[i] = new_o
        self.dataChanged.emit(self.index(0), self.index(len(self) - 1))

    def __repr__(self):
        return "PyListModel(%s)" % repr(self._list)

    def __bool__(self):
        return len(self) != 0

    def emitDataChanged(self, indexList):
        if isinstance(indexList, int):
            indexList = [indexList]

        #TODO: group indexes into ranges
        for ind in indexList:
            self.dataChanged.emit(self.index(ind), self.index(ind))

    ###########
    # Drag/drop
    ###########

    def supportedDropActions(self):
        return self._supportedDropActions

    def mimeTypes(self):
        return self.MIME_TYPES + list(QAbstractListModel.mimeTypes(self))

    def mimeData(self, indexlist):
        if len(indexlist) <= 0:
            return None

        items = [self[i.row()] for i in indexlist]
        mime = QAbstractListModel.mimeData(self, indexlist)
        data = pickle.dumps(vars)
        mime.setData(self.MIME_TYPE, QByteArray(data))
        mime._items = items
        return mime

    def dropMimeData(self, mime, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True

        if not mime.hasFormat(self.MIME_TYPE):
            return False

        if hasattr(mime, "_vars"):
            vars_ = mime._vars
        else:
            desc = str(mime.data(self.MIME_TYPE))
            vars_ = pickle.loads(desc)

        return QAbstractListModel.dropMimeData(
            self, mime, action, row, column, parent)


class VariableListModel(PyListModel):

    MIME_TYPE = "application/x-Orange-VariableList"

    def data(self, index, role=Qt.DisplayRole):
        if self._is_index_valid_for(index, self):
            i = index.row()
            var = self[i]
            if role == Qt.DisplayRole:
                return var.name
            elif role == Qt.DecorationRole:
                return gui.getAttributeIcons().get(var.var_type, -1)
            elif role == Qt.ToolTipRole:
                return self.variable_tooltip(var)
            else:
                return PyListModel.data(self, index, role)

    def variable_tooltip(self, var):
        if isinstance(var, DiscreteVariable):
            return self.discrete_variable_tooltip(var)
        elif isinstance(var, ContinuousVariable):
            return self.continuous_variable_toltip(var)
        elif isinstance(var, StringVariable):
            return self.string_variable_tooltip(var)

    def variable_labels_tooltip(self, var):
        text = ""
        if var.attributes:
            items = [(safe_text(key), safe_text(value))
                     for key, value in var.attributes.items()]
            labels = list(map("%s = %s".__mod__, items))
            text += "<br/>Variable Labels:<br/>"
            text += "<br/>".join(labels)
        return text

    def discrete_variable_tooltip(self, var):
        text = "<b>%s</b><br/>Discrete with %i values: " %\
               (safe_text(var.name), len(var.values))
        text += ", ".join("%r" % safe_text(v) for v in var.values)
        text += self.variable_labels_tooltip(var)
        return text

    def continuous_variable_toltip(self, var):
        text = "<b>%s</b><br/>Continuous" % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

    def string_variable_tooltip(self, var):
        text = "<b>%s</b><br/>String" % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

    def python_variable_tooltip(self, var):
        text = "<b>%s</b><br/>Python" % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

_html_replace = [("<", "&lt;"), (">", "&gt;")]


def safe_text(text):
    for old, new in _html_replace:
        text = text.replace(old, new)
    return text


class VariableEditor(QWidget):
    def __init__(self, var, parent):
        QWidget.__init__(self, parent)
        self.var = var
        layout = QHBoxLayout()
        self._attrs = gui.getAttributeIcons()
        self.type_cb = QComboBox(self)
        for attr, icon in self._attrs.items():
            if attr != -1:
                self.type_cb.addItem(icon, str(attr))
        layout.addWidget(self.type_cb)

        self.name_le = QLineEdit(self)
        layout.addWidget(self.name_le)

        self.setLayout(layout)

        self.type_cb.currentIndexChanged.connect(self.edited)
        self.name_le.editingFinished.connect(self.edited)

    def edited(self, *_):
        self.emit(SIGNAL("edited()"))

    def setData(self, tpe, name):
        self.type_cb.setCurrentIndex(list(self._attr.keys()).index(tpe))
        self.name_le.setText(name)


class DiscreteVariableEditor(VariableEditor):
    def __init__(self, var, parent):
        VariableEditor.__init__(self, var, parent)


class ContinuousVariableEditor(QLineEdit):
    def setVariable(self, var):
        self.setText(var.name)

    def getVariable(self):
        return ContinuousVariable(self.text())


class StringVariableEditor(QLineEdit):
    def setVariable(self, var):
        self.setText(str(var.name))

    def getVariable(self):
        return StringVariable(self.text())


class VariableDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        var = index.data(Qt.EditRole)
        if isinstance(var, DiscreteVariable):
            return DiscreteVariableEditor(var, parent)
        elif isinstance(var, ContinuousVariable):
            return ContinuousVariableEditor(var, parent)
        elif isinstance(var, StringVariable):
            return StringVariableEditor(var, parent)
#        return VariableEditor(var, parent)

    def setEditorData(self, editor, index):
        var = index.data(Qt.EditRole)
        editor.variable = var

    def setModelData(self, editor, model, index):
        model.setData(index, editor.variable, Qt.EditRole)


class ListSingleSelectionModel(QItemSelectionModel):
    """ Item selection model for list item models with single selection.

    Defines signal:
        - selectedIndexChanged(QModelIndex)

    """
    def __init__(self, model, parent=None):
        QItemSelectionModel.__init__(self, model, parent)
        self.selectionChanged.connect(self.onSelectionChanged)

    def onSelectionChanged(self, new, _):
        index = list(new.indexes())
        if index:
            index = index.pop()
        else:
            index = QModelIndex()
        self.emit(SIGNAL("selectedIndexChanged(QModelIndex)"), index)

    def selectedRow(self):
        """ Return QModelIndex of the selected row or invalid if no selection.
        """
        rows = self.selectedRows()
        if rows:
            return rows[0]
        else:
            return QModelIndex()

    def select(self, index, flags=QItemSelectionModel.ClearAndSelect):
        if isinstance(index, int):
            index = self.model().index(index)
        return QItemSelectionModel.select(self, index, flags)


class ModelActionsWidget(QWidget):
    def __init__(self, actions=None, parent=None,
                 direction=QBoxLayout.LeftToRight):
        QWidget.__init__(self, parent)
        self.actions = []
        self.buttons = []
        layout = QBoxLayout(direction)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        if actions is not None:
            for action in actions:
                self.addAction(action)
        self.setLayout(layout)

    def actionButton(self, action):
        if isinstance(action, QAction):
            button = QToolButton(self)
            button.setDefaultAction(action)
            return button
        elif isinstance(action, QAbstractButton):
            return action

    def insertAction(self, ind, action, *args):
        button = self.actionButton(action)
        self.layout().insertWidget(ind, button, *args)
        self.buttons.insert(ind, button)
        self.actions.insert(ind, action)
        return button

    def addAction(self, action, *args):
        return self.insertAction(-1, action, *args)
