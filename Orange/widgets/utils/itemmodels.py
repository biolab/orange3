import pickle
from numbers import Number, Integral
from math import isnan, isinf

import operator
from collections import namedtuple, Sequence, defaultdict
from contextlib import contextmanager
from functools import reduce, partial, lru_cache
from xml.sax.saxutils import escape

from PyQt4.QtGui import  QItemSelectionModel, QColor
from PyQt4.QtCore import (
    Qt, QAbstractListModel, QAbstractTableModel, QModelIndex, QByteArray
)
from PyQt4.QtCore import pyqtSignal as Signal

from PyQt4.QtGui import (
    QWidget, QBoxLayout, QToolButton, QAbstractButton, QAction
)

import numpy

from Orange.data import Variable, Storage
from Orange.widgets import gui
from Orange.widgets.utils import datacaching
from Orange.statistics import basic_stats


class _store(dict):
    pass


def _argsort(seq, cmp=None, key=None, reverse=False):
    if key is not None:
        return sorted(enumerate(seq), key=lambda pair: key(pair[1]), reverse=reverse)
    elif cmp is not None:
        from functools import cmp_to_key
        return sorted(enumerate(seq), key=cmp_to_key(lambda a, b: cmp(a[1], b[1])), reverse=reverse)
    else:
        return sorted(enumerate(seq), key=operator.itemgetter(1), reverse=reverse)


@contextmanager
def signal_blocking(obj):
    blocked = obj.signalsBlocked()
    obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(blocked)


def _as_contiguous_range(the_slice, length):
    start, stop, step = the_slice.indices(length)
    if step == -1:
        # Equivalent range with positive step
        start, stop, step = stop + 1, start + 1, 1
    elif not (step == 1 or step is None):
        raise IndexError("Non-contiguous range.")
    return start, stop, step


class PyTableModel(QAbstractTableModel):
    """ A model for displaying python tables (sequences of sequences) in
    QTableView objects.

    Parameters
    ----------
    sequence : list
        The initial list to wrap.
    parent : QObject
        Parent QObject.
    editable: bool or sequence
        If True, all items are flagged editable. If sequence, the True-ish
        fields mark their respective columns editable.

    Notes
    -----
    The model rounds numbers to human readable precision, e.g.:
    1.23e-04, 1.234, 1234.5, 12345, 1.234e06.

    To set additional item roles, use setData().
    """

    @staticmethod
    def _RoleData():
        return defaultdict(lambda: defaultdict(dict))

    # All methods are either necessary overrides of super methods, or
    # methods likened to the Python list's. Hence, docstrings aren't.
    # pylint: disable=missing-docstring
    def __init__(self, sequence=None, parent=None, editable=False):
        super().__init__(parent)
        self._headers = {}
        self._editable = editable
        self._table = None
        self._roleData = None
        self.wrap(sequence or [])
        self.view = None

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else max(map(len, self._table), default=0)

    def flags(self, index):
        flags = super().flags(index)
        if not self._editable or not index.isValid():
            return flags
        if isinstance(self._editable, Sequence):
            return flags | Qt.ItemIsEditable if self._editable[index.column()] else flags
        return flags | Qt.ItemIsEditable

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self[index.row()][index.column()] = value
            self.dataChanged.emit(index, index)
        else:
            self._roleData[index.row()][index.column()][role] = value
        return True

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return

        role_value = (self._roleData
                      .get(index.row(), {})
                      .get(index.column(), {})
                      .get(role))
        if role_value is not None:
            return role_value

        try:
            value = self[index.row()][index.column()]
        except IndexError:
            return
        if role == Qt.EditRole:
            return value
        if role == Qt.DecorationRole and isinstance(value, Variable):
            return gui.attributeIconDict[value]
        if role == Qt.DisplayRole:
            if (isinstance(value, Number) and
                    not (isnan(value) or isinf(value) or isinstance(value, Integral))):
                absval = abs(value)
                strlen = len(str(int(absval)))
                value = '{:.{}{}}'.format(value,
                                          2 if absval < .001 else
                                          3 if strlen < 2 else
                                          1 if strlen < 5 else
                                          0 if strlen < 6 else
                                          3,
                                          'f' if (absval == 0 or
                                                  absval >= .001 and
                                                  strlen < 6)
                                          else 'e')
            return str(value)
        if role == Qt.TextAlignmentRole and isinstance(value, Number):
            return Qt.AlignRight | Qt.AlignVCenter
        if role == Qt.ToolTipRole:
            return str(value)

    def sort(self, column, order=Qt.AscendingOrder):
        self.beginResetModel()
        indices = sorted(range(len(self._table)),
                         key=lambda i: self._table[i][column],
                         reverse=order != Qt.AscendingOrder)
        self._table[:] = [self._table[i] for i in indices]

        rd = self._roleData
        self._roleData = self._RoleData()
        self._roleData.update((i, rd.get(row))
                              for i, row in enumerate(indices)
                              if rd.get(row))

        vheaders = self._headers.get(Qt.Vertical, ())
        if vheaders:
            vheaders = tuple(vheaders) + ('',) * max(0, (len(self._table) - len(vheaders)))
            vheaders = [vheaders[i] for i in indices]
            self._headers[Qt.Vertical] = vheaders
        self.endResetModel()

    def setHorizontalHeaderLabels(self, labels):
        self._headers[Qt.Horizontal] = labels

    def setVerticalHeaderLabels(self, labels):
        self._headers[Qt.Vertical] = labels

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            headers = self._headers.get(orientation)
            return headers[section] if headers and section < len(headers) else str(section)

    def removeRows(self, row, count, parent=QModelIndex()):
        if not parent.isValid():
            del self[row:row + count]
            for row in range(row, row + count):
                self._roleData.pop(row, None)
            return True
        return False

    def removeColumns(self, column, count, parent=QModelIndex()):
        self.beginRemoveColumns(parent, column, column + count - 1)
        for row in self._table:
            del row[column:column + count]
        for cols in self._roleData.values():
            for col in range(column, column + count):
                cols.pop(col, None)
        del self._headers.get(Qt.Horizontal, [])[column:column + count]
        self.endRemoveColumns()

    def insertRows(self, row, count, parent=QModelIndex()):
        self.beginInsertRows(parent, row, row + count - 1)
        self._table[row:row] = [[''] * self.columnCount() for _ in range(count)]
        self.endInsertRows()

    def insertColumns(self, column, count, parent=QModelIndex()):
        self.beginInsertColumns(parent, column, column + count - 1)
        for row in self._table:
            row[column:column] = [''] * count
        self.endInsertColumns()

    def __len__(self):
        return len(self._table)

    def __bool__(self):
        return len(self) != 0

    def __iter__(self):
        return iter(self._table)

    def __getitem__(self, item):
        return self._table[item]

    def __delitem__(self, i):
        if isinstance(i, slice):
            start, stop, _ = _as_contiguous_range(i, len(self))
            stop -= 1
        else:
            start = stop = i = i if i >= 0 else len(self) + i
        self.beginRemoveRows(QModelIndex(), start, stop)
        del self._table[i]
        self.endRemoveRows()

    def __setitem__(self, i, value):
        if isinstance(i, slice):
            start, stop, _ = _as_contiguous_range(i, len(self))
            stop -= 1
        else:
            start = stop = i = i if i >= 0 else len(self) + i
        self._table[i] = value
        self.dataChanged.emit(self.index(start, 0),
                              self.index(stop, self.columnCount() - 1))

    def wrap(self, table):
        self.beginResetModel()
        self._table = table
        self._roleData = self._RoleData()
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._table.clear()
        self._roleData.clear()
        self.endResetModel()

    def append(self, row):
        self.extend([row])

    def _insertColumns(self, rows):
        n_max = max(map(len, rows))
        if self.columnCount() < n_max:
            self.insertColumns(self.columnCount(), n_max - self.columnCount())

    def extend(self, rows):
        i, rows = len(self), list(rows)
        self.insertRows(i, len(rows))
        self._insertColumns(rows)
        self[i:] = rows

    def insert(self, i, row):
        self.insertRows(i, 1)
        self._insertColumns((row,))
        self[i] = row

    def remove(self, val):
        del self[self._table.index(val)]


class PyListModel(QAbstractListModel):
    """ A model for displaying python list like objects in Qt item view classes
    """
    MIME_TYPES = ["application/x-Orange-PyListModelData"]
    Separator = object()

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
        self.beginResetModel()
        self._list = lst
        self._other_data = [_store() for _ in lst]
        self.endResetModel()


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

    def indexOf(self, value):
        return self._list.index(value)

    def clear(self):
        del self[:]

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
            start, stop, step = _as_contiguous_range(s, len(self))
            self.beginRemoveRows(QModelIndex(), start, stop - 1)
        else:
            s = operator.index(s)
            s = len(self) + s if s < 0 else s
            self.beginRemoveRows(QModelIndex(), s, s)
        del self._list[s]
        del self._other_data[s]
        self.endRemoveRows()

    def __setitem__(self, s, value):
        if isinstance(s, slice):
            start, stop, step = _as_contiguous_range(s, len(self))
            self.__delitem__(slice(start, stop, step))

            if not isinstance(value, list):
                value = list(value)
            separators = [start + i for i, v in enumerate(value) if v is self.Separator]
            self.beginInsertRows(QModelIndex(), start, start + len(value) - 1)
            self._list[start:start] = value
            self._other_data[start:start] = (_store() for _ in value)
            for idx in separators:
                self._other_data[idx]['flags'] = Qt.NoItemFlags
                self._other_data[idx][Qt.AccessibleDescriptionRole] = 'separator'
            self.endInsertRows()
        else:
            s = operator.index(s)
            s = len(self) + s if s < 0 else s
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
        mime.set_data(self.MIME_TYPE, QByteArray(data))
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


class PyListModelTooltip(PyListModel):
    def __init__(self):
        super().__init__()
        self.tooltips = []

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.ToolTipRole:
            return self.tooltips[index.row()]
        else:
            return super().data(index, role)


class VariableListModel(PyListModel):

    MIME_TYPE = "application/x-Orange-VariableList"

    def data(self, index, role=Qt.DisplayRole):
        if self._is_index_valid_for(index, self):
            var = self[index.row()]
            if not isinstance(var, Variable):
                return super().data(index, role)
            elif role == Qt.DisplayRole:
                return var.name
            elif role == Qt.DecorationRole:
                return gui.attributeIconDict[var]
            elif role == Qt.ToolTipRole:
                return self.variable_tooltip(var)
            else:
                return PyListModel.data(self, index, role)

    def variable_tooltip(self, var):
        if var.is_discrete:
            return self.discrete_variable_tooltip(var)
        elif var.is_continuous:
            return self.continuous_variable_toltip(var)
        elif var.is_string:
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
        text = str(text).replace(old, new)
    return text


class ListSingleSelectionModel(QItemSelectionModel):
    """ Item selection model for list item models with single selection.

    Defines signal:
        - selectedIndexChanged(QModelIndex)

    """
    selectedIndexChanged = Signal(QModelIndex)

    def __init__(self, model, parent=None):
        QItemSelectionModel.__init__(self, model, parent)
        self.selectionChanged.connect(self.onSelectionChanged)

    def onSelectionChanged(self, new, _):
        index = list(new.indexes())
        if index:
            index = index.pop()
        else:
            index = QModelIndex()

        self.selectedIndexChanged.emit(index)

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


def select_row(view, row):
    """
    Select a `row` in an item view.
    """
    selmodel = view.selectionModel()
    selmodel.select(view.model().index(row, 0),
                    QItemSelectionModel.ClearAndSelect |
                    QItemSelectionModel.Rows)


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


class TableModel(QAbstractTableModel):
    """
    An adapter for using Orange.data.Table within Qt's Item View Framework.

    :param Orange.data.Table sourcedata: Source data table.
    :param QObject parent:
    """
    #: Orange.data.Value for the index.
    ValueRole = gui.TableValueRole  # next(gui.OrangeUserRole)
    #: Orange.data.Value of the row's class.
    ClassValueRole = gui.TableClassValueRole  # next(gui.OrangeUserRole)
    #: Orange.data.Variable of the column.
    VariableRole = gui.TableVariable  # next(gui.OrangeUserRole)
    #: Basic statistics of the column
    VariableStatsRole = next(gui.OrangeUserRole)
    #: The column's role (position) in the domain.
    #: One of Attribute, ClassVar or Meta
    DomainRole = next(gui.OrangeUserRole)

    #: Column domain roles
    ClassVar, Meta, Attribute = range(3)

    #: Default background color for domain roles
    ColorForRole = {
        ClassVar: QColor(160, 160, 160),
        Meta: QColor(220, 220, 200),
        Attribute: None,
    }

    #: Standard column descriptor
    Column = namedtuple(
        "Column", ["var", "role", "background", "format"])
    #: Basket column descriptor (i.e. sparse X/Y/metas/ compressed into
    #: a single column).
    Basket = namedtuple(
        "Basket", ["vars", "role", "background", "density", "format"])

    def __init__(self, sourcedata, parent=None):
        super().__init__(parent)
        self.source = sourcedata
        self.domain = domain = sourcedata.domain

        self.X_density = sourcedata.X_density()
        self.Y_density = sourcedata.Y_density()
        self.M_density = sourcedata.metas_density()

        def format_sparse(vars, datagetter, instance):
            data = datagetter(instance)
            return ", ".join("{}={}".format(vars[i].name, vars[i].repr_val(v))
                             for i, v in zip(data.indices, data.data))

        def format_sparse_bool(vars, datagetter, instance):
            data = datagetter(instance)
            return ", ".join(vars[i].name for i in data.indices)

        def format_dense(var, instance):
            return str(instance[var])

        def make_basket_formater(vars, density, role):
            formater = (format_sparse if density == Storage.SPARSE
                        else format_sparse_bool)
            if role == TableModel.Attribute:
                getter = operator.attrgetter("sparse_x")
            elif role == TableModel.ClassVar:
                getter = operator.attrgetter("sparse_y")
            elif role == TableModel.Meta:
                getter = operator.attrgetter("sparse_meta")
            return partial(formater, vars, getter)

        def make_basket(vars, density, role):
            return TableModel.Basket(
                vars, TableModel.Attribute, self.ColorForRole[role], density,
                make_basket_formater(vars, density, role)
            )

        def make_column(var, role):
            return TableModel.Column(
                var, role, self.ColorForRole[role],
                partial(format_dense, var)
            )

        columns = []

        if self.Y_density != Storage.DENSE:
            coldesc = make_basket(domain.class_vars, self.Y_density,
                                  TableModel.ClassVar)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.ClassVar)
                        for var in domain.class_vars]

        if self.M_density != Storage.DENSE:
            coldesc = make_basket(domain.metas, self.M_density,
                                  TableModel.Meta)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.Meta)
                        for var in domain.metas]

        if self.X_density != Storage.DENSE:
            coldesc = make_basket(domain.attributes, self.X_density,
                                  TableModel.Attribute)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.Attribute)
                        for var in domain.attributes]

        #: list of all domain variables (class_vars + metas + attrs)
        self.vars = domain.class_vars + domain.metas + domain.attributes
        self.columns = columns

        #: A list of all unique attribute labels (in all variables)
        self._labels = sorted(
            reduce(operator.ior,
                   [set(var.attributes) for var in self.vars],
                   set()))

        @lru_cache(maxsize=1000)
        def row_instance(index):
            return self.source[int(index)]
        self._row_instance = row_instance

        # column basic statistics (VariableStatsRole), computed when
        # first needed.
        self.__stats = None
        self.__rowCount = sourcedata.approx_len()
        self.__columnCount = len(self.columns)

        if self.__rowCount > (2 ** 31 - 1):
            raise ValueError("len(sourcedata) > 2 ** 31 - 1")

        self.__sortColumn = -1
        self.__sortOrder = Qt.AscendingOrder
        # Indices sorting the source table
        self.__sortInd = None
        # The inverse of __sortInd
        self.__sortIndInv = None

    def sort(self, column, order):
        """
        Sort the data by `column` index into `order`

        To reset the sort order pass -1 as the column.

        :type column: int
        :type order: Qt.SortOrder

        Reimplemented from QAbstractItemModel.sort

        .. note::
            This only affects the model's data presentation, the
            underlying data table is left unmodified.

        """
        self.layoutAboutToBeChanged.emit()

        # Store persistent indices as well as their (actual) rows in the
        # source data table.
        persistent = self.persistentIndexList()
        persistent_rows = numpy.array([ind.row() for ind in persistent], int)
        if self.__sortInd is not None:
            persistent_rows = self.__sortInd[persistent_rows]

        self.__sortColumn = column
        self.__sortOrder = order

        if column < 0:
            indices = None
        else:
            keydata = self.columnSortKeyData(column, TableModel.ValueRole)
            if keydata is not None:
                if keydata.dtype == object:
                    indices = sorted(range(self.__rowCount),
                                     key=lambda i: str(keydata[i]))
                    indices = numpy.array(indices)
                else:
                    indices = numpy.argsort(keydata, kind="mergesort")
            else:
                indices = numpy.arange(0, self.__rowCount)

            if order == Qt.DescendingOrder:
                indices = indices[::-1]

            if self.__sortInd is not None:
                indices = self.__sortInd[indices]

        if indices is not None:
            self.__sortInd = indices
            self.__sortIndInv = numpy.argsort(indices)
        else:
            self.__sortInd = None
            self.__sortIndInv = None

        if self.__sortInd is not None:
            persistent_rows = self.__sortIndInv[persistent_rows]

        for pind, row in zip(persistent, persistent_rows):
            self.changePersistentIndex(pind, self.index(row, pind.column()))
        self.layoutChanged.emit()

    def columnSortKeyData(self, column, role):
        """
        Return a sequence of objects which can be used as `keys` for sorting.

        :param int column: Sort column.
        :param Qt.ItemRole role: Sort item role.

        """
        coldesc = self.columns[column]
        if isinstance(coldesc, TableModel.Column) \
                and role == TableModel.ValueRole:
            col_view, _ = self.source.get_column_view(coldesc.var)
            col_data = numpy.asarray(col_view)
            if self.__sortInd is not None:
                col_data = col_data[self.__sortInd]
            return col_data
        else:
            if self.__sortInd is not None:
                indices = self.__sortInd
            else:
                indices = range(self.rowCount())
            return numpy.asarray([self.index(i, column).data(role)
                                  for i in indices])

    def sortColumn(self):
        """
        The column currently used for sorting (-1 if no sorting is applied).
        """
        return self.__sortColumn

    def sortOrder(self):
        """
        The current sort order.
        """
        return self.__sortOrder

    def mapToTableRows(self, modelrows):
        """
        Return the row indices in the source table for the given model rows.
        """
        if self.__sortColumn < 0:
            return modelrows
        else:
            return self.__sortInd[modelrows].tolist()

    def mapFromTableRows(self, tablerows):
        """
        Return the row indices in the model for the given source table rows.
        """
        if self.__sortColumn < 0:
            return tablerows
        else:
            return self.__sortIndInv[tablerows].tolist()

    def data(self, index, role,
             # For optimizing out LOAD_GLOBAL byte code instructions in
             # the item role tests.
             _str=str,
             _Qt_DisplayRole=Qt.DisplayRole,
             _Qt_EditRole=Qt.EditRole,
             _Qt_BackgroundRole=Qt.BackgroundRole,
             _ValueRole=ValueRole,
             _ClassValueRole=ClassValueRole,
             _VariableRole=VariableRole,
             _DomainRole=DomainRole,
             _VariableStatsRole=VariableStatsRole,
             # Some cached local precomputed values.
             # All of the above roles we respond to
             _recognizedRoles=set([Qt.DisplayRole,
                                   Qt.EditRole,
                                   Qt.BackgroundRole,
                                   ValueRole,
                                   ClassValueRole,
                                   VariableRole,
                                   DomainRole,
                                   VariableStatsRole]),
             ):
        """
        Reimplemented from `QAbstractItemModel.data`
        """
        if role not in _recognizedRoles:
            return None

        row, col = index.row(), index.column()
        if  not 0 <= row <= self.__rowCount:
            return None

        if self.__sortInd is not None:
            row = self.__sortInd[row]

        try:
            instance = self._row_instance(row)
        except IndexError:
            self.layoutAboutToBeChanged.emit()
            self.beginRemoveRows(self.parent(), row, max(self.rowCount(), row))
            self.__rowCount = min(row, self.__rowCount)
            self.endRemoveRows()
            self.layoutChanged.emit()
            return None
        coldesc = self.columns[col]

        if role == _Qt_DisplayRole:
            return coldesc.format(instance)
        elif role == _Qt_EditRole and isinstance(coldesc, TableModel.Column):
            return instance[coldesc.var]
        elif role == _Qt_BackgroundRole:
            return coldesc.background
            return self.color_for_role[coldesc.role]
        elif role == _ValueRole and isinstance(coldesc, TableModel.Column):
            return instance[coldesc.var]
        elif role == _ClassValueRole:
            try:
                return instance.get_class()
            except TypeError:
                return None
        elif role == _VariableRole and isinstance(coldesc, TableModel.Column):
            return coldesc.var
        elif role == _DomainRole:
            return coldesc.role
        elif role == _VariableStatsRole:
            return self._stats_for_column(col)
        else:
            return None

    def setData(self, index, value, role):
        row, col = self.__sortIndInv[index.row()], index.column()
        if role == Qt.EditRole:
            try:
                self.source[row, col] = value
            except (TypeError, IndexError):
                return False
            else:
                self.dataChanged.emit(index, index)
                return True
        else:
            return False

    def parent(self, index=QModelIndex()):
        """Reimplemented from `QAbstractTableModel.parent`."""
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        """Reimplemented from `QAbstractTableModel.rowCount`."""
        return 0 if parent.isValid() else self.__rowCount

    def columnCount(self, parent=QModelIndex()):
        """Reimplemented from `QAbstractTableModel.columnCount`."""
        return 0 if parent.isValid() else self.__columnCount

    def headerData(self, section, orientation, role):
        """Reimplemented from `QAbstractTableModel.headerData`."""
        if orientation == Qt.Vertical:
            if role == Qt.DisplayRole:
                if self.__sortInd is not None:
                    return int(self.__sortInd[section] + 1)
                else:
                    return int(section + 1)
            else:
                return None

        coldesc = self.columns[section]
        if role == Qt.DisplayRole:
            if isinstance(coldesc, TableModel.Basket):
                return "{...}"
            else:
                return coldesc.var.name
        elif role == Qt.ToolTipRole:
            return self._tooltip(coldesc)
        elif role == TableModel.VariableRole \
                and isinstance(coldesc, TableModel.Column):
            return coldesc.var
        elif role == TableModel.VariableStatsRole:
            return self._stats_for_column(section)
        elif role == TableModel.DomainRole:
            return coldesc.role
        else:
            return None

    def _tooltip(self, coldesc):
        """
        Return an header tool tip text for an `column` descriptor.
        """
        if isinstance(coldesc, TableModel.Basket):
            return None

        labels = self._labels
        variable = coldesc.var
        pairs = [(escape(key), escape(str(variable.attributes[key])))
                 for key in labels if key in variable.attributes]
        tip = "<b>%s</b>" % escape(variable.name)
        tip = "<br/>".join([tip] + ["%s = %s" % pair for pair in pairs])
        return tip

    def _stats_for_column(self, column):
        """
        Return BasicStats for `column` index.
        """
        coldesc = self.columns[column]
        if isinstance(coldesc, TableModel.Basket):
            return None

        if self.__stats is None:
            self.__stats = datacaching.getCached(
                self.source, basic_stats.DomainBasicStats,
                (self.source, True)
            )

        return self.__stats[coldesc.var]
