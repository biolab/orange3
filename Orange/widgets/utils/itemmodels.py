from numbers import Number, Integral
from math import isnan, isinf

import operator
from collections import namedtuple, Sequence, defaultdict
from contextlib import contextmanager
from functools import reduce, partial, lru_cache, wraps
from itertools import chain
from warnings import warn
from xml.sax.saxutils import escape

from AnyQt.QtCore import (
    Qt, QObject, QAbstractListModel, QAbstractTableModel, QModelIndex,
    QItemSelectionModel, QT_VERSION
)
from AnyQt.QtCore import pyqtSignal as Signal
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import (
    QWidget, QBoxLayout, QToolButton, QAbstractButton, QAction
)

import numpy

from Orange.data import Variable, Storage, DiscreteVariable, ContinuousVariable
from Orange.data.domain import filter_visible
from Orange.widgets import gui
from Orange.widgets.utils import datacaching
from Orange.statistics import basic_stats
from Orange.util import deprecated


class _store(dict):
    pass


def _argsort(seq, cmp=None, key=None, reverse=False):
    indices = range(len(seq))
    if key is not None:
        return sorted(indices, key=lambda i: key(seq[i]), reverse=reverse)
    elif cmp is not None:
        from functools import cmp_to_key
        return sorted(indices, key=cmp_to_key(lambda a, b: cmp(seq[a], seq[b])),
                      reverse=reverse)
    else:
        return sorted(indices, key=lambda i: seq[i], reverse=reverse)


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


class AbstractSortTableModel(QAbstractTableModel):
    """
    A sorting proxy table model that sorts its rows in fast numpy,
    avoiding potentially thousands of calls into
    ``QSortFilterProxyModel.lessThan()`` or any potentially costly
    reordering of original data.

    Override ``sortColumnData()``, adapting it to your underlying model.

    Make sure to use ``mapToSourceRows()``/``mapFromSourceRows()``
    whenever fetching or manipulating table data, such as in ``data()``.

    When updating the model (inserting, removing rows), the sort order
    needs to be accounted for (e.g. reset and re-applied).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__sortInd = None     #: Indices sorting the source table
        self.__sortIndInv = None  #: The inverse of __sortInd
        self.__sortColumn = -1    #: Sort key column, or -1
        self.__sortOrder = Qt.AscendingOrder

    def sortColumnData(self, column):
        """Return raw, sortable data for column"""
        raise NotImplementedError

    def _sortColumnData(self, column):
        try:
            # Call the overridden implementation if available
            data = numpy.asarray(self.sortColumnData(column))
        except NotImplementedError:
            # Fallback to slow implementation
            data = numpy.array([self.index(row, column).data()
                                for row in range(self.rowCount())])

        assert data.ndim == 1, 'Data should be 1-dimensional'
        data = data[self.mapToSourceRows(Ellipsis)]
        return data

    def sortColumn(self):
        """The column currently used for sorting (-1 if no sorting is applied)"""
        return self.__sortColumn

    def sortOrder(self):
        """The current sort order"""
        return self.__sortOrder

    def mapToSourceRows(self, rows):
        """Return array of row indices in the source table for given model rows

        Parameters
        ----------
        rows : int or list of int or numpy.ndarray of dtype=int or Ellipsis
            View (sorted) rows.

        Returns
        -------
        numpy.ndarray
            Source rows matching input rows. If they are the same,
            simply input `rows` is returned.
        """
        # self.__sortInd[rows] fails if `rows` is an empty list or array
        if self.__sortInd is not None \
                and (isinstance(rows, (Integral, type(Ellipsis)))
                     or len(rows)):
            new_rows = self.__sortInd[rows]
            if rows is Ellipsis:
                new_rows.setflags(write=False)
            rows = new_rows
        return rows

    def mapFromSourceRows(self, rows):
        """Return array of row indices in the model for given source table rows

        Parameters
        ----------
        rows : int or list of int or numpy.ndarray of dtype=int or Ellipsis
            Source model rows.

        Returns
        -------
        numpy.ndarray
            ModelIndex (sorted) rows matching input source rows.
            If they are the same, simply input `rows` is returned.
        """
        # self.__sortInd[rows] fails if `rows` is an empty list or array
        if self.__sortIndInv is not None \
                and (isinstance(rows, (Integral, type(Ellipsis)))
                     or len(rows)):
            new_rows = self.__sortIndInv[rows]
            if rows is Ellipsis:
                new_rows.setflags(write=False)
            rows = new_rows
        return rows

    @deprecated('Orange.widgets.utils.itemmodels.AbstractSortTableModel.mapFromSourceRows')
    def mapFromTableRows(self, rows):
        return self.mapFromSourceRows(rows)

    @deprecated('Orange.widgets.utils.itemmodels.AbstractSortTableModel.mapToSourceRows')
    def mapToTableRows(self, rows):
        return self.mapToSourceRows(rows)

    def resetSorting(self):
        """Invalidates the current sorting"""
        return self.sort(-1)

    def _argsortData(self, data: numpy.ndarray, order):
        """
        Return indices of sorted data. May be reimplemented to handle
        sorting in a certain way, e.g. to sort NaN values last.
        """
        indices = numpy.argsort(data, kind="mergesort")
        if order == Qt.DescendingOrder:
            indices = indices[::-1]
        return indices

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        """
        Sort the data by `column` into `order`.

        To reset the order, pass column=-1.

        Reimplemented from QAbstractItemModel.sort().

        Notes
        -----
        This only affects the model's data presentation. The underlying
        data table is left unmodified. Use mapToSourceRows()/mapFromSourceRows()
        when accessing data by row indexes.
        """
        if QT_VERSION >= 0x50000:
            self.layoutAboutToBeChanged.emit(
                [], QAbstractTableModel.VerticalSortHint
            )
        else:
            self.layoutAboutToBeChanged.emit()

        # Store persistent indices as well as their (actual) rows in the
        # source data table.
        persistent = self.persistentIndexList()
        persistent_rows = self.mapToSourceRows([i.row() for i in persistent])

        self.__sortColumn = -1 if column < 0 else column
        self.__sortOrder = order

        indices = None
        if column >= 0:
            data = numpy.asarray(self._sortColumnData(column))
            if data is None:
                data = numpy.arange(self.rowCount())
            elif data.dtype == object:
                data = data.astype(str)

            indices = self.mapToSourceRows(self._argsortData(data, order))

        if indices is not None:
            self.__sortInd = indices
            self.__sortIndInv = numpy.argsort(indices)
        else:
            self.__sortInd = None
            self.__sortIndInv = None

        persistent_rows = self.mapFromSourceRows(persistent_rows)

        self.changePersistentIndexList(
            persistent,
            [self.index(row, pind.column())
             for row, pind in zip(persistent_rows, persistent)])
        if QT_VERSION >= 0x50000:
            self.layoutChanged.emit([], QAbstractTableModel.VerticalSortHint)
        else:
            self.layoutChanged.emit()


class PyTableModel(AbstractSortTableModel):
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
        row = self.mapFromSourceRows(index.row())
        if role == Qt.EditRole:
            self[row][index.column()] = value
            self.dataChanged.emit(index, index)
        else:
            self._roleData[row][index.column()][role] = value
        return True

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return

        row, column = self.mapToSourceRows(index.row()), index.column()

        role_value = self._roleData.get(row, {}).get(column, {}).get(role)
        if role_value is not None:
            return role_value

        try:
            value = self[row][column]
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

    def sortColumnData(self, column):
        return [row[column] for row in self._table]

    def setHorizontalHeaderLabels(self, labels):
        """
        Parameters
        ----------
        labels : list of str or list of Variable
        """
        self._headers[Qt.Horizontal] = tuple(labels)

    def setVerticalHeaderLabels(self, labels):
        """
        Parameters
        ----------
        labels : list of str or list of Variable
        """
        self._headers[Qt.Vertical] = tuple(labels)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        headers = self._headers.get(orientation)

        if headers and section < len(headers):
            section = self.mapToSourceRows(section) if orientation == Qt.Vertical else section
            value = headers[section]

            if role == Qt.ToolTipRole:
                role = Qt.DisplayRole

            if role == Qt.DisplayRole:
                return value.name if isinstance(value, Variable) else value

            if role == Qt.DecorationRole:
                if isinstance(value, Variable):
                    return gui.attributeIconDict[value]

        # Use QAbstractItemModel default for non-existent header/sections
        return super().headerData(section, orientation, role)

    def removeRows(self, row, count, parent=QModelIndex()):
        if not parent.isValid():
            del self[row:row + count]
            for rowidx in range(row, row + count):
                self._roleData.pop(rowidx, None)
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
        return True

    def insertRows(self, row, count, parent=QModelIndex()):
        self.beginInsertRows(parent, row, row + count - 1)
        self._table[row:row] = [[''] * self.columnCount() for _ in range(count)]
        self.endInsertRows()
        return True

    def insertColumns(self, column, count, parent=QModelIndex()):
        self.beginInsertColumns(parent, column, column + count - 1)
        for row in self._table:
            row[column:column] = [''] * count
        self.endInsertColumns()
        return True

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
        self._check_sort_order()
        self.beginRemoveRows(QModelIndex(), start, stop)
        del self._table[i]
        self.endRemoveRows()

    def __setitem__(self, i, value):
        if isinstance(i, slice):
            start, stop, _ = _as_contiguous_range(i, len(self))
            stop -= 1
        else:
            start = stop = i = i if i >= 0 else len(self) + i
        self._check_sort_order()
        self._table[i] = value
        self.dataChanged.emit(self.index(start, 0),
                              self.index(stop, self.columnCount() - 1))

    def _check_sort_order(self):
        if self.mapToSourceRows(Ellipsis) is not Ellipsis:
            warn("Can't modify PyTableModel when it's sorted",
                 RuntimeWarning, stacklevel=3)
            raise RuntimeError("Can't modify PyTableModel when it's sorted")

    def wrap(self, table):
        self.beginResetModel()
        self._table = table
        self._roleData = self._RoleData()
        self.resetSorting()
        self.endResetModel()

    def tolist(self):
        return self._table

    def clear(self):
        self.beginResetModel()
        self._table.clear()
        self.resetSorting()
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
    MIME_TYPE = "application/x-Orange-PyListModelData"
    Separator = object()

    def __init__(self, iterable=None, parent=None,
                 flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled,
                 list_item_role=Qt.DisplayRole,
                 enable_dnd=False,
                 supportedDropActions=Qt.MoveAction):
        super().__init__(parent)
        self._list = []
        self._other_data = []
        if enable_dnd:
            flags |= Qt.ItemIsDragEnabled
        self._flags = flags
        self.list_item_role = list_item_role

        self._supportedDropActions = supportedDropActions
        if iterable is not None:
            self.extend(iterable)

    def _is_index_valid(self, index):
        # This error would happen if one wraps a list into a model and then
        # modifies a list instead of a model
        if len(self) != len(self._other_data):
            raise RuntimeError("Mismatched length of model and its _other_data")
        if isinstance(index, QModelIndex) and index.isValid():
            row, column = index.row(), index.column()
            return 0 <= row < len(self) and column == 0
        elif isinstance(index, int):
            return -len(self) <= index < len(self)
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
                and self._is_index_valid(index):
            return self[row]
        elif self._is_index_valid(row):
            return self._other_data[row].get(role, None)

    def itemData(self, index):
        mapping = QAbstractListModel.itemData(self, index)
        if self._is_index_valid(index):
            items = list(self._other_data[index.row()].items())
        else:
            items = []
        for key, value in items:
            mapping[key] = value
        return mapping

    def parent(self, index=QModelIndex()):
        return QModelIndex()

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            if self._is_index_valid(index):
                self[index.row()] = value  # Will emit proper dataChanged signal
                return True
        elif self._is_index_valid(index):
            self._other_data[index.row()][role] = value
            self.dataChanged.emit(index, index)
            return True
        return False

    def setItemData(self, index, data):
        data = dict(data)
        with signal_blocking(self):
            for role, value in data.items():
                if role == Qt.EditRole and \
                        self._is_index_valid(index):
                    self[index.row()] = value
                elif self._is_index_valid(index):
                    self._other_data[index.row()][role] = value

        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        if self._is_index_valid(index):
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

    def __contains__(self, value):
        return value in self._list

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __add__(self, iterable):
        new_list = PyListModel(list(self._list),
                               # method parent is overloaded in Model
                               QObject.parent(self),
                               flags=self._flags,
                               list_item_role=self.list_item_role,
                               supportedDropActions=self.supportedDropActions())
        # pylint: disable=protected-access
        new_list._other_data = list(self._other_data)
        new_list.extend(iterable)
        return new_list

    def __iadd__(self, iterable):
        self.extend(iterable)
        return self

    def __delitem__(self, s):
        if isinstance(s, slice):
            start, stop, _ = _as_contiguous_range(s, len(self))
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
        for i, (new_l, new_o) in enumerate(zip(lst, other)):
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
        return [self.MIME_TYPE] + list(QAbstractListModel.mimeTypes(self))

    def mimeData(self, indexlist):
        if len(indexlist) <= 0:
            return None

        items = [self[i.row()] for i in indexlist]
        itemdata = [self.itemData(i) for i in indexlist]
        mime = QAbstractListModel.mimeData(self, indexlist)
        mime.setData(self.MIME_TYPE, b'see properties: _items, _itemdata')
        mime.setProperty('_items', items)
        mime.setProperty('_itemdata', itemdata)
        return mime

    def dropMimeData(self, mime, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True

        if not mime.hasFormat(self.MIME_TYPE):
            return False

        items = mime.property('_items')
        itemdata = mime.property('_itemdata')

        if not items:
            return False

        if row == -1:
            row = len(self)

        self[row:row] = items
        for i, data in enumerate(itemdata):
            self.setItemData(self.index(row + i), data)
        return True


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

    def __init__(self, *args, placeholder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.placeholder = placeholder

    def data(self, index, role=Qt.DisplayRole):
        if self._is_index_valid(index):
            var = self[index.row()]
            if var is None and role == Qt.DisplayRole:
                return self.placeholder or "None"
            if not isinstance(var, Variable):
                return super().data(index, role)
            elif role == Qt.DisplayRole:
                return var.name
            elif role == Qt.DecorationRole:
                return gui.attributeIconDict[var]
            elif role == Qt.ToolTipRole:
                return self.variable_tooltip(var)
            elif role == gui.TableVariable:
                return var
            else:
                return PyListModel.data(self, index, role)

    def variable_tooltip(self, var):
        if var.is_discrete:
            return self.discrete_variable_tooltip(var)
        elif var.is_time:
            return self.time_variable_toltip(var)
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
        text = "<b>%s</b><br/>Categorical with %i values: " %\
               (safe_text(var.name), len(var.values))
        text += ", ".join("%r" % safe_text(v) for v in var.values)
        text += self.variable_labels_tooltip(var)
        return text

    def time_variable_toltip(self, var):
        text = "<b>%s</b><br/>Time" % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

    def continuous_variable_toltip(self, var):
        text = "<b>%s</b><br/>Numeric" % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

    def string_variable_tooltip(self, var):
        text = "<b>%s</b><br/>Text" % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text


class DomainModel(VariableListModel):
    ATTRIBUTES, CLASSES, METAS = 1, 2, 4
    MIXED = ATTRIBUTES | CLASSES | METAS
    SEPARATED = (CLASSES, PyListModel.Separator,
                 METAS, PyListModel.Separator,
                 ATTRIBUTES)
    PRIMITIVE = (DiscreteVariable, ContinuousVariable)

    def __init__(self, order=SEPARATED, separators=True, placeholder=None,
                 valid_types=None, alphabetical=False, skip_hidden_vars=True, **kwargs):
        """

        Parameters
        ----------
        order: tuple or int
            Order of attributes, metas, classes, separators and other options
        separators: bool
            If False, remove separators from `order`.
        placeholder: str
            The text that is shown when no variable is selected
        valid_types: tuple
            (Sub)types of `Variable` that are included in the model
        alphabetical: bool
            If true, variables are sorted alphabetically.
        skip_hidden_vars: bool
            If true, variables marked as "hidden" are skipped.
        """
        super().__init__(placeholder=placeholder, **kwargs)
        if isinstance(order, int):
            order = (order,)
        if placeholder is not None and None not in order:
            # Add None for the placeholder if it's not already there
            # Include separator if the current order uses them
            order = (None,) + \
                    (self.Separator, ) * (self.Separator in order) + \
                    order
        if not separators:
            order = [e for e in order if e is not self.Separator]
        self.order = order
        self.valid_types = valid_types
        self.alphabetical = alphabetical
        self.skip_hidden_vars = skip_hidden_vars
        self._within_set_domain = False
        self.set_domain(None)

    def set_domain(self, domain):
        self.beginResetModel()
        content = []
        # The logic related to separators is a bit complicated: it ensures that
        # even when a section is empty we don't have two separators in a row
        # or a separator at the end
        add_separator = False
        for section in self.order:
            if section is self.Separator:
                add_separator = True
                continue
            if isinstance(section, int):
                if domain is None:
                    continue
                to_add = list(chain(
                    *(vars for i, vars in enumerate(
                        (domain.attributes, domain.class_vars, domain.metas))
                      if (1 << i) & section)))
                if self.skip_hidden_vars:
                    to_add = list(filter_visible(to_add))
                if self.valid_types is not None:
                    to_add = [var for var in to_add
                              if isinstance(var, self.valid_types)]
                if self.alphabetical:
                    to_add = sorted(to_add, key=lambda x: x.name)
            elif isinstance(section, list):
                to_add = section
            else:
                to_add = [section]
            if to_add:
                if add_separator and content:
                    content.append(self.Separator)
                    add_separator = False
                content += to_add
        try:
            self._within_set_domain = True
            self[:] = content
        finally:
            self._within_set_domain = False
        self.endResetModel()

    def prevent_modification(method):  # pylint: disable=no-self-argument
        @wraps(method)
        # pylint: disable=protected-access
        def e(self, *args, **kwargs):
            if self._within_set_domain:
                method(self, *args, **kwargs)
            else:
                raise TypeError(
                    "{} can be modified only by calling 'set_domain'".
                    format(type(self).__name__))
        return e

    @prevent_modification
    def extend(self, iterable):
        return super().extend(iterable)

    @prevent_modification
    def append(self, item):
        return super().append(item)

    @prevent_modification
    def insert(self, i, val):
        return super().insert(i, val)

    @prevent_modification
    def remove(self, val):
        return super().remove(val)

    @prevent_modification
    def pop(self, i):
        return super().pop(i)

    @prevent_modification
    def clear(self):
        return super().clear()

    @prevent_modification
    def __delitem__(self, s):
        return super().__delitem__(s)

    @prevent_modification
    def __setitem__(self, s, value):
        return super().__setitem__(s, value)

    @prevent_modification
    def reverse(self):
        return super().reverse()

    @prevent_modification
    def sort(self, *args, **kwargs):
        return super().sort(*args, **kwargs)


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


class TableModel(AbstractSortTableModel):
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

    # The class uses the same names (X_density etc) as Table
    # pylint: disable=invalid-name
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
                getter = operator.attrgetter("sparse_metas")
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

        if self.Y_density != Storage.DENSE and domain.class_vars:
            coldesc = make_basket(domain.class_vars, self.Y_density,
                                  TableModel.ClassVar)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.ClassVar)
                        for var in domain.class_vars]

        if self.M_density != Storage.DENSE and domain.metas:
            coldesc = make_basket(domain.metas, self.M_density,
                                  TableModel.Meta)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.Meta)
                        for var in domain.metas]

        if self.X_density != Storage.DENSE and domain.attributes:
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

    def sortColumnData(self, column):
        return self._columnSortKeyData(column, TableModel.ValueRole)

    @deprecated('Orange.widgets.utils.itemmodels.TableModel.sortColumnData')
    def columnSortKeyData(self, column, role):
        return self._columnSortKeyData(column, role)

    def _columnSortKeyData(self, column, role):
        """
        Return a sequence of source table objects which can be used as
        `keys` for sorting.

        :param int column: Sort column.
        :param Qt.ItemRole role: Sort item role.

        """
        coldesc = self.columns[column]
        if isinstance(coldesc, TableModel.Column) \
                and role == TableModel.ValueRole:
            col_data = numpy.asarray(self.source.get_column_view(coldesc.var)[0])

            if coldesc.var.is_continuous:
                # continuous from metas have dtype object; cast it to float
                col_data = col_data.astype(float)
            return col_data
        else:
            return numpy.asarray([self.index(i, column).data(role)
                                  for i in range(self.rowCount())])

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
             _recognizedRoles=frozenset([Qt.DisplayRole,
                                         Qt.EditRole,
                                         Qt.BackgroundRole,
                                         ValueRole,
                                         ClassValueRole,
                                         VariableRole,
                                         DomainRole,
                                         VariableStatsRole])):
        """
        Reimplemented from `QAbstractItemModel.data`
        """
        if role not in _recognizedRoles:
            return None

        row, col = index.row(), index.column()
        if  not 0 <= row <= self.__rowCount:
            return None

        row = self.mapToSourceRows(row)

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
        row = self.mapFromSourceRows(index.row())
        if role == Qt.EditRole:
            try:
                self.source[row, index.column()] = value
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
                return int(self.mapToSourceRows(section) + 1)
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
