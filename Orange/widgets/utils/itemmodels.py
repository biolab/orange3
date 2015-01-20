import pickle
import operator

from contextlib import contextmanager
from functools import reduce, lru_cache
from xml.sax.saxutils import escape

from PyQt4 import QtGui
from PyQt4.QtGui import  QItemSelectionModel
from PyQt4.QtCore import (
    Qt, QAbstractListModel, QAbstractTableModel, QModelIndex, QByteArray
)
from PyQt4.QtCore import pyqtSignal as Signal

from PyQt4.QtGui import (
    QWidget, QBoxLayout, QToolButton, QAbstractButton, QAction
)

from Orange.data import (
    Variable, DiscreteVariable, ContinuousVariable, StringVariable
)
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


def _as_contiguous_range(start, stop, step):
    if step == -1:
        # Equivalent range with positive step
        start, stop = stop + 1, start + 1
    elif not (step == 1 or step is None):
        raise IndexError("Non-contiguous range.")
    return start, stop, step


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
            start, stop, step = s.indices(len(self))
            start, stop, step = _as_contiguous_range(start, stop, step)
            self.beginRemoveRows(QModelIndex(), start, stop - 1)
        else:
            s = len(self) + s if s < 0 else s
            self.beginRemoveRows(QModelIndex(), s, s)
        del self._list[s]
        del self._other_data[s]
        self.endRemoveRows()

    def __setitem__(self, s, value):
        if isinstance(s, slice):
            start, stop, step = s.indices(len(self))
            start, stop, step = _as_contiguous_range(start, stop, step)
            self.__delitem__(slice(start, stop, step))

            if not isinstance(value, list):
                value = list(value)
            self.beginInsertRows(QModelIndex(), start, start + len(value) - 1)
            self._list[s] = value
            self._other_data[s] = (_store() for _ in value)
            self.endInsertRows()
        else:
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


from . import datacaching
from Orange.statistics import basic_stats


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
    #: One of `Attribute, ClassVar, Meta
    DomainRole = next(gui.OrangeUserRole)

    #: Column domain roles
    Attribute, ClassVar, Meta = range(3)

    def __init__(self, sourcedata, parent=None):
        super().__init__(parent)
        self.source = sourcedata
        #: source table domain
        self.domain = domain = sourcedata.domain
        #: list of all domain variables (attrs + class_vars + metas)
        self.vars = domain.attributes + domain.class_vars + domain.metas

        self.cls_color = QtGui.QColor(160, 160, 160)
        self.meta_color = QtGui.QColor(220, 220, 200)

        # domain roles for all table columns
        self._column_roles = \
            (([TableModel.Attribute] * len(domain.attributes)) +
             ([TableModel.ClassVar] * len(domain.class_vars)) +
             ([TableModel.Meta] * len(domain.metas)))

        role_to_color = {TableModel.Attribute: None,
                         TableModel.ClassVar: self.cls_color,
                         TableModel.Meta: self.meta_color}

        self._background = [role_to_color[r] for r in self._column_roles]

        self._labels = sorted(
            reduce(operator.ior,
                   [set(attr.attributes) for attr in self.vars],
                   set()))

        self._extra_data = {}

        @lru_cache(maxsize=1000)
        def row_instance(index):
            return self.source[index]

        self._row_instance = row_instance
        # column basic statistics (VariableStatsRole), computed when
        # first needed.
        self._stats = None
        self.__rowCount = len(sourcedata)
        self.__columnCount = len(self.vars)

    def data(self, index, role,
             # For optimizing out LOAD_GLOBAL byte code instructions in
             # the item role tests (goes from 14 us to 9 us average).
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

        row, col = index.row(), index.column()
        if role not in _recognizedRoles:
            return self._extra_data.get((row, col, role), None)

        instance = self._row_instance(row)
        var = self.vars[col]

        if role == _Qt_DisplayRole:
            return _str(instance[var])
        elif role == _Qt_EditRole:
            return instance[var]
        elif role == _Qt_BackgroundRole:
            return self._background[col]
        elif role == _ValueRole:
            return instance[var]
        elif role == _ClassValueRole:
            try:
                return instance.get_class()
            except TypeError:
                return None
        elif role == _VariableRole:
            return var
        elif role == _DomainRole:
            return self._column_roles[col]
        elif role == _VariableStatsRole:
            return self._stats_for_column(col)
        else:
            return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            try:
                self.source[index.row(), index.column()] = value
            except (TypeError, IndexError):
                return False
        else:
            self._other_data[index.row(), index.column(), role] = value

        self.dataChanged.emit(index, index)
        return True

    def parent(self, index):
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.__rowCount

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.__columnCount

    def headerData(self, section, orientation, role):
        if orientation == Qt.Vertical:
            return section + 1 if role == Qt.DisplayRole else None

        var = self.vars[section]
        if role == Qt.DisplayRole:
            return var.name
        elif role == Qt.ToolTipRole:
            return self._tooltip(var, self._labels)
        elif role == TableModel.VariableRole:
            return var
        elif role == TableModel.VariableStatsRole:
            return self._stats_for_column(section)
        else:
            return None

    def _tooltip(self, variable, labels=None):
        """
        Return an header tool tip text for an `Orange.data.Variable` instance.
        """

        if labels is None:
            labels = variable.attributes.keys()

        pairs = [(escape(key), escape(str(variable.attributes[key])))
                 for key in labels if key in variable.attributes]
        tip = "<b>%s</b>" % escape(variable.name)
        tip = "<br/>".join([tip] + ["%s = %s" % pair for pair in pairs])
        return tip

    def _stats_for_column(self, column):
        if self._stats is None:
            self._stats = datacaching.getCached(
                self.source, basic_stats.DomainBasicStats,
                (self.source, True)
            )
        return self._stats[column]

from collections import namedtuple
from Orange.data import Storage


class SparseTableModel(TableModel):

    Basket = namedtuple("Basket", ["vars"])

    def __init__(self, sourcedata, parent=None):
        super().__init__(sourcedata)

        self.X_density = sourcedata.X_density()
        self.Y_density = sourcedata.Y_density()
        self.M_density = sourcedata.metas_density()

        domain = sourcedata.domain

        vars = []
        background = []
        columnroles = []

        if self.X_density != Storage.DENSE:
            vars += [SparseTableModel.Basket(domain.attributes)]
            background += [None]
            columnroles += [TableModel.Attribute]
        else:
            vars += domain.attributes
            background += [None] * len(domain.attributes)
            columnroles += [TableModel.Attribute] * len(domain.attributes)

        if self.Y_density != Storage.DENSE:
            vars += [SparseTableModel.Basket(domain.class_vars)]
            background += [self.cls_color]
            columnroles += [TableModel.ClassVar]
        else:
            vars += domain.class_vars
            background += [self.cls_color] * len(domain.class_vars)
            columnroles += [TableModel.ClassVar] * len(domain.class_vars)

        if self.M_density != Storage.DENSE:
            vars += [SparseTableModel.Basket(domain.metas)]
            background += [self.meta_color]
            columnroles += [TableModel.Meta]
        else:
            vars += domain.metas
            background += [self.meta_color] * len(domain.metas)
            columnroles += [TableModel.Meta] * len(domain.metas)

        self.vars = vars

        self._background = background
        self._column_roles = columnroles
        self.__columnCount = len(self.vars)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.__columnCount

    def headerData(self, section, orientation, role):
        if orientation == Qt.Vertical:
            return super().headerData(section, orientation, role)
        var = self.vars[section]

        if not isinstance(var, SparseTableModel.Basket):
            return super().headerData(section, orientation, role)

        if role == Qt.DisplayRole:
            if isinstance(var, SparseTableModel.Basket):
                return "{...}"
            else:
                return var.name
        elif role == TableModel.VariableRole:
            return None
        else:
            return None

    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()

        var = self.vars[col]

        if not isinstance(var, SparseTableModel.Basket):
            return super().data(index, role)

        instance = self._row_instance(row)

        if role == Qt.DisplayRole:
            colrole = self._column_roles[col]
            if colrole == TableModel.Attribute:
                data = instance.sparse_x
                density = self.X_density
            elif colrole == TableModel.ClassVar:
                data = instance.sparse_y
                density = self.Y_density
            else:
                data = instance.sparse_meta
                density = self.M_density

            if density == Storage.SPARSE:
                return ", ".join(
                    "{}={}".format(var.vars[i].name, var.vars[i].repr_val(v))
                    for i, v in zip(data.indices, data.data))
            else:
                return ", ".join(var.vars[i].name for i in data.indices)

        elif role == Qt.EditRole:
            return None
        elif role == Qt.BackgroundRole:
            return self._background[col]
        elif role == TableModel.ValueRole:
            return None
        elif role == TableModel.ClassValueRole:
            try:
                return instance.get_class()
            except TypeError:
                return None
        elif role == TableModel.VariableRole:
            return None
        elif role == TableModel.DomainRole:
            return self._column_roles[col]
        else:
            return None

    def _stats_for_column(self, column):
        if self._stats is None:
            self._stats = datacaching.getCached(
                self.source, basic_stats.DomainBasicStats,
                (self.source, True)
            )
        var = self.vars[column]
        return self._stats[var]
