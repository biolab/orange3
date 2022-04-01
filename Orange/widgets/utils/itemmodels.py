from numbers import Number, Integral
from math import isnan, isinf

import operator
from collections import namedtuple, defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from functools import reduce, partial, lru_cache, wraps
from itertools import chain
from warnings import warn
from xml.sax.saxutils import escape

from AnyQt.QtCore import (
    Qt, QObject, QAbstractListModel, QModelIndex,
    QItemSelectionModel, QItemSelection)
from AnyQt.QtCore import pyqtSignal as Signal
from AnyQt.QtGui import QColor, QBrush
from AnyQt.QtWidgets import (
    QWidget, QBoxLayout, QToolButton, QAbstractButton, QAction
)

import numpy

from orangewidget.utils.itemmodels import (
    PyListModel, AbstractSortTableModel as _AbstractSortTableModel
)

from Orange.widgets.utils.colorpalettes import ContinuousPalettes, ContinuousPalette
from Orange.data import Variable, Storage, DiscreteVariable, ContinuousVariable
from Orange.data.domain import filter_visible
from Orange.widgets import gui
from Orange.widgets.utils import datacaching
from Orange.statistics import basic_stats
from Orange.util import deprecated

__all__ = [
    "PyListModel", "VariableListModel", "PyListModelTooltip", "DomainModel",
    "AbstractSortTableModel", "PyTableModel", "TableModel",
    "ModelActionsWidget", "ListSingleSelectionModel"
]

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


class AbstractSortTableModel(_AbstractSortTableModel):
    # these were defined on TableModel below. When the AbstractSortTableModel
    # was extracted and made a base of TableModel these deprecations were
    # misplaced, they belong in TableModel.
    @deprecated('Orange.widgets.utils.itemmodels.AbstractSortTableModel.mapFromSourceRows')
    def mapFromTableRows(self, rows):
        return self.mapFromSourceRows(rows)

    @deprecated('Orange.widgets.utils.itemmodels.AbstractSortTableModel.mapToSourceRows')
    def mapToTableRows(self, rows):
        return self.mapToSourceRows(rows)


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
        self._rows = self._cols = 0
        self._headers = {}
        self._editable = editable
        self._table = None
        self._roleData = {}
        if sequence is None:
            sequence = []
        self.wrap(sequence)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._rows

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._cols

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
            self._rows = self._table_dim()[0]
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
        self._cols = self._table_dim()[1]
        self.endRemoveColumns()
        return True

    def _table_dim(self):
        return len(self._table), max(map(len, self), default=0)

    def insertRows(self, row, count, parent=QModelIndex()):
        self.beginInsertRows(parent, row, row + count - 1)
        self._table[row:row] = [[''] * self.columnCount() for _ in range(count)]
        self._rows = self._table_dim()[0]
        self.endInsertRows()
        return True

    def insertColumns(self, column, count, parent=QModelIndex()):
        self.beginInsertColumns(parent, column, column + count - 1)
        for row in self._table:
            row[column:column] = [''] * count
        self._rows = self._table_dim()[0]
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
        if stop < start:
            return
        self._check_sort_order()
        self.beginRemoveRows(QModelIndex(), start, stop)
        del self._table[i]
        rows = self._table_dim()[0]
        self._rows = rows
        self.endRemoveRows()
        self._update_column_count()

    def __setitem__(self, i, value):
        self._check_sort_order()
        if isinstance(i, slice):
            start, stop, _ = _as_contiguous_range(i, len(self))
            self.removeRows(start, stop - start)
            if len(value) == 0:
                return
            self.beginInsertRows(QModelIndex(), start, start + len(value) - 1)
            self._table[start:start] = value
            self._rows = self._table_dim()[0]
            self.endInsertRows()
            self._update_column_count()
        else:
            self._table[i] = value
            self.dataChanged.emit(self.index(i, 0),
                                  self.index(i, self.columnCount() - 1))

    def _update_column_count(self):
        cols_before = self._cols
        cols_after = self._table_dim()[1]
        if cols_before < cols_after:
            self.beginInsertColumns(QModelIndex(), cols_before, cols_after - 1)
            self._cols = cols_after
            self.endInsertColumns()
        elif cols_before > cols_after:
            self.beginRemoveColumns(QModelIndex(), cols_after, cols_before - 1)
            self._cols = cols_after
            self.endRemoveColumns()

    def _check_sort_order(self):
        if self.mapToSourceRows(Ellipsis) is not Ellipsis:
            warn("Can't modify PyTableModel when it's sorted",
                 RuntimeWarning, stacklevel=3)
            raise RuntimeError("Can't modify PyTableModel when it's sorted")

    def wrap(self, table):
        self.beginResetModel()
        self._table = table
        self._roleData = self._RoleData()
        self._rows, self._cols = self._table_dim()
        self.resetSorting()
        self.endResetModel()

    def tolist(self):
        return self._table

    def clear(self):
        self.beginResetModel()
        self._table.clear()
        self.resetSorting()
        self._roleData.clear()
        self._rows, self._cols = self._table_dim()
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


class PyListModelTooltip(PyListModel):
    def __init__(self, iterable=None, tooltips=(), **kwargs):
        super().__init__(iterable, **kwargs)
        if not isinstance(tooltips, Sequence):
            # may be a generator; if not, fail
            tooltips = list(tooltips)
        self.tooltips = tooltips

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.ToolTipRole:
            if index.row() >= len(self.tooltips):
                return None
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

    def setData(self, index, value, role=Qt.EditRole):
        # reimplemented
        if role == Qt.EditRole:
            return False
        else:
            return super().setData(index, value, role)

    def setItemData(self, index, data):
        # reimplemented
        if Qt.EditRole in data:
            return False
        else:
            return super().setItemData(index, data)

    def insertRows(self, row, count, parent=QModelIndex()):
        # reimplemented
        return False

    def removeRows(self, row, count, parent=QModelIndex()):
        # reimplemented
        return False

_html_replace = [("<", "&lt;"), (">", "&gt;")]


def safe_text(text):
    for old, new in _html_replace:
        text = str(text).replace(old, new)
    return text


class ContinuousPalettesModel(QAbstractListModel):
    """
    Model for combo boxes
    """
    KeyRole = Qt.UserRole + 1
    def __init__(self, parent=None, categories=None, icon_width=64):
        super().__init__(parent)
        self.icon_width = icon_width

        palettes = list(ContinuousPalettes.values())
        if categories is None:
            # Use dict, not set, to keep order of categories
            categories = dict.fromkeys(palette.category for palette in palettes)

        self.items = []
        for category in categories:
            self.items.append(category)
            self.items += [palette for palette in palettes
                           if palette.category == category]
        if len(categories) == 1:
            del self.items[0]

    def rowCount(self, parent):
        return 0 if parent.isValid() else len(self.items)

    @staticmethod
    def columnCount(parent):
        return 0 if parent.isValid() else 1

    def data(self, index, role):
        item = self.items[index.row()]
        if isinstance(item, str):
            if role in [Qt.EditRole, Qt.DisplayRole]:
                return item
        else:
            if role in [Qt.EditRole, Qt.DisplayRole]:
                return item.friendly_name
            if role == Qt.DecorationRole:
                return item.color_strip(self.icon_width, 16)
            if role == Qt.UserRole:
                return item
            if role == self.KeyRole:
                return item.name
        return None

    def flags(self, index):
        item = self.items[index.row()]
        if isinstance(item, ContinuousPalette):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        else:
            return Qt.NoItemFlags

    def indexOf(self, x):
        if isinstance(x, str):
            for i, item in enumerate(self.items):
                if not isinstance(item, str) \
                        and x in (item.name, item.friendly_name):
                    return i
        elif isinstance(x, ContinuousPalette):
            return self.items.index(x)
        return None


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


def select_rows(view, row_indices, command=QItemSelectionModel.ClearAndSelect):
    """
    Select several rows in view.

    :param QAbstractItemView view:
    :param row_indices: Integer indices of rows to select.
    :param command: QItemSelectionModel.SelectionFlags
    """
    selmodel = view.selectionModel()
    model = view.model()
    selection = QItemSelection()
    for row in row_indices:
        index = model.index(row, 0)
        selection.select(index, index)
    selmodel.select(selection, command | QItemSelectionModel.Rows)


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

        brush_for_role = {
            role: QBrush(c) if c is not None else None
            for role, c in self.ColorForRole.items()
        }

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
                vars, TableModel.Attribute, brush_for_role[role], density,
                make_basket_formater(vars, density, role)
            )

        def make_column(var, role):
            return TableModel.Column(
                var, role, brush_for_role[role],
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
