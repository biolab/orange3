# pylint: disable=missing-docstring
from typing import Iterable, Set
from collections import defaultdict
from itertools import product, chain

import numpy as np
from scipy import sparse as sp

from AnyQt.QtCore import (Qt, QSize, QItemSelection, QItemSelectionModel,
                          pyqtSignal)
from AnyQt.QtGui import QStandardItem, QColor, QStandardItemModel
from AnyQt.QtWidgets import (QTableView, QSizePolicy, QHeaderView,
                             QStyledItemDelegate, QCheckBox, QFrame, QWidget,
                             QGridLayout)

from Orange.data import (Table, DiscreteVariable, Variable, Domain,
                         ContinuousVariable, TimeVariable)
from Orange.data.domain import filter_visible
from Orange.data.util import get_unique_names_duplicates, get_unique_names
from Orange.data.filter import FilterContinuous, FilterDiscrete, Values
from Orange.statistics.util import (nanmin, nanmax, nanunique, nansum, nanvar,
                                    nanmean, nanmedian, nanmode, bincount)
from Orange.util import Enum
from Orange.widgets import gui
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg


BorderRole = next(gui.OrangeUserRole)
BorderColorRole = next(gui.OrangeUserRole)


class AggregationFunctionsEnum(Enum):
    (Count, Count_defined, Sum, Mean, Min, Max,
     Mode, Median, Var, Majority) = range(10)

    def __init__(self, *_, **__):
        super().__init__()
        self.func = None

    @property
    def value(self):
        return self._value_

    def __call__(self, *args):
        return self.func(args)  # pylint: disable=not-callable

    def __str__(self):
        return self._name_.replace("_", " ")

    def __gt__(self, other):
        return self._value_ > other.value


class Pivot:
    Functions = AggregationFunctionsEnum
    (Count, Count_defined, Sum, Mean, Min, Max,
     Mode, Median, Var, Majority) = Functions

    AutonomousFunctions = (Count,)
    AnyVarFunctions = (Count_defined,)
    ContVarFunctions = (Sum, Mean, Min, Max, Mode, Median, Var)
    DiscVarFunctions = (Majority,)
    TimeVarFunctions = (Mean, Min, Max, Mode, Median)
    FloatFunctions = (Count, Count_defined, Sum, Var)

    class Tables:
        table = None  # type: Table
        total_h = None  # type: Table
        total_v = None  # type: Table
        total = None  # type: Table

        def __call__(self):
            return self.table, self.total_h, self.total_v, self.total

    def __init__(self, table: Table, agg_funs: Iterable[Functions],
                 row_var: Variable, col_var: Variable = None,
                 val_var: Variable = None):
        self._group_tables = self.Tables()
        self._pivot_tables = self.Tables()
        self._table = table
        self._row_var = row_var
        self._col_var = col_var if col_var else row_var
        self.renamed = []

        if not table:
            return
        if not self._row_var.is_primitive():
            raise TypeError("Row variable should be DiscreteVariable"
                            " or ContinuousVariable")
        if self._col_var and not self._col_var.is_discrete:
            raise TypeError("Column variable should be DiscreteVariable")

        self._row_var_col = table.get_column_view(row_var)[0].astype(float)
        self._col_var_col = table.get_column_view(self._col_var)[0].astype(float)
        self._row_var_groups = nanunique(self._row_var_col)
        self._col_var_groups = nanunique(self._col_var_col)

        self._total_var = DiscreteVariable("Total", values=("total", ))
        self._current_agg_functions = sorted(agg_funs)
        self._indepen_agg_done = {}  # type: Dict[Functions, int]
        self._depen_agg_done = {}  # type: Dict[Functions, Dict[Variable, int]]

        self._initialize(agg_funs, val_var)

    @property
    def group_table(self) -> Table:
        table = self._group_tables.table
        if not table or len(table) == 0:
            return None
        indices = [0, 1] if not self.single_var_grouping else [0]
        for f in self._current_agg_functions:
            if f in self._indepen_agg_done:
                indices.append(self._indepen_agg_done[f])
        for v in self._table.domain.variables + self._table.domain.metas:
            for f in self._current_agg_functions:
                if f in self._depen_agg_done and v in self._depen_agg_done[f]:
                    indices.append(self._depen_agg_done[f][v])
        return table[:, indices]

    @property
    def pivot_table(self) -> Table:
        return self._pivot_tables.table

    @property
    def pivot_total_h(self) -> Table:
        return self._pivot_tables.total_h

    @property
    def pivot_total_v(self) -> Table:
        return self._pivot_tables.total_v

    @property
    def pivot_total(self) -> Table:
        return self._pivot_tables.total

    @property
    def pivot_tables(self) -> Table:
        return self._pivot_tables()

    @property
    def single_var_grouping(self) -> bool:
        return self._row_var is self._col_var

    def update_group_table(self, agg_funs: Iterable[Functions],
                           val_var: Variable = None):
        if not self._group_tables:
            return
        self._current_agg_functions = sorted(agg_funs)
        agg_funs = set(self._indepen_agg_done.keys()) | \
            set(self._depen_agg_done.keys()) | set(agg_funs)
        self._initialize(sorted(agg_funs), val_var)

    def _initialize(self, agg_funs, val_var):
        var_indep_funs, var_dep_funs = self.__group_aggregations(agg_funs)
        self._create_group_tables(var_indep_funs, var_dep_funs)
        self.__reference_aggregations(var_indep_funs, var_dep_funs)
        self._create_pivot_tables(val_var)

    def __group_aggregations(self, agg_funs):
        auto_funcs = self.AutonomousFunctions
        var_indep_funs = [fun for fun in agg_funs if fun in auto_funcs]
        var_dep_funs = []
        attrs = self._table.domain.variables + self._table.domain.metas
        prod = product(filter_visible(attrs),
                       [fun for fun in agg_funs if fun not in auto_funcs])
        for var, fun in prod:
            if self.__include_aggregation(fun, var):
                var_dep_funs.append((var, fun))
        return var_indep_funs, var_dep_funs

    def __include_aggregation(self, fun, var):
        return fun in self.ContVarFunctions and var.is_continuous or \
               fun in self.DiscVarFunctions and var.is_discrete or \
               fun in self.AnyVarFunctions

    def __reference_aggregations(self, var_indep_funs, var_dep_funs):
        self._indepen_agg_done = {}
        self._depen_agg_done = defaultdict(dict)
        i = 1 - int(bool(self.single_var_grouping))
        for i, fun in enumerate(var_indep_funs, i + 1):
            self._indepen_agg_done[fun] = i
        for j, (var, fun) in enumerate(var_dep_funs, i + 1):
            self._depen_agg_done[fun].update({var: j})

    def _create_group_tables(self, var_indep_funs, var_dep_funs):
        attrs = [ContinuousVariable(f"({str(fun).lower()})")
                 for fun in var_indep_funs]
        for var, fun in var_dep_funs:
            name = f"{var.name} ({str(fun).lower()})"
            if fun in self.DiscVarFunctions:
                attrs.append(DiscreteVariable(name, var.values))
            else:
                if isinstance(var, TimeVariable) and \
                        fun in self.TimeVarFunctions:
                    attrs.append(TimeVariable(name, have_date=var.have_date,
                                              have_time=var.have_time))
                else:
                    attrs.append(ContinuousVariable(name))
        args = (var_indep_funs, var_dep_funs, attrs)
        for t, var in (("table", None), ("total_h", self._col_var),
                       ("total_v", self._row_var), ("total", self._total_var)):
            setattr(self._group_tables, t, self.__get_group_table(var, *args))

    def __get_group_table(self, var, var_indep_funs, var_dep_funs, attrs):
        if var is self._total_var:
            group_tab = self._group_tables.total
            offset = int(bool(not self.single_var_grouping))
            leading_vars = [self._total_var]
            combs = np.array([[0]])
            sub_table_getter = lambda x: \
                self._table[np.where((~np.isnan(self._row_var_col)) &
                                     (~np.isnan(self._col_var_col)))[0]]
        elif var is self._row_var or self.single_var_grouping:
            group_tab = self._group_tables.total_v
            offset = int(bool(not self.single_var_grouping))
            leading_vars = [self._row_var]
            combs = self._row_var_groups[:, None]
            sub_table_getter = lambda x: \
                self._table[np.where((~np.isnan(self._col_var_col)) &
                                     (self._row_var_col == x[0]))[0]]
        elif var is self._col_var:
            group_tab = self._group_tables.total_h
            offset = int(bool(not self.single_var_grouping))
            leading_vars = [self._col_var]
            combs = self._col_var_groups[:, None]
            sub_table_getter = lambda x: \
                self._table[np.where((~np.isnan(self._row_var_col)) &
                                     (self._col_var_col == x[0]))[0]]
        else:
            group_tab = self._group_tables.table
            offset = 0
            leading_vars = [self._row_var, self._col_var]
            combs = np.array(list(product(self._row_var_groups,
                                          self._col_var_groups)))
            sub_table_getter = lambda x: \
                self._table[np.where((self._row_var_col == x[0])
                                     & (self._col_var_col == x[1]))[0]]

        if not combs.shape[0]:
            return None

        n = len(var_indep_funs) + len(var_dep_funs)
        X = np.zeros((len(combs), n), dtype=float)
        for i, comb in enumerate(combs):
            sub_table = sub_table_getter(comb)
            j = -1
            for j, fun in enumerate(var_indep_funs):
                if fun in self._indepen_agg_done:
                    # TODO - optimize - after this line is executed,
                    # the whole column is already set
                    X[:, j] = group_tab.X[:, self._indepen_agg_done[fun] - offset]
                else:
                    X[i, j] = fun(sub_table)
            for k, (v, fun) in enumerate(var_dep_funs, j + 1):
                if fun in self._depen_agg_done:
                    X[:, k] = group_tab.X[:, self._depen_agg_done[fun][v] - offset]
                else:
                    X[i, k] = fun(sub_table.get_column_view(v)[0])

        #rename leading vars (seems the easiest) if needed
        current = [var.name for var in attrs]
        uniq_leading_vars = []
        for v in leading_vars:
            uniq = get_unique_names(current, v.name)
            if uniq != v.name:
                self.renamed.append(v.name)
                v = v.copy(name=uniq)
            uniq_leading_vars.append(v)
            current.append(uniq)

        return Table(Domain(uniq_leading_vars + attrs), np.hstack((combs, X)))

    def update_pivot_table(self, val_var: Variable):
        self._create_pivot_tables(val_var)

    def _create_pivot_tables(self, val_var):
        if not self._group_tables.table:
            self._pivot_tables = self.Tables()
            return

        agg_funs = [fun for fun in self._current_agg_functions
                    if fun in self.AutonomousFunctions
                    or val_var and self.__include_aggregation(fun, val_var)]
        X, X_h, X_v, X_t = self.__get_pivot_tab_x(val_var, agg_funs)
        dom, dom_h, dom_v, dom_t = self.__get_pivot_tab_domain(
            val_var, X, X_h, X_v, X_t, agg_funs)
        for t, d, x in (("table", dom, X), ("total_h", dom_h, X_h),
                        ("total_v", dom_v, X_v), ("total", dom_t, X_t)):
            setattr(self._pivot_tables, t, Table(d, x))

    # pylint: disable=invalid-name
    def __get_pivot_tab_domain(self, val_var, X, X_h, X_v, X_t, agg_funs):
        def map_values(index, _X):
            values = np.unique(_X[:, index])
            values = np.delete(values, np.where(values == "nan")[0])
            for j, value in enumerate(values):
                _X[:, index][_X[:, index] == value] = j
            return values

        create_time_var = \
            isinstance(val_var, TimeVariable) and \
            all(fun in self.TimeVarFunctions for fun in agg_funs)
        create_cont_var = \
            not val_var or val_var.is_continuous and \
            (not isinstance(val_var, TimeVariable) or
             all(fun in self.FloatFunctions for fun in agg_funs))

        vals = np.array(self._col_var.values)[self._col_var_groups.astype(int)]
        if create_time_var:
            kwargs = {"have_date": val_var.have_date,
                      "have_time": val_var.have_time}
            attrs = [[TimeVariable(f"{v}", **kwargs) for v in vals]] * 2
            attrs.extend([[TimeVariable("Total", **kwargs)]] * 2)
        elif create_cont_var:
            attrs = [[ContinuousVariable(f"{v}", 1) for v in vals]] * 2
            attrs.extend([[ContinuousVariable("Total", 1)]] * 2)
        else:
            attrs = []
            for x in (X, X_h):
                attrs.append([DiscreteVariable(f"{v}", map_values(i, x))
                              for i, v in enumerate(vals, 2)])
            for x in (X_v, X_t):
                attrs.append([DiscreteVariable("Total", map_values(0, x))])
        row_var_h = DiscreteVariable(self._row_var.name, values=["Total"])
        aggr_attr = DiscreteVariable('Aggregate', [str(f) for f in agg_funs])

        same_row_col = self._col_var is self._row_var

        extra_vars = [self._row_var, aggr_attr]
        uniq_a = get_unique_names_duplicates([v.name for v in extra_vars]
                                             + [atr.name for atr in attrs[0]])
        for (idx, var), u in zip(enumerate(chain(extra_vars, attrs[0])), uniq_a):
            if var.name == u:
                continue
            if idx == 0:
                self.renamed.append(self._row_var.name)
                self._row_var = self._row_var.copy(name=u)
                if same_row_col:
                    self._col_var = self._row_var
                row_var_h = row_var_h.copy(name=u)
            elif idx == 1:
                self.renamed.append(aggr_attr.name)
                aggr_attr = aggr_attr.copy(name=u)
            else:
                self.renamed.append(var.name)
                attrs[0][idx-2] = var.copy(name=u)
                attrs[1][idx-2] = var.copy(name=u)

        if same_row_col:
            vals = tuple(v.name for v in attrs[0])
            self._row_var.make(self._row_var.name, values=vals)
            vals = tuple(v.name for v in attrs[2])
            row_var_h.make(row_var_h.name, vals)

        return (Domain([self._row_var, aggr_attr] + attrs[0]),
                Domain([row_var_h, aggr_attr] + attrs[1]),
                Domain(attrs[2]), Domain(attrs[3]))

    def __get_pivot_tab_x(self, val_var, agg_funs):
        gt = self._group_tables
        n_fun = len(agg_funs)
        n_rows, n_cols = len(self._row_var_groups), len(self._col_var_groups)
        is_float_type = not val_var or val_var.is_continuous
        if isinstance(val_var, TimeVariable):
            is_float_type = \
                all(fun in self.TimeVarFunctions for fun in agg_funs) or \
                all(fun in self.FloatFunctions for fun in agg_funs)
        kwargs = {"fill_value": np.nan, "dtype": float} if is_float_type \
            else {"fill_value": "", "dtype": object}
        X = np.full((n_rows * n_fun, 2 + n_cols), **kwargs)
        X_h = np.full((n_fun, 2 + n_cols), **kwargs)
        X_v = np.full((n_rows * n_fun, 1), **kwargs)
        X_t = np.full((n_fun, 1), **kwargs)
        for i, fun in enumerate(agg_funs):
            args = (val_var, fun, is_float_type)
            X[i::n_fun, 2:] = self.__rows_for_function(n_rows, n_cols, *args)
            X[i::n_fun, :2] = np.array([[row_val, agg_funs.index(fun)] for
                                        row_val in self._row_var_groups])
            X_h[i, :2] = 0, agg_funs.index(fun)
            X_h[i, 2:] = self.__total_for_function(gt.total_h, *args)
            X_v[i::n_fun, 0] = self.__total_for_function(gt.total_v, *args)
            X_t[i] = self.__total_for_function(gt.total, *args)
        return X, X_h, X_v, X_t

    def __total_for_function(self, group_tab, val_var, fun, is_float_type):
        ref = self._indepen_agg_done.get(fun, None) \
              or self._depen_agg_done[fun][val_var]
        ref -= int(bool(not self.single_var_grouping))
        return self.__check_continuous(val_var, group_tab.X[:, ref],
                                       fun, is_float_type)

    def __rows_for_function(self, n_rows, n_cols, val_var, fun, is_float_type):
        ref = self._indepen_agg_done.get(fun, None) \
              or self._depen_agg_done[fun][val_var]
        column = self._group_tables.table.X[:, ref]
        if self.single_var_grouping:
            rows = np.full((n_rows, n_cols), fun(np.array([]), ), dtype=float)
            rows[np.diag_indices_from(rows)] = column
        else:
            rows = column.reshape(n_rows, n_cols)
        return self.__check_continuous(val_var, rows, fun, is_float_type)

    def __check_continuous(self, val_var, column, fun, is_float_type):
        if val_var and not val_var.is_continuous:
            column = column.astype(str)
            if fun in self.DiscVarFunctions:
                for j, val in enumerate(val_var.values):
                    column[column == str(float(j))] = val
        elif isinstance(val_var, TimeVariable) and not is_float_type:
            shape = column.shape
            column = column.flatten()
            column_ = column.astype(str)
            if fun in self.TimeVarFunctions:
                for i in range(column.shape[0]):
                    if not np.isnan(column[i]):
                        column_[i] = val_var.repr_val(column[i])
            return column_.reshape(shape)
        return column

    @staticmethod
    def count_defined(x):
        if x.shape[0] == 0:
            return 0
        if x.size and np.issubdtype(x.dtype, np.number) and not sp.issparse(x):
            nans = np.isnan(x).sum(axis=0)
        elif sp.issparse(x) and x.size:
            nans = np.bincount(x.nonzero()[1], minlength=x.shape[1])
            x = x.tocsc()
        else:
            x_str = x.astype(str)
            nans = ((x_str == "nan") | (x_str == "")).sum(axis=0) \
                if x.size else np.zeros(x.shape[1])
        return x.shape[0] - nans

    @staticmethod
    def stat(x, f):
        return f(x.astype(float), axis=0) if x.shape[0] > 0 else np.nan

    @staticmethod
    def mode(x):
        return Pivot.stat(x, nanmode).mode if x.shape[0] > 0 else np.nan

    @staticmethod
    def majority(x):
        if x.shape[0] == 0:
            return np.nan
        counts = bincount(x)[0]
        return np.argmax(counts) if counts.shape[0] else np.nan

    Count.func = lambda x: len(x[0])
    Count_defined.func = lambda x: Pivot.count_defined(x[0])
    Sum.func = lambda x: nansum(x[0], axis=0) if x[0].shape[0] > 0 else 0
    Mean.func = lambda x: Pivot.stat(x[0], nanmean)
    Min.func = lambda x: Pivot.stat(x[0], nanmin)
    Max.func = lambda x: Pivot.stat(x[0], nanmax)
    Median.func = lambda x: Pivot.stat(x[0], nanmedian)
    Mode.func = lambda x: Pivot.mode(x[0])
    Var.func = lambda x: Pivot.stat(x[0], nanvar)
    Majority.func = lambda x: Pivot.majority(x[0])


class BorderedItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        """Overloads `paint` to draw borders"""
        QStyledItemDelegate.paint(self, painter, option, index)
        if index.data(BorderRole):
            painter.save()
            painter.setPen(index.data(BorderColorRole))
            rect = option.rect
            painter.drawLine(rect.topLeft(), rect.topRight())
            painter.restore()


class PivotTableView(QTableView):
    selection_changed = pyqtSignal()

    TOTAL_STRING = "Total"

    def __init__(self):
        super().__init__(editTriggers=QTableView.NoEditTriggers)
        self._n_classesv = None  # number of row_feature values
        self._n_classesh = None  # number of col_feature values
        self._n_agg_func = None  # number of aggregation functions
        self._n_leading_rows = None  # number of leading rows
        self._n_leading_cols = None  # number of leading columns

        self.table_model = QStandardItemModel(self)
        self.setModel(self.table_model)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.horizontalHeader().setMinimumSectionSize(60)
        self.setShowGrid(False)
        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        self.setItemDelegate(BorderedItemDelegate())
        self.pressed.connect(self.__cell_clicked)
        self.clicked.connect(self.__cell_clicked)
        self.entered.connect(self.__cell_entered)
        self.__clicked_cell = None

    @property
    def add_agg_column(self) -> bool:
        return self._n_agg_func > 1

    def __cell_entered(self, model_index):
        if self.__clicked_cell is None:
            return
        index = self.table_model.index
        selection = None
        i_end, j_end = model_index.row(), model_index.column()
        i_start, j_start = self.__clicked_cell
        i_start, i_end = sorted([i_start, i_end])
        j_start, j_end = sorted([j_start, j_end])
        if i_start >= self._n_leading_rows and j_start >= self._n_leading_cols:
            i_start = (i_start - self._n_leading_rows) // self._n_agg_func * \
                self._n_agg_func + self._n_leading_rows
            i_end = (i_end - self._n_leading_rows) // self._n_agg_func * \
                self._n_agg_func + self._n_leading_rows + self._n_agg_func - 1
            start, end = index(i_start, j_start), index(i_end, j_end)
            selection = QItemSelection(start, end)
        if selection is not None:
            self.selectionModel().select(
                selection, QItemSelectionModel.ClearAndSelect)
        self.selection_changed.emit()

    def __cell_clicked(self, model_index):
        i, j = model_index.row(), model_index.column()
        self.__clicked_cell = (i, j)
        m, n = self.table_model.rowCount(), self.table_model.columnCount()
        index = self.table_model.index
        selection = None
        if i > m - self._n_agg_func - 1 and j == n - 1:
            start_index = index(self._n_leading_rows, self._n_leading_cols)
            selection = QItemSelection(start_index, index(m - 1, n - 1))
        elif i == self._n_leading_rows - 1 or i > m - self._n_agg_func - 1:
            start_index = index(self._n_leading_rows, j)
            selection = QItemSelection(start_index, index(m - 1, j))
        elif j in (self._n_leading_cols - 1, n - 1, 1):
            i_start = (i - self._n_leading_rows) // self._n_agg_func * \
                      self._n_agg_func + self._n_leading_rows
            i_end = i_start + self._n_agg_func - 1
            start_index = index(i_start, self._n_leading_cols)
            selection = QItemSelection(start_index, index(i_end, n - 1))
        elif i >= self._n_leading_rows and j >= self._n_leading_cols:
            i_start = (i - self._n_leading_rows) // self._n_agg_func * \
                      self._n_agg_func + self._n_leading_rows
            i_end = i_start + self._n_agg_func - 1
            selection = QItemSelection(index(i_start, j), index(i_end, j))

        if selection is not None:
            self.selectionModel().select(
                selection, QItemSelectionModel.ClearAndSelect)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.selection_changed.emit()

    def update_table(self, titleh: str, titlev: str, table: Table,
                     table_total_h: Table, table_total_v: Table,
                     table_total: Table):
        self.clear()
        if not table:
            return

        self._initialize(table, table_total_h)
        self._set_headers(titleh, titlev, table)
        self._set_values(table[:, 2:])
        self._set_totals(table_total_h[:, 2:], table_total_v, table_total)
        self._draw_lines()
        self._resize(table)

    def _initialize(self, table, table_total_h):
        self._n_classesv = int(len(table) / len(table_total_h))
        self._n_classesh = table.X.shape[1] - 2
        self._n_agg_func = len(table_total_h)
        self._n_leading_rows = 2
        self._n_leading_cols = 2 + int(len(table_total_h) > 1)

    def _set_headers(self, titleh, titlev, table):
        self.__set_horizontal_title(titleh)
        self.__set_vertical_title(titlev)
        self.__set_flags_title()
        self.__set_horizontal_headers(table)
        self.__set_vertical_headers(table)

    def __set_horizontal_title(self, titleh):
        item = QStandardItem()
        item.setData(titleh, Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignCenter)
        self.table_model.setItem(0, self._n_leading_cols, item)
        self.setSpan(0, self._n_leading_cols, 1, self._n_classesh + 3)

    def __set_vertical_title(self, titlev):
        item = QStandardItem()
        item.setData(titlev, Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.setItemDelegateForColumn(0, gui.VerticalItemDelegate(extend=True))
        self.table_model.setItem(self._n_leading_rows, 0, item)
        row_span = self._n_classesv * self._n_agg_func + 1
        self.setSpan(self._n_leading_rows, 0, row_span, 1)

    def __set_flags_title(self):
        item = self.table_model.item(0, self._n_leading_cols)
        item.setFlags(Qt.NoItemFlags)
        item = self.table_model.item(self._n_leading_rows, 0)
        item.setFlags(Qt.NoItemFlags)
        for i, j in product(range(self._n_leading_rows),
                            range(self._n_leading_cols)):
            item = QStandardItem()
            item.setFlags(Qt.NoItemFlags)
            self.table_model.setItem(i, j, item)

    def __set_horizontal_headers(self, table):
        labels = [a.name for a in table.domain[1:]] + [self.TOTAL_STRING]
        if not self.add_agg_column:
            labels[0] = str(table[0, 1])
        for i, label in enumerate(labels, self._n_leading_cols - 1):
            self.table_model.setItem(1, i, self._create_header_item(label))

    def __set_vertical_headers(self, table):
        labels = [(str(row[0]), str(row[1])) for row in table]
        i = self._n_leading_rows - 1
        for i, (l1, l2) in enumerate(labels, self._n_leading_rows):
            l1 = "" if (i - self._n_leading_rows) % self._n_agg_func else l1
            self.table_model.setItem(i, 1, self._create_header_item(l1))
            if self.add_agg_column:
                self.table_model.setItem(i, 2, self._create_header_item(l2))

        if self.add_agg_column:
            labels = [str(row[1]) for row in table[:self._n_agg_func]]
            start = self._n_leading_rows + self._n_agg_func * self._n_classesv
            for j, l2 in enumerate(labels, i + 1):
                l1 = self.TOTAL_STRING if j == start else ""
                self.table_model.setItem(j, 1, self._create_header_item(l1))
                self.table_model.setItem(j, 2, self._create_header_item(l2))
        else:
            item = self._create_header_item(self.TOTAL_STRING)
            self.table_model.setItem(i + 1, 1, item)

    def _set_values(self, table):
        for i, j in product(range(len(table)), range(len(table[0]))):
            value = table[i, j]
            item = self._create_value_item(str(value))
            self.table_model.setItem(i + self._n_leading_rows,
                                     j + self._n_leading_cols, item)

    def _set_totals(self, table_total_h, table_total_v, table_total):
        def set_total_item(table, get_row, get_col):
            for i, j in product(range(len(table)), range(len(table[0]))):
                item = self._create_header_item(str(table[i, j]))
                self.table_model.setItem(get_row(i), get_col(j), item)

        last_row = self._n_leading_rows + self._n_classesv * self._n_agg_func
        last_col = self._n_leading_cols + self._n_classesh
        set_total_item(table_total_v, lambda x: x + self._n_leading_rows,
                       lambda x: last_col)
        set_total_item(table_total_h, lambda x: x + last_row,
                       lambda x: x + self._n_leading_cols)
        set_total_item(table_total, lambda x: x + last_row, lambda x: last_col)

    def _create_header_item(self, text):
        bold_font = self.table_model.invisibleRootItem().font()
        bold_font.setBold(True)
        item = QStandardItem()
        item.setData(text, Qt.DisplayRole)
        item.setFont(bold_font)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        item.setFlags(Qt.ItemIsEnabled)
        return item

    @staticmethod
    def _create_value_item(text):
        item = QStandardItem()
        item.setData(text, Qt.DisplayRole)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        return item

    def _draw_lines(self):
        end_col = self._n_leading_cols + self._n_classesh + 1
        total_row = self._n_leading_rows + self._n_classesv * self._n_agg_func
        indices = [(total_row, j) for j in range(1, end_col)]
        for i in range(self._n_classesv):
            inner_row = self._n_agg_func * i + self._n_leading_rows
            inner_indices = [(inner_row, j) for j in range(1, end_col)]
            indices = indices + inner_indices
            if not self.add_agg_column:
                break
        for i, j in indices:
            item = self.table_model.item(i, j)
            item.setData("t", BorderRole)
            item.setData(QColor(160, 160, 160), BorderColorRole)

    def _resize(self, table):
        labels = [a.name for a in table.domain[1:]] + [self.TOTAL_STRING]
        if len(' '.join(labels)) < 120:
            self.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents)
        else:
            self.horizontalHeader().setDefaultSectionSize(60)

    def get_selection(self) -> Set:
        m, n = self._n_leading_rows, self._n_leading_cols
        return {(ind.row() - m, ind.column() - n)
                for ind in self.selectedIndexes()}

    def set_selection(self, indexes: Set):
        selection = QItemSelection()
        index = self.model().index
        for row, col in indexes:
            sel = index(row + self._n_leading_rows, col + self._n_leading_cols)
            selection.select(sel, sel)
        self.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    def clear(self):
        self.table_model.clear()


class OWPivot(OWWidget):
    name = "Pivot Table"
    description = "Reshape data table based on column values."
    category = "Transform"
    icon = "icons/Pivot.svg"
    priority = 1220
    keywords = ["pivot", "group", "aggregate"]

    class Inputs:
        data = Input("Data", Table, default=True)

    class Outputs:
        pivot_table = Output("Pivot Table", Table, default=True)
        filtered_data = Output("Filtered Data", Table)
        grouped_data = Output("Grouped Data", Table)

    class Warning(OWWidget.Warning):
        # TODO - inconsistent for different variable types
        no_col_feature = Msg("Column feature should be selected.")
        cannot_aggregate = Msg("Some aggregations ({}) cannot be computed.")
        renamed_vars = Msg("Some variables have been renamed in some tables"
                           "to avoid duplicates.\n{}")
        too_many_values = Msg("Selected variable has too many values.")
        no_variables = Msg("At least 1 primitive variable is required.")

    settingsHandler = DomainContextHandler()
    row_feature = ContextSetting(None)
    col_feature = ContextSetting(None)
    val_feature = ContextSetting(None)
    sel_agg_functions = Setting(set([Pivot.Count]))
    selection = Setting(set(), schema_only=True)
    auto_commit = Setting(True)

    AGGREGATIONS = (Pivot.Count,
                    Pivot.Count_defined,
                    None,  # separator
                    Pivot.Sum,
                    Pivot.Mean,
                    Pivot.Var,
                    Pivot.Median,
                    2,  # column break
                    Pivot.Mode,
                    Pivot.Min,
                    Pivot.Max,
                    None,
                    Pivot.Majority)

    MAX_VALUES = 100

    def __init__(self):
        super().__init__()
        self.data = None  # type: Table
        self.pivot = None  # type: Pivot
        self.__pending_selection = self.selection  # type: Set
        self._add_control_area_controls()
        self._add_main_area_controls()

    def _add_control_area_controls(self):
        gui.comboBox(gui.vBox(self.controlArea, box="Rows"),
                     self, "row_feature",
                     contentsLength=14,
                     searchable=True,
                     model=DomainModel(valid_types=DomainModel.PRIMITIVE),
                     callback=self.__feature_changed,
                     orientation=Qt.Horizontal)
        gui.comboBox(gui.vBox(self.controlArea, box="Columns"),
                     self, "col_feature",
                     contentsLength=14,
                     searchable=True,
                     model=DomainModel(placeholder="(Same as rows)",
                                       valid_types=DiscreteVariable),
                     callback=self.__feature_changed,
                     orientation=Qt.Horizontal)
        gui.comboBox(gui.vBox(self.controlArea, box="Values"),
                     self, "val_feature",
                     contentsLength=14,
                     searchable=True,
                     model=DomainModel(placeholder="(None)"),
                     callback=self.__val_feature_changed,
                     orientation=Qt.Horizontal)
        self.__add_aggregation_controls()
        gui.rubber(self.controlArea)
        gui.auto_apply(self.buttonsArea, self, "auto_commit")

    def __add_aggregation_controls(self):
        def new_inbox():
            nonlocal row, col, inbox
            inbox = QWidget()
            layout = QGridLayout()
            inbox.setLayout(layout)
            layout.setContentsMargins(0, 0, 0, 0)
            box.layout().addWidget(inbox)
            row = col = 0

        box = gui.vBox(self.controlArea, "Aggregations")
        row = col = 0
        inbox = None
        new_inbox()
        self.aggregation_checkboxes = []  # for test purposes
        for agg in self.AGGREGATIONS:
            if agg is None:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setLineWidth(1)
                line.setFrameShadow(QFrame.Sunken)
                box.layout().addWidget(line)
                new_inbox()
                continue
            elif agg == 2:
                col += 1
                row = 0
                continue
            check_box = QCheckBox(str(agg), inbox)
            check_box.setChecked(agg in self.sel_agg_functions)
            check_box.clicked.connect(lambda *args, a=agg:
                                      self.__aggregation_cb_clicked(a, args[0]))
            inbox.layout().addWidget(check_box, row, col)
            self.aggregation_checkboxes.append(check_box)
            row += 1

    def _add_main_area_controls(self):
        self.table_view = PivotTableView()
        self.table_view.selection_changed.connect(self.__invalidate_filtered)
        self.mainArea.layout().addWidget(self.table_view)

    @property
    def no_col_feature(self):
        return self.col_feature is None and self.row_feature is not None \
            and self.row_feature.is_continuous

    @property
    def skipped_aggs(self):
        def add(fun):
            data, var = self.data, self.val_feature
            primitive_funcs = Pivot.ContVarFunctions + Pivot.DiscVarFunctions
            return data and not var and fun not in Pivot.AutonomousFunctions \
                or var and var.is_discrete and fun in Pivot.ContVarFunctions \
                or var and var.is_continuous and fun in Pivot.DiscVarFunctions \
                or var and not var.is_primitive() and fun in primitive_funcs
        skipped = [str(fun) for fun in self.sel_agg_functions if add(fun)]
        return ", ".join(sorted(skipped))

    @property
    def data_has_primitives(self):
        if not self.data:
            return False
        domain = self.data.domain
        return any(v.is_primitive() for v in domain.variables + domain.metas)

    def __feature_changed(self):
        self.selection = set()
        self.pivot = None
        self.commit.deferred()

    def __val_feature_changed(self):
        self.selection = set()
        if self.no_col_feature or not self.pivot:
            return
        self.pivot.update_pivot_table(self.val_feature)
        self.commit.deferred()

    def __aggregation_cb_clicked(self, agg_fun: Pivot.Functions, checked: bool):
        self.selection = set()
        if checked:
            self.sel_agg_functions.add(agg_fun)
        else:
            self.sel_agg_functions.remove(agg_fun)
        if self.no_col_feature or not self.pivot or not self.data:
            return
        self.pivot.update_group_table(self.sel_agg_functions, self.val_feature)
        self.commit.deferred()

    def __invalidate_filtered(self):
        self.selection = self.table_view.get_selection()
        self.commit.deferred()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.selection = set()
        self.data = data
        self.pivot = None
        self.check_data()
        self.init_attr_values()
        if self.data_has_primitives:
            self.openContext(self.data)
        self.commit.now()

    def check_data(self):
        self.clear_messages()

    def init_attr_values(self):
        domain = self.data.domain if self.data and len(self.data) else None
        for attr in ("row_feature", "col_feature", "val_feature"):
            getattr(self.controls, attr).model().set_domain(domain)
            setattr(self, attr, None)
        model = self.controls.row_feature.model()
        if model:
            self.row_feature = model[0]
        model = self.controls.val_feature.model()
        if model and len(model) > 2:
            allvars = domain.variables + domain.metas
            self.val_feature = allvars[0] if allvars[0] in model else model[2]

    @gui.deferred
    def commit(self):
        def send_outputs(pivot_table, filtered_data, grouped_data):
            if self.data:
                if grouped_data:
                    grouped_data.name = self.data.name
                if pivot_table:
                    pivot_table.name = self.data.name
                if filtered_data:
                    filtered_data.name = self.data.name
            self.Outputs.grouped_data.send(grouped_data)
            self.Outputs.pivot_table.send(pivot_table)
            self.Outputs.filtered_data.send(filtered_data)

        self.Warning.renamed_vars.clear()
        self.Warning.too_many_values.clear()
        self.Warning.cannot_aggregate.clear()
        self.Warning.no_col_feature.clear()

        self.table_view.clear()

        if self.pivot is None:
            if self.data:
                if not self.data_has_primitives:
                    self.Warning.no_variables()
                    send_outputs(None, None, None)
                    return

            if self.no_col_feature:
                self.Warning.no_col_feature()
                send_outputs(None, None, None)
                return

            if self.data:
                col_var = self.col_feature or self.row_feature
                col = self.data.get_column_view(col_var)[0].astype(float)
                if len(nanunique(col)) >= self.MAX_VALUES:
                    self.table_view.clear()
                    self.Warning.too_many_values()
                    send_outputs(None, None, None)
                    return

            self.pivot = Pivot(self.data, self.sel_agg_functions,
                               self.row_feature,
                               self.col_feature, self.val_feature)

        if self.skipped_aggs:
            self.Warning.cannot_aggregate(self.skipped_aggs)
        self._update_graph()

        send_outputs(self.pivot.pivot_table,
                     self.get_filtered_data(),
                     self.pivot.group_table)

        if self.pivot.renamed:
            self.Warning.renamed_vars(self.pivot.renamed)

    def _update_graph(self):
        if self.pivot.pivot_table:
            col_feature = self.col_feature or self.row_feature
            self.table_view.update_table(col_feature.name,
                                         self.row_feature.name,
                                         *self.pivot.pivot_tables)
            selection = self.__pending_selection or self.selection
            self.table_view.set_selection(selection)
            self.selection = self.table_view.get_selection()
            self.__pending_selection = set()

    def get_filtered_data(self):
        if not self.data or not self.selection or not self.pivot.pivot_table:
            return None

        cond = []
        for i, j in self.selection:
            f = []
            for at, val in [(self.row_feature, self.pivot.pivot_table.X[i, 0]),
                            (self.col_feature, j)]:
                if isinstance(at, DiscreteVariable):
                    f.append(FilterDiscrete(at, [val]))
                elif isinstance(at, ContinuousVariable):
                    f.append(FilterContinuous(at, FilterContinuous.Equal, val))
            cond.append(Values(f))
        return Values(cond, conjunction=False)(self.data)

    @staticmethod
    def sizeHint():
        return QSize(640, 525)

    def send_report(self):
        self.report_items((
            ("Row feature", self.row_feature),
            ("Column feature", self.col_feature),
            ("Value feature", self.val_feature)))
        if self.data and self.val_feature is not None:
            self.report_table("", self.table_view)
        if not self.data:
            self.report_items((("Group by", self.row_feature),))
            self.report_table(self.table_view)


if __name__ == "__main__":
    WidgetPreview(OWPivot).run(set_data=Table("heart_disease"))
