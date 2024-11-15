from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import \
    Any, Dict, List, Optional, Set, Union, NamedTuple, Callable, Type

import pandas as pd
from numpy import nan
from AnyQt.QtCore import (
    QAbstractTableModel,
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    Qt,
)
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGridLayout,
    QHeaderView,
    QTableView,
)
from orangewidget.settings import ContextSetting, Setting
from orangewidget.utils.listview import ListViewFilter
from orangewidget.utils.signals import Input, Output
from orangewidget.utils import enum_as_int
from orangewidget.widget import Msg
from pandas.core.dtypes.common import is_datetime64_any_dtype

from Orange.data import (
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    StringVariable,
    Table,
    TimeVariable,
    Variable,
)
from Orange.data.aggregate import OrangeTableGroupBy
from Orange.util import wrap_callback
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget


class Aggregation(NamedTuple):
    function: Union[str, Callable]
    types: Set[Type[Variable]]
    # Gives the type of the result,
    # or True to copy the original variable,
    # or False to create a new variable of the same type as the input
    result_type: Union[Type[Variable], bool]


def concatenate(x):
    """
    Concatenate values of series if value is not missing (nan or empty string
    for StringVariable)
    """
    return " ".join(str(v) for v in x if not pd.isnull(v) and len(str(v)) > 0)


def std(s):
    """
    Std that also handle time variable. Pandas's std return Timedelta object in
    case of datetime columns - transform TimeDelta to seconds
    """
    std_ = s.std()
    if isinstance(std_, pd.Timedelta):
        return std_.total_seconds()
    # std returns NaT when cannot compute value - change it to nan to keep colum numeric
    return nan if pd.isna(std_) else std_


def var(s):
    """
    Variance that also handle time variable. Pandas's variance function somehow
    doesn't support DateTimeArray - this function fist converts datetime series
    to UNIX epoch and then computes variance
    """
    if is_datetime64_any_dtype(s):
        initial_ts = pd.Timestamp("1970-01-01", tz=None if s.dt.tz is None else "UTC")
        s = (s - initial_ts) / pd.Timedelta("1s")
    var_ = s.var()
    return var_.total_seconds() if isinstance(var_, pd.Timedelta) else var_


def span(s):
    """
    Span that also handle time variable. Time substitution return Timedelta
    object in case of datetime columns - transform TimeDelta to seconds
    """
    span_ = pd.Series.max(s) - pd.Series.min(s)
    return span_.total_seconds() if isinstance(span_, pd.Timedelta) else span_


AGGREGATIONS = {
    "Mean": Aggregation(
        "mean",
        {ContinuousVariable, TimeVariable},
        False),
    "Median": Aggregation(
        "median",
        {ContinuousVariable, TimeVariable},
        True),
    "Q1": Aggregation(
        lambda s: s.quantile(0.25),
        {ContinuousVariable, TimeVariable},
        True),
    "Q3": Aggregation(
        lambda s: s.quantile(0.75),
        {ContinuousVariable, TimeVariable},
        True),
    "Min. value": Aggregation(
        "min",
        {ContinuousVariable, TimeVariable},
        True),
    "Max. value": Aggregation(
        "max",
        {ContinuousVariable, TimeVariable},
        True),
    "Mode": Aggregation(
        lambda x: pd.Series.mode(x).get(0, nan),
        {ContinuousVariable, DiscreteVariable, TimeVariable},
        True
    ),
    "Standard deviation": Aggregation(
        std,
        {ContinuousVariable, TimeVariable},
        ContinuousVariable
    ),
    "Variance": Aggregation(
        var,
        {ContinuousVariable, TimeVariable},
        ContinuousVariable
    ),
    "Sum": Aggregation(
        "sum",
        {ContinuousVariable},
        True),
    "Concatenate": Aggregation(
        concatenate,
        {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable},
        StringVariable
    ),
    "Span": Aggregation(
        span,
        {ContinuousVariable, TimeVariable},
        ContinuousVariable
    ),
    "First value": Aggregation(
        "first",
        {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable},
        True
    ),
    "Last value": Aggregation(
        "last",
        {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable},
        True
    ),
    "Random value": Aggregation(
        lambda x: x.sample(1, random_state=0),
        {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable},
        True
    ),
    "Count defined": Aggregation(
        "count",
        {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable},
        ContinuousVariable
    ),
    "Count": Aggregation(
        "size",
        {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable},
        ContinuousVariable
    ),
    "Proportion defined": Aggregation(
        lambda x: x.count() / x.size,
        {ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable},
        ContinuousVariable
    ),
}
# list of ordered aggregation names is required on several locations so we
# prepare it in advance
AGGREGATIONS_ORD = list(AGGREGATIONS)

# use first aggregation suitable for each type as default
DEFAULT_AGGREGATIONS = {
    var: {next(name for name, agg in AGGREGATIONS.items() if var in agg.types)}
    for var in (ContinuousVariable, TimeVariable, DiscreteVariable, StringVariable)
}


@dataclass
class Result:
    group_by: OrangeTableGroupBy = None
    result_table: Optional[Table] = None


def _run(
    data: Table,
    group_by_attrs: List[Variable],
    aggregations: Dict[Variable, Set[str]],
    result: Result,
    state: TaskState,
) -> Result:
    def progress(part):
        state.set_progress_value(part * 100)
        if state.is_interruption_requested():
            raise Exception

    state.set_status("Aggregating")
    # group table rows
    if result.group_by is None:
        result.group_by = data.groupby(group_by_attrs)
    state.set_partial_result(result)

    aggregations = {
        var: [
            (agg, AGGREGATIONS[agg].function, AGGREGATIONS[agg].result_type)
            for agg in sorted(aggs, key=AGGREGATIONS_ORD.index)
        ]
        for var, aggs in aggregations.items()
    }
    result.result_table = result.group_by.aggregate(
        aggregations, wrap_callback(progress, 0.2, 1)
    )
    return result


class TabColumn:
    attribute = 0
    aggregations = 1


TABLE_COLUMN_NAMES = ["Attributes", "Aggregations"]


class VarTableModel(QAbstractTableModel):
    def __init__(self, parent: "OWGroupBy", *args):
        super().__init__(*args)
        self.domain = None
        self.parent = parent

    def set_domain(self, domain: Domain) -> None:
        """
        Reset the table view to new domain
        """
        self.domain = domain
        self.modelReset.emit()

    def update_aggregation(self, attribute: str) -> None:
        """
        Reset the aggregation values in the table for the attribute
        """
        index = self.domain.index(attribute)
        if index < 0:
            # indices of metas are negative: first meta -1, second meta -2, ...
            index = len(self.domain.variables) - 1 - index
        index = self.index(index, 1)
        self.dataChanged.emit(index, index)

    def rowCount(self, parent=None) -> int:
        return (
            0
            if self.domain is None or (parent is not None and parent.isValid())
            else len(self.domain.variables) + len(self.domain.metas)
        )

    @staticmethod
    def columnCount(parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(TABLE_COLUMN_NAMES)

    def data(self, index, role=Qt.DisplayRole) -> Any:
        row, col = index.row(), index.column()
        val = (self.domain.variables + self.domain.metas)[row]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == TabColumn.attribute:
                return str(val)
            else:  # col == TabColumn.aggregations
                # plot first two aggregations comma separated and write n more
                # for others
                aggs = sorted(
                    self.parent.aggregations.get(val, []), key=AGGREGATIONS_ORD.index
                )
                n_more = "" if len(aggs) <= 3 else f" and {len(aggs) - 3} more"
                return ", ".join(aggs[:3]) + n_more
        elif role == Qt.DecorationRole and col == TabColumn.attribute:
            return gui.attributeIconDict[val]
        return None

    def headerData(self, i, orientation, role=Qt.DisplayRole) -> str:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and i < 2:
            return TABLE_COLUMN_NAMES[i]
        return super().headerData(i, orientation, role)


class AggregateListViewSearch(ListViewFilter):
    """ListViewSearch that disables unselecting all items in the list"""

    def selectionCommand(
        self, index: QModelIndex, event: QEvent = None
    ) -> QItemSelectionModel.SelectionFlags:
        flags = super().selectionCommand(index, event)
        selmodel = self.selectionModel()
        if not index.isValid():  # Click on empty viewport; don't clear
            return QItemSelectionModel.NoUpdate
        if selmodel.isSelected(index):
            currsel = selmodel.selectedIndexes()
            if len(currsel) == 1 and index == currsel[0]:
                # Is the last selected index; do not deselect it
                return QItemSelectionModel.NoUpdate
        if (
            event is not None
            and event.type() == QEvent.MouseMove
            and flags & QItemSelectionModel.ToggleCurrent
        ):
            # Disable ctrl drag 'toggle'; can be made to deselect the last
            # index, would need to keep track of the current selection
            # (selectionModel does this but does not expose it)
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select
        return flags


class CheckBox(QCheckBox):
    def __init__(self, text, parent):
        super().__init__(text)
        self.parent: OWGroupBy = parent

    def nextCheckState(self) -> None:
        """
        Custom behaviour for switching between steps. It is required since
        sometimes user will select different types of attributes at the same
        time. In this case we step between unchecked, partially checked and
        checked or just between unchecked and checked - depending on situation
        """
        if self.checkState() == Qt.Checked:
            # if checked always uncheck
            self.setCheckState(Qt.Unchecked)
        else:
            agg = self.text()
            selected_attrs = self.parent.get_selected_attributes()
            types = set(type(attr) for attr in selected_attrs)
            can_be_applied_all = types <= AGGREGATIONS[agg].types

            # true if aggregation applied to all attributes that can be
            # aggregated with selected aggregation
            applied_all = all(
                type(attr) not in AGGREGATIONS[agg].types
                or agg in self.parent.aggregations[attr]
                for attr in selected_attrs
            )
            if self.checkState() == Qt.PartiallyChecked:
                # if partially check: 1) check if agg can be applied to all
                # 2) else uncheck if agg already applied to all
                # 3) else leve partially checked to apply to all that can be aggregated
                if can_be_applied_all:
                    self.setCheckState(Qt.Checked)
                elif applied_all:
                    self.setCheckState(Qt.Unchecked)
                else:
                    self.setCheckState(Qt.PartiallyChecked)
                    # since checkbox state stay same signal is not emitted
                    # automatically but we need a callback call so we emit it
                    self.stateChanged.emit(enum_as_int(Qt.PartiallyChecked))
            else:  # self.checkState() == Qt.Unchecked
                # if unchecked: check if all can be checked else partially check
                self.setCheckState(
                    Qt.Checked if can_be_applied_all else Qt.PartiallyChecked
                )


@contextmanager
def block_signals(widget):
    widget.blockSignals(True)
    try:
        yield
    finally:
        widget.blockSignals(False)


class OWGroupBy(OWWidget, ConcurrentWidgetMixin):
    name = "Group by"
    description = ""
    category = "Transform"
    icon = "icons/GroupBy.svg"
    keywords = "aggregate, group by"
    priority = 1210

    class Inputs:
        data = Input("Data", Table, doc="Input data table")

    class Outputs:
        data = Output("Data", Table, doc="Aggregated data")

    class Error(OWWidget.Error):
        unexpected_error = Msg("{}")

    settingsHandler = DomainContextHandler()

    gb_attrs: List[Variable] = ContextSetting([])
    aggregations: Dict[Variable, Set[str]] = ContextSetting({})
    auto_commit: bool = Setting(True)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        self.data = None
        self.result = None

        self.gb_attrs_model = DomainModel(
            separators=False,
        )
        self.agg_table_model = VarTableModel(self)
        self.agg_checkboxes = {}

        self.__init_control_area()
        self.__init_main_area()

    def __init_control_area(self) -> None:
        """Init all controls in the control area"""
        gui.listView(
            self.controlArea,
            self,
            "gb_attrs",
            box="Group by",
            model=self.gb_attrs_model,
            viewType=AggregateListViewSearch,
            callback=self.__gb_changed,
            selectionMode=ListViewFilter.ExtendedSelection,
        )

        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def __init_main_area(self) -> None:
        """Init all controls in the main area"""
        # aggregation table
        self.agg_table_view = tableview = QTableView()
        tableview.setModel(self.agg_table_model)
        tableview.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableview.selectionModel().selectionChanged.connect(self.__rows_selected)
        tableview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        vbox = gui.vBox(self.mainArea, " ")
        vbox.layout().addWidget(tableview)

        # aggregations checkboxes
        grid_layout = QGridLayout()
        gui.widgetBox(self.mainArea, orientation=grid_layout, box="Aggregations")

        col = 0
        row = 0
        break_rows = (6, 6, 99)
        for agg in AGGREGATIONS:
            self.agg_checkboxes[agg] = cb = CheckBox(agg, self)
            cb.setDisabled(True)
            cb.stateChanged.connect(partial(self.__aggregation_changed, agg))
            grid_layout.addWidget(cb, row, col)
            row += 1
            if row == break_rows[col]:
                row = 0
                col += 1

    ############
    # Callbacks

    def __rows_selected(self) -> None:
        """Callback for table selection change; update checkboxes"""
        selected_attrs = self.get_selected_attributes()

        types = {type(attr) for attr in selected_attrs}
        active_aggregations = [self.aggregations[attr] for attr in selected_attrs]
        for agg, cb in self.agg_checkboxes.items():
            cb.setDisabled(not types & AGGREGATIONS[agg].types)

            activated = {agg in a for a in active_aggregations}
            with block_signals(cb):
                # check if aggregation active for all selected attributes,
                # partially check if active for some else uncheck
                cb.setCheckState(
                    Qt.Checked
                    if activated == {True}
                    else (Qt.Unchecked if activated == {False} else Qt.PartiallyChecked)
                )

    def __gb_changed(self) -> None:
        """Callback for Group-by attributes selection change"""
        self.result = Result()
        self.commit.deferred()

    def __aggregation_changed(self, agg: str) -> None:
        """
        Callback for aggregation change; update aggregations dictionary and call
        commit
        """
        selected_attrs = self.get_selected_attributes()
        for attr in selected_attrs:
            if self.agg_checkboxes[agg].isChecked() and self.__aggregation_compatible(
                agg, attr
            ):
                self.aggregations[attr].add(agg)
            else:
                self.aggregations[attr].discard(agg)
            self.agg_table_model.update_aggregation(attr)
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data: Table) -> None:
        self.closeContext()
        self.data = data

        # reset states
        self.cancel()
        self.result = Result()
        self.Outputs.data.send(None)
        self.gb_attrs_model.set_domain(data.domain if data else None)
        self.gb_attrs = self.gb_attrs_model[:1] if self.gb_attrs_model else []
        self.aggregations = (
            {
                attr: DEFAULT_AGGREGATIONS[type(attr)].copy()
                for attr in data.domain.variables + data.domain.metas
            }
            if data
            else {}
        )
        default_aggregations = self.aggregations.copy()

        self.openContext(self.data)

        # restore aggregations
        self.aggregations.update({k: v for k, v in default_aggregations.items()
                                  if k not in self.aggregations})

        # update selections in widgets and re-plot
        self.agg_table_model.set_domain(data.domain if data else None)
        self._set_gb_selection()

        self.commit.now()

    #########################
    # Task connected methods

    @gui.deferred
    def commit(self) -> None:
        self.Error.clear()
        self.Warning.clear()
        if self.data:
            self.start(_run, self.data, self.gb_attrs, self.aggregations, self.result)

    def on_done(self, result: Result) -> None:
        self.result = result
        self.Outputs.data.send(result.result_table)

    def on_partial_result(self, result: Result) -> None:
        # store result in case the task is canceled and on_done is not called
        self.result = result

    def on_exception(self, ex: Exception):
        self.Error.unexpected_error(str(ex))

    ###################
    # Helper methods

    def get_selected_attributes(self):
        """Get select attributes in the table"""
        selection_model = self.agg_table_view.selectionModel()
        sel_rows = selection_model.selectedRows()
        vars_ = self.data.domain.variables + self.data.domain.metas
        return [vars_[index.row()] for index in sel_rows]

    def _set_gb_selection(self) -> None:
        """
        Update selected attributes. When context includes variable hidden in
        data, it will match and gb_attrs may include hidden attribute. Remove it
        since otherwise widget groups by attribute that is not present in view.
        """
        values = self.gb_attrs_model[:]
        self.gb_attrs = [var_ for var_ in self.gb_attrs if var_ in values]
        if not self.gb_attrs and self.gb_attrs_model:
            # if gb_attrs empty select first
            self.gb_attrs = self.gb_attrs_model[:1]

    @staticmethod
    def __aggregation_compatible(agg, attr):
        """Check a compatibility of aggregation with the variable"""
        return type(attr) in AGGREGATIONS[agg].types

    @classmethod
    def migrate_context(cls, context, _):
        """
        Before widget allowed using Sum on Time variable, now it is forbidden.
        This function removes Sum from the context for TimeVariables (104)
        """
        for var_, v in context.values["aggregations"][0].items():
            if len(var_) == 2:
                if var_[1] == 104:
                    v.discard("Sum")


if __name__ == "__main__":
    # pylint: disable=ungrouped-imports
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWGroupBy).run(Table("iris"))
