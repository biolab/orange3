import concurrent.futures
from dataclasses import dataclass
from typing import Optional, Union, Sequence, List, TypedDict, Tuple, Dict

from scipy.sparse import issparse

from AnyQt.QtWidgets import QTableView, QHeaderView, QApplication, QStyle
from AnyQt.QtGui import QColor, QClipboard
from AnyQt.QtCore import Qt, QSize, QMetaObject, QItemSelectionModel
from AnyQt.QtCore import Slot

import Orange.data
from Orange.data import Variable
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable

from Orange.widgets import gui
from Orange.widgets.data.utils.models import RichTableModel, TableSliceProxy
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemdelegates import TableDataDelegate
from Orange.widgets.utils.tableview import table_selection_to_mime_data
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils import disconnected
from Orange.widgets.data.utils.tableview import RichTableView
from Orange.widgets.data.utils import tablesummary as tsummary


class DataTableView(gui.HScrollStepMixin, RichTableView):
    pass


class TableBarItemDelegate(gui.TableBarItem, TableDataDelegate):
    pass


@dataclass
class InputData:
    table: Table
    summary: Union[tsummary.Summary, tsummary.ApproxSummary]
    model: TableModel


class _Selection(TypedDict):
    rows: Tuple[int]
    columns: Tuple[int]


_Sorting = List[Tuple[str, int]]


class OWTable(OWWidget):
    name = "Data Table"
    description = "View the dataset in a spreadsheet."
    icon = "icons/Table.svg"
    priority = 50
    keywords = "data table, view"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Warning(OWWidget.Warning):
        missing_sort_columns = Msg(
            "Cannot restore sorting.\n"
            "Missing columns in input table: {}"
        )
        non_sortable_input = Msg(
            "Cannot restore sorting.\n"
            "Input table cannot be sorted due to implementation constraints."
        )
    buttons_area_orientation = Qt.Vertical

    show_distributions = Setting(False)
    show_attribute_labels = Setting(True)
    select_rows = Setting(True)
    auto_commit = Setting(True)

    color_by_class = Setting(True)
    stored_selection: _Selection = Setting(
        {"rows": [], "columns": []}, schema_only=True
    )
    stored_sort: _Sorting = Setting(
        [], schema_only=True
    )
    settings_version = 1

    def __init__(self):
        super().__init__()
        self.input: Optional[InputData] = None
        self.__pending_selection: Optional[_Selection] = self.stored_selection
        self.__pending_sort: Optional[_Sorting] = self.stored_sort
        self.dist_color = QColor(220, 220, 220, 255)

        info_box = gui.vBox(self.controlArea, "Info")
        self.info_text = gui.widgetLabel(info_box)

        box = gui.vBox(self.controlArea, "Variables")
        self.c_show_attribute_labels = gui.checkBox(
            box, self, "show_attribute_labels",
            "Show variable labels (if present)",
            callback=self._update_variable_labels)

        gui.checkBox(box, self, "show_distributions",
                     'Visualize numeric values',
                     callback=self._on_distribution_color_changed)
        gui.checkBox(box, self, "color_by_class", 'Color by instance classes',
                     callback=self._on_distribution_color_changed)

        box = gui.vBox(self.controlArea, "Selection")

        gui.checkBox(box, self, "select_rows", "Select full rows",
                     callback=self._on_select_rows_changed)

        gui.rubber(self.controlArea)

        gui.button(self.buttonsArea, self, "Restore Original Order",
                   callback=self.restore_order,
                   tooltip="Show rows in the original order",
                   autoDefault=False,
                   attribute=Qt.WA_LayoutUsesWidgetRect)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        view = DataTableView(
            sortingEnabled=True
        )
        view.setSortingEnabled(True)
        view.setItemDelegate(TableDataDelegate(view))

        if self.select_rows:
            view.setSelectionBehavior(QTableView.SelectRows)

        header = view.horizontalHeader()
        header.setSectionsMovable(True)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, Qt.AscendingOrder)
        header.sortIndicatorChanged.connect(
            self._on_sort_indicator_changed, Qt.UniqueConnection
        )

        self.view = view
        self.mainArea.layout().addWidget(self.view)
        self._update_input_summary()

    def copy_to_clipboard(self):
        self.copy()

    def sizeHint(self):
        return QSize(800, 500)

    @Inputs.data
    def set_dataset(self, data: Optional[Table]):
        """Set the input dataset."""
        # reset the (header) view state.
        self.view.setModel(None)
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        if data is not None:
            self.input = InputData(
                table=data,
                summary=tsummary.table_summary(data),
                model=RichTableModel(data)
            )
            self._setup_table_view()
        else:
            self.input = None

    def handleNewSignals(self):
        super().handleNewSignals()
        self.Warning.non_sortable_input.clear()
        self.Warning.missing_sort_columns.clear()
        data: Optional[Table] = self.input.table if self.input else None
        slot = self.input
        if slot is not None and isinstance(slot.summary.len, concurrent.futures.Future):
            def update(_):
                QMetaObject.invokeMethod(
                    self, "_update_info", Qt.QueuedConnection)
            slot.summary.len.add_done_callback(update)

        self._update_input_summary()

        if data is not None and self.__pending_sort is not None:
            self.__restore_sort()

        if data is not None and self.__pending_selection is not None:
            selection = self.__pending_selection
            self.__pending_selection = None
            rows = selection["rows"]
            columns = selection["columns"]
            self.set_selection(rows, columns)
        self.commit.now()

    def _setup_table_view(self):
        """Setup the view with current input data."""
        if self.input is None:
            self.view.setModel(None)
            return

        datamodel = self.input.model
        view = self.view
        data = self.input.table
        rowcount = data.approx_len()

        if self.color_by_class and data.domain.has_discrete_class:
            color_schema = [
                QColor(*c) for c in data.domain.class_var.colors]
        else:
            color_schema = None
        if self.show_distributions:
            view.setItemDelegate(
                TableBarItemDelegate(
                    view, color=self.dist_color, color_schema=color_schema)
            )
        else:
            view.setItemDelegate(TableDataDelegate(view))

        view.setModel(datamodel)

        vheader = view.verticalHeader()
        option = view.viewOptions()
        size = view.style().sizeFromContents(
            QStyle.CT_ItemViewItem, option,
            QSize(20, 20), view)

        vheader.setDefaultSectionSize(size.height() + 2)
        vheader.setMinimumSectionSize(5)
        vheader.setSectionResizeMode(QHeaderView.Fixed)

        # Limit the number of rows displayed in the QTableView
        # (workaround for QTBUG-18490 / QTBUG-28631)
        maxrows = (2 ** 31 - 1) // (vheader.defaultSectionSize() + 2)
        if rowcount > maxrows:
            sliceproxy = TableSliceProxy(
                parent=view, rowSlice=slice(0, maxrows))
            sliceproxy.setSourceModel(datamodel)
            # First reset the view (without this the header view retains
            # it's state - at this point invalid/broken)
            view.setModel(None)
            view.setModel(sliceproxy)

        assert view.model().rowCount() <= maxrows
        assert vheader.sectionSize(0) > 1 or datamodel.rowCount() == 0

        # update the header (attribute names)
        self._update_variable_labels()
        view.selectionFinished.connect(self.update_selection)

    def _update_input_summary(self):
        def format_summary(summary):
            if isinstance(summary, tsummary.ApproxSummary):
                length = summary.len.result() if summary.len.done() else \
                    summary.approx_len
            elif isinstance(summary, tsummary.Summary):
                length = summary.len
            return length

        summary, details = self.info.NoInput, ""
        if self.input:
            summary = format_summary(self.input.summary)
            details = format_summary_details(self.input.table)
        self.info.set_input_summary(summary, details)

        if self.input is None:
            summary = ["No data."]
        else:
            summary = tsummary.format_summary(self.input.summary)
        self.info_text.setText("\n".join(summary))

    def _update_variable_labels(self):
        """Update the variable labels visibility for current view."""
        if self.input is None:
            return
        model = self.input.model
        if self.show_attribute_labels:
            model.setRichHeaderFlags(
                RichTableModel.Labels | RichTableModel.Name
            )
        else:
            model.setRichHeaderFlags(RichTableModel.Name)

    def _on_distribution_color_changed(self):
        if self.input is None:
            return
        widget = self.view
        model = self.input.model
        data = model.source
        class_var = data.domain.class_var
        if self.color_by_class and class_var and class_var.is_discrete:
            color_schema = [QColor(*c) for c in class_var.colors]
        else:
            color_schema = None
        if self.show_distributions:
            delegate = TableBarItemDelegate(widget, color=self.dist_color,
                                            color_schema=color_schema)
        else:
            delegate = TableDataDelegate(widget)
        widget.setItemDelegate(delegate)

    def _on_select_rows_changed(self):
        if self.input is None:
            return
        selection_model = self.view.selectionModel()
        selection_model.setSelectBlocks(not self.select_rows)
        if self.select_rows:
            self.view.setSelectionBehavior(QTableView.SelectRows)
            # Expand the current selection to full row selection.
            selection_model.select(
                selection_model.selection(),
                QItemSelectionModel.Select | QItemSelectionModel.Rows
            )
        else:
            self.view.setSelectionBehavior(QTableView.SelectItems)

    def restore_order(self):
        """Restore the original data order of the current view."""
        self.view.sortByColumn(-1, Qt.AscendingOrder)
        self.stored_sort = []
        self.Warning.missing_sort_columns.clear()

    @Slot()
    def _update_info(self):
        self._update_input_summary()

    def _on_sort_indicator_changed(self, index: int, order: Qt.SortOrder) -> None:
        if index == -1:
            self.stored_sort = []
        elif self.input is not None:
            model = self.input.model
            var = model.headerData(index, Qt.Horizontal, TableModel.VariableRole)
            order = -1 if order == Qt.DescendingOrder else 1
            # Drop any previously applied sort on this column
            self.stored_sort = [(n, d) for n, d in self.stored_sort
                                if n != var.name]
            self.stored_sort.append((var.name, order))
        self.update_selection()
        self.Warning.missing_sort_columns.clear()

    def set_sort_columns(self, sorting: List[Tuple[str, int]]):
        """
        Set the model sorting parameters.

        Parameters
        ----------
        sorting: List[Tuple[str, int]]
            For each (name: str, inc: int) tuple where `name` is the column
            name and `inc` is 1 for increasing order and -1 for decreasing
            order, the model is sorted by that column.
        """
        if self.input is None:
            return  # pragma: no cover
        self.stored_sort = []
        # Map header names/titles to column indices
        columns = {var.name: i for
                   var, i in self.__header_variable_indices().items()}
        # Suppress the _on_sort_indicator_changed -> commit calls
        with disconnected(self.view.horizontalHeader().sortIndicatorChanged,
                          self._on_sort_indicator_changed, Qt.UniqueConnection):
            for name, order in sorting:
                if name in columns:
                    self.view.sortByColumn(
                        columns[name],
                        Qt.AscendingOrder if order == 1 else Qt.DescendingOrder
                    )
                self.stored_sort.append((name, order))

    def __restore_sort(self) -> None:
        assert self.input is not None
        sort = self.__pending_sort
        self.__pending_sort = None
        if sort is None:
            return  # pragma: no cover
        if not self.view.isSortingEnabled() and sort:
            self.Warning.non_sortable_input()
            self.Warning.missing_sort_columns.clear()
            return
        # Map header names/titles to column indices
        vars_ = self.__header_variable_indices()
        columns = {var.name: i for var, i in vars_.items()}
        missing_columns = []
        sort_ = []
        for name, order in sort:
            if name in columns:
                sort_.append((name, order))
            else:
                missing_columns.append(name)
        self.set_sort_columns(sort_)
        if missing_columns:
            self.Warning.missing_sort_columns(", ".join(missing_columns))

    def __header_variable_indices(self) -> Dict[Variable, int]:
        model = self.view.model()
        if model is None:
            return {}  # pragma: no cover
        vars_ = [model.headerData(i, Qt.Horizontal, TableModel.VariableRole)
                 for i in range(model.columnCount())]
        return {v: i for i, v in enumerate(vars_) if isinstance(v, Variable)}

    def update_selection(self, *_):
        self.commit.deferred()

    def set_selection(self, rows: Sequence[int], columns: Sequence[int]) -> None:
        """
        Set the selected `rows` and `columns`.

        `rows` are indices into underlying :class:`Table`
        """
        self.view.setBlockSelection(rows, columns)

    def get_selection(self):
        """
        Return the selected row and column indices of the selection in view.
        """
        return self.view.blockSelection()

    @gui.deferred
    def commit(self):
        """
        Commit/send the current selected row/column selection.
        """
        selected_data = table = rowsel = None
        if self.input is not None:
            model = self.input.model
            table = self.input.table

            # Selections of individual instances are not implemented
            # for SqlTables
            if isinstance(table, SqlTable):
                self.Outputs.selected_data.send(selected_data)
                self.Outputs.annotated_data.send(None)
                return

            rowsel, colsel = self.get_selection()
            self.stored_selection = {"rows": rowsel, "columns": colsel}

            domain = table.domain

            if len(colsel) < len(domain.variables) + len(domain.metas):
                # only a subset of the columns is selected
                allvars = domain.class_vars + domain.metas + domain.attributes
                columns = [(c, model.headerData(c, Qt.Horizontal,
                                                TableModel.DomainRole))
                           for c in colsel]
                assert all(role is not None for _, role in columns)

                def select_vars(role):
                    """select variables for role (TableModel.DomainRole)"""
                    return [allvars[c] for c, r in columns if r == role]

                attrs = select_vars(TableModel.Attribute)
                if attrs and issparse(table.X):
                    # for sparse data you can only select all attributes
                    attrs = table.domain.attributes
                class_vars = select_vars(TableModel.ClassVar)
                metas = select_vars(TableModel.Meta)
                domain = Orange.data.Domain(attrs, class_vars, metas)

            sortsection = self.view.horizontalHeader().sortIndicatorSection()
            if rowsel:
                selected_data = table.from_table(domain, table, rowsel)
            elif sortsection != -1:
                # Send sorted data
                permutation = model.mapToSourceRows(...)
                selected_data = table.from_table(table.domain, table, permutation)
            else:
                # Send all data by default
                selected_data = table

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(create_annotated_table(table, rowsel))

    def copy(self):
        """
        Copy current table selection to the clipboard.
        """
        if self.input is not None:
            mime = table_selection_to_mime_data(self.view)
            QApplication.clipboard().setMimeData(
                mime, QClipboard.Clipboard
            )

    def send_report(self):
        if self.input is None:
            return
        model = self.input.model
        self.report_data_brief(model.source)
        self.report_table(self.view)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWTable).run(
        input_data=Table("iris"),
    )
