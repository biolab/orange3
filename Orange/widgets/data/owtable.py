import concurrent.futures
from dataclasses import dataclass
from typing import (
    Optional, Union, Sequence, List, TypedDict, Tuple, Any, Container
)

from scipy.sparse import issparse

from AnyQt.QtWidgets import (
    QTableView, QHeaderView, QApplication, QStyle, QStyleOptionHeader,
    QStyleOptionViewItem
)
from AnyQt.QtGui import QColor, QClipboard, QPainter
from AnyQt.QtCore import (
    Qt, QSize, QMetaObject, QItemSelectionModel, QModelIndex, QRect
)
from AnyQt.QtCore import Slot

from orangewidget.gui import OrangeUserRole

import Orange.data
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
from Orange.widgets.utils.headerview import HeaderView
from Orange.widgets.data.utils.tableview import RichTableView
from Orange.widgets.data.utils import tablesummary as tsummary


SubsetRole = next(OrangeUserRole)


class HeaderViewWithSubsetIndicator(HeaderView):
    _IndicatorChar = "\N{BULLET}"

    def paintSection(
            self, painter: QPainter, rect: QRect, logicalIndex: int
    ) -> None:
        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        self.initStyleOptionForIndex(opt, logicalIndex)
        model = self.model()
        if model is None:
            return  # pragma: no cover
        opt.rect = rect
        issubset = model.headerData(logicalIndex, Qt.Vertical, SubsetRole)
        style = self.style()
        # draw background
        style.drawControl(QStyle.CE_HeaderSection, opt, painter, self)
        indicator_rect = QRect(rect)
        text_rect = QRect(rect)
        indicator_width = opt.fontMetrics.horizontalAdvance(
            self._IndicatorChar + " "
        )
        indicator_rect.setWidth(indicator_width)
        text_rect.setLeft(indicator_width)
        if issubset:
            optindicator = QStyleOptionHeader(opt)
            optindicator.rect = indicator_rect
            optindicator.textAlignment = Qt.AlignCenter
            optindicator.text = self._IndicatorChar
            # draw subset indicator
            style.drawControl(QStyle.CE_HeaderLabel, optindicator, painter, self)
        opt.rect = text_rect
        # draw section label
        style.drawControl(QStyle.CE_HeaderLabel, opt, painter, self)

    def sectionSizeFromContents(self, logicalIndex: int) -> QSize:
        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        super().initStyleOptionForIndex(opt, logicalIndex)
        opt.text = self._IndicatorChar + " " + opt.text
        return self.style().sizeFromContents(QStyle.CT_HeaderSection, opt, QSize(), self)


class DataTableView(gui.HScrollStepMixin, RichTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vheader = HeaderViewWithSubsetIndicator(
            Qt.Vertical, self, highlightSections=True
        )
        vheader.setSectionsClickable(True)
        self.setVerticalHeader(vheader)


class _TableDataDelegate(TableDataDelegate):
    DefaultRoles = TableDataDelegate.DefaultRoles + (SubsetRole,)


class SubsetTableDataDelegate(_TableDataDelegate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset_opacity = 0.5

    def paint(
            self, painter: QPainter, option: QStyleOptionViewItem,
            index: QModelIndex
    ) -> None:
        issubset = self.cachedData(index, SubsetRole)
        opacity = painter.opacity()
        if not issubset:
            painter.setOpacity(self.subset_opacity)
        super().paint(painter, option, index)
        if not issubset:
            painter.setOpacity(opacity)


class TableBarItemDelegate(SubsetTableDataDelegate, gui.TableBarItem,
                           _TableDataDelegate):
    pass


class _TableModel(RichTableModel):
    SubsetRole = SubsetRole

    def __init__(self, *args, subsets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._subset = subsets or set()

    def setSubsetRowIds(self, subsetids: Container[int]):
        self._subset = subsetids
        if self.rowCount():
            self.headerDataChanged.emit(Qt.Vertical, 0, self.rowCount() - 1)
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
                [SubsetRole],
            )

    def _is_subset(self, row):
        row = self.mapToSourceRows(row)
        try:
            id_ = self.source.ids[row]
        except (IndexError, AttributeError):  # pragma: no cover
            return False
        return int(id_) in self._subset

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> Any:
        if role == _TableModel.SubsetRole:
            return self._is_subset(index.row())
        return super().data(index, role)

    def headerData(self, section, orientation, role):
        if orientation == Qt.Vertical and role == _TableModel.SubsetRole:
            return self._is_subset(section)
        return super().headerData(section, orientation, role)


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
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

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
        self._subset_ids: Optional[set] = None
        self.__pending_selection: Optional[_Selection] = self.stored_selection
        self.__pending_sort: Optional[_Sorting] = self.stored_sort
        self.__have_new_data = False
        self.__have_new_subset = False
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
        self.clear_button = gui.button(
            box, self, "Clear Selection", callback=self.clear_selection,
            autoDefault=False)
        gui.checkBox(box, self, "select_rows", "Select full rows",
                     callback=self._on_select_rows_changed)

        gui.rubber(self.controlArea)

        self.restore_button = gui.button(
            self.buttonsArea, self, "Restore Original Order",
            callback=self.restore_order,
            tooltip="Show rows in the original order",
            autoDefault=False,
            attribute=Qt.WA_LayoutUsesWidgetRect)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        view = DataTableView(sortingEnabled=True)
        view.setItemDelegate(SubsetTableDataDelegate(view))
        view.selectionFinished.connect(self.update_selection)

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
        if data is not None:
            summary = tsummary.table_summary(data)
            self.input = InputData(
                table=data,
                summary=summary,
                model=_TableModel(data)
            )
            if isinstance(summary.len, concurrent.futures.Future):
                def update(_):
                    QMetaObject.invokeMethod(
                        self, "_update_info", Qt.QueuedConnection)
                summary.len.add_done_callback(update)
        else:
            self.input = None
        self.__have_new_data = True

    @Inputs.data_subset
    def set_subset_dataset(self, subset: Optional[Table]):
        """Set the data subset"""
        if subset is not None and not isinstance(subset, SqlTable):
            ids = set(subset.ids)
        else:
            ids = None
        self._subset_ids = ids
        self.__have_new_subset = True

    def handleNewSignals(self):
        self.restore_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        super().handleNewSignals()
        self.Warning.non_sortable_input.clear()
        self.Warning.missing_sort_columns.clear()
        data: Optional[Table] = self.input.table if self.input else None
        model = self.input.model if self.input else None

        if self.__have_new_data:
            self._setup_table_view()
            self._update_input_summary()

            if data is not None and self.__pending_sort is not None:
                self.__restore_sort()

            if data is not None and self.__pending_selection is not None:
                selection = self.__pending_selection
                self.__pending_selection = None
                rows = selection["rows"]
                columns = selection["columns"]
                self.set_selection(rows, columns)

        if self.__have_new_subset and model is not None:
            model.setSubsetRowIds(self._subset_ids or set())
            self.__have_new_subset = False

        self._setup_view_delegate()

        if self.__have_new_data:
            self.commit.now()
            self.__have_new_data = False

    def _setup_table_view(self):
        """Setup the view with current input data."""
        if self.input is None:
            self.view.setModel(None)
            return

        datamodel = self.input.model
        datamodel.setSubsetRowIds(self._subset_ids or set())

        view = self.view
        data = self.input.table
        rowcount = data.approx_len()
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

        self._setup_view_delegate()
        # update the header (attribute names)
        self._update_variable_labels()

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
            return  # pragma: no cover
        self._setup_view_delegate()

    def _setup_view_delegate(self):
        if self.input is None:
            return
        model = self.input.model
        data = model.source
        class_var = data.domain.class_var
        if self.color_by_class and class_var and class_var.is_discrete:
            color_schema = [QColor(*c) for c in class_var.colors]
        else:
            color_schema = None
        if self.show_distributions:
            delegate = TableBarItemDelegate(
                self.view, color=self.dist_color, color_schema=color_schema
            )
        else:
            delegate = SubsetTableDataDelegate(self.view)
        delegate.subset_opacity = 0.5 if self._subset_ids is not None else 1.0
        self.view.setItemDelegate(delegate)

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
        self.restore_button.setEnabled(index != -1)
        if index == -1:
            self.stored_sort = []
        elif self.input is not None:
            model = self.input.model
            coldesc = model.columns[index]
            colid = self.__encode_column_id(coldesc)
            order = -1 if order == Qt.DescendingOrder else 1
            # Drop any previously applied sort on this column
            self.stored_sort = [(n, d) for n, d in self.stored_sort
                                if n != colid]
            self.stored_sort.append((colid, order))
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
        # Map header ids (names) to column indices
        columns = {id_: i for i, id_ in enumerate(self.__header_ids())}
        # Suppress the _on_sort_indicator_changed -> commit calls
        with disconnected(self.view.horizontalHeader().sortIndicatorChanged,
                          self._on_sort_indicator_changed, Qt.UniqueConnection):
            for colid, order in sorting:
                if colid in columns:
                    self.view.sortByColumn(
                        columns[colid],
                        Qt.AscendingOrder if order == 1 else Qt.DescendingOrder
                    )
                self.stored_sort.append((colid, order))

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
        # Map header ids (names) to column indices
        columns = {id_: i for i, id_ in enumerate(self.__header_ids())}
        missing_columns = []
        sort_ = []
        for colid, order in sort:
            if colid in columns:
                sort_.append((colid, order))
            else:
                missing_columns.append(self.__decode_column_id(colid))
        self.set_sort_columns(sort_)
        self.restore_button.setEnabled(True)
        if missing_columns:
            self.Warning.missing_sort_columns(", ".join(missing_columns))

    @staticmethod
    def __encode_column_id(
            coldesc: Union[TableModel.Column, TableModel.Basket]
    ) -> str:
        def escape(s: str) -> str:  # escape possible leading slash
            if s.startswith("\\"):
                return "\\" + s
            return s
        if isinstance(coldesc, TableModel.Column):
            return escape(coldesc.var.name)
        else:
            lookup = ("TARGET", "META", "FEATURES",)
            return f"\\BASKET({lookup[coldesc.role]})"

    @staticmethod
    def __decode_column_id(cid: str) -> str:
        if cid.startswith("\\"):
            return cid[1:]
        return cid

    def __header_ids(self) -> List[str]:
        if self.input is None:
            return []
        return [self.__encode_column_id(c) for c in self.input.model.columns]

    def update_selection(self, *_):
        # Calling get_selection is expensive, so we consult selectionModel directly
        sel_model = self.view.selectionModel()
        selection = sel_model.selection()
        self.clear_button.setEnabled(not selection.isEmpty())
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

    def clear_selection(self):
        self.set_selection([], [])

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
