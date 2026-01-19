import itertools
import concurrent.futures

from collections import namedtuple
from typing import List, Optional

import numpy
from scipy.sparse import issparse

from AnyQt.QtWidgets import QTableView, QHeaderView, QApplication, QStyle
from AnyQt.QtGui import QColor, QClipboard
from AnyQt.QtCore import (
    Qt, QSize, QMetaObject,
    QAbstractProxyModel,
    QItemSelectionModel, QItemSelection, QItemSelectionRange,
)
from AnyQt.QtCore import pyqtSlot as Slot

import Orange.data
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable

from Orange.widgets import gui
from Orange.widgets.data.utils.tableview import RichTableView
from Orange.widgets.settings import Setting
from Orange.widgets.data.utils.models import TableSliceProxy, RichTableModel
from Orange.widgets.utils.itemdelegates import TableDataDelegate
from Orange.widgets.utils.itemselectionmodel import (
    BlockSelectionModel, ranges, selection_blocks
)
from Orange.widgets.utils.tableview import table_selection_to_mime_data
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, MultiInput, Output, Msg
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.data.utils import tablesummary as tsummary


TableSlot = namedtuple("TableSlot", ["input_id", "table", "summary", "view"])


class DataTableView(gui.HScrollStepMixin, RichTableView):
    dataset: Table
    input_slot: TableSlot


class TableBarItemDelegate(gui.TableBarItem, TableDataDelegate):
    pass


class OWDataTable(OWWidget):
    category = "Orange Obsolete"
    replaces = ["Orange.widgets.data.owtable.OWDataTable"]

    name = "Data Table"
    description = "View the dataset in a spreadsheet."
    icon = "../data/icons/Table.svg"
    priority = 50
    keywords = "_keywords"

    class Inputs:
        data = MultiInput("Data", Table, auto_summary=False, filter_none=True)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Warning(OWWidget.Warning):
        multiple_inputs = Msg(
            "Multiple Data inputs are deprecated.\n"
            "This functionality will be removed soon.\n"
            "Use multiple Tables instead.")

    buttons_area_orientation = Qt.Vertical

    show_distributions = Setting(False)
    show_attribute_labels = Setting(True)
    select_rows = Setting(True)
    auto_commit = Setting(True)

    color_by_class = Setting(True)
    selected_rows = Setting([], schema_only=True)
    selected_cols = Setting([], schema_only=True)

    settings_version = 2

    def __init__(self):
        super().__init__()
        self._inputs: List[TableSlot] = []
        self.__pending_selected_rows = self.selected_rows
        self.selected_rows = None
        self.__pending_selected_cols = self.selected_cols
        self.selected_cols = None

        self.dist_color = QColor(220, 220, 220, 255)

        info_box = gui.vBox(self.controlArea, "Info")
        self.info_text = gui.widgetLabel(info_box)
        self._set_input_summary(None)

        box = gui.vBox(self.controlArea, "Variables")
        self.c_show_attribute_labels = gui.checkBox(
            box, self, "show_attribute_labels",
            "Show variable labels (if present)",
            callback=self._on_show_variable_labels_changed)

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

        # GUI with tabs
        self.tabs = gui.tabWidget(self.mainArea)
        self.tabs.currentChanged.connect(self._on_current_tab_changed)

    def copy_to_clipboard(self):
        self.copy()

    def sizeHint(self):
        return QSize(800, 500)

    def _create_table_view(self):
        view = DataTableView()
        view.setSortingEnabled(True)
        view.setItemDelegate(TableDataDelegate(view))

        if self.select_rows:
            view.setSelectionBehavior(QTableView.SelectRows)

        header = view.horizontalHeader()
        header.setSectionsMovable(True)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, Qt.AscendingOrder)

        # QHeaderView does not 'reset' the model sort column,
        # because there is no guaranty (requirement) that the
        # models understand the -1 sort column.
        def sort_reset(index, order):
            if view.model() is not None and index == -1:
                view.model().sort(index, order)
        header.sortIndicatorChanged.connect(sort_reset)
        return view

    @Inputs.data
    def set_dataset(self, index: int, data: Table):
        """Set the input dataset."""
        datasetname = getattr(data, "name", "Data")
        slot = self._inputs[index]
        view = slot.view
        # reset the (header) view state.
        view.setModel(None)
        view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        assert self.tabs.indexOf(view) != -1
        self.tabs.setTabText(self.tabs.indexOf(view), datasetname)
        view.dataset = data
        slot = TableSlot(index, data, tsummary.table_summary(data), view)
        view.input_slot = slot
        self._inputs[index] = slot
        self._setup_table_view(view, data)
        self.tabs.setCurrentWidget(view)
        self._set_multi_input_warning()

    @Inputs.data.insert
    def insert_dataset(self, index: int, data: Table):
        datasetname = getattr(data, "name", "Data")
        view = self._create_table_view()
        slot = TableSlot(None, data, tsummary.table_summary(data), view)
        view.dataset = data
        view.input_slot = slot
        self._inputs.insert(index, slot)
        self.tabs.insertTab(index, view, datasetname)
        self._setup_table_view(view, data)
        self.tabs.setCurrentWidget(view)
        self._set_multi_input_warning()

    @Inputs.data.remove
    def remove_dataset(self, index):
        slot = self._inputs.pop(index)
        view = slot.view
        self.tabs.removeTab(self.tabs.indexOf(view))
        view.setModel(None)
        view.hide()
        view.deleteLater()

        current = self.tabs.currentWidget()
        if current is not None:
            self._set_input_summary(current.input_slot)
        self._set_multi_input_warning()

    def _set_multi_input_warning(self):
        self.Warning.multiple_inputs(shown=len(self._inputs) > 1)

    def handleNewSignals(self):
        super().handleNewSignals()
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)
        data: Optional[Table] = None
        current = self.tabs.currentWidget()
        slot = None
        if current is not None:
            data = current.dataset
            slot = current.input_slot

        if slot and isinstance(slot.summary.len, concurrent.futures.Future):
            def update(_):
                QMetaObject.invokeMethod(
                    self, "_update_info", Qt.QueuedConnection)
            slot.summary.len.add_done_callback(update)
        self._set_input_summary(slot)

        if data is not None and self.__pending_selected_rows is not None:
            self.selected_rows = self.__pending_selected_rows
            self.__pending_selected_rows = None
        else:
            self.selected_rows = []

        if data and self.__pending_selected_cols is not None:
            self.selected_cols = self.__pending_selected_cols
            self.__pending_selected_cols = None
        else:
            self.selected_cols = []

        self.set_selection()
        self.commit.now()

    def _setup_table_view(self, view, data):
        """Setup the `view` (QTableView) with `data` (Orange.data.Table)
        """
        datamodel = RichTableModel(data)
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

        header = view.horizontalHeader()
        header.sortIndicatorChanged.connect(self.update_selection)

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
        self._update_variable_labels(view)

        selmodel = BlockSelectionModel(
            view.model(), parent=view, selectBlocks=not self.select_rows)
        view.setSelectionModel(selmodel)
        view.selectionFinished.connect(self.update_selection)

    def _set_input_summary(self, slot):
        def format_summary(summary):
            if isinstance(summary, tsummary.ApproxSummary):
                length = summary.len.result() if summary.len.done() else \
                    summary.approx_len
            elif isinstance(summary, tsummary.Summary):
                length = summary.len
            return length

        summary, details = self.info.NoInput, ""
        if slot:
            summary = format_summary(slot.summary)
            details = format_summary_details(slot.table)
        self.info.set_input_summary(summary, details)
        if slot is None:
            summary = ["No data."]
        else:
            summary = tsummary.format_summary(slot.summary)
        self.info_text.setText("\n".join(summary))

    def _on_current_tab_changed(self, index):
        """Update the status bar on current tab change"""
        view = self.tabs.widget(index)
        if view is not None and view.model() is not None:
            self._set_input_summary(view.input_slot)
            self.update_selection()
        else:
            self._set_input_summary(None)

    def _update_variable_labels(self, view):
        "Update the variable labels visibility for `view`"
        model = view.model()
        if isinstance(model, TableSliceProxy):
            model = model.sourceModel()

        if self.show_attribute_labels:
            model.setRichHeaderFlags(
                RichTableModel.Labels | RichTableModel.Name)
        else:
            model.setRichHeaderFlags(RichTableModel.Name)

    def _on_show_variable_labels_changed(self):
        """The variable labels (var.attribues) visibility was changed."""
        for slot in self._inputs:
            self._update_variable_labels(slot.view)

    def _on_distribution_color_changed(self):
        for ti in range(self.tabs.count()):
            widget = self.tabs.widget(ti)
            model = widget.model()
            while isinstance(model, QAbstractProxyModel):
                model = model.sourceModel()
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
        tab = self.tabs.currentWidget()
        if tab:
            tab.reset()

    def _on_select_rows_changed(self):
        for slot in self._inputs:
            selection_model = slot.view.selectionModel()
            selection_model.setSelectBlocks(not self.select_rows)
            if self.select_rows:
                slot.view.setSelectionBehavior(QTableView.SelectRows)
                # Expand the current selection to full row selection.
                selection_model.select(
                    selection_model.selection(),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows
                )
            else:
                slot.view.setSelectionBehavior(QTableView.SelectItems)

    def restore_order(self):
        """Restore the original data order of the current view."""
        table = self.tabs.currentWidget()
        if table is not None:
            table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

    @Slot()
    def _update_info(self):
        current = self.tabs.currentWidget()
        if current is not None and current.model() is not None:
            self._set_input_summary(current.input_slot)

    def update_selection(self, *_):
        self.commit.deferred()

    def set_selection(self):
        if self.selected_rows and self.selected_cols:
            view = self.tabs.currentWidget()
            model = view.model()
            if model.rowCount() <= self.selected_rows[-1] or \
                    model.columnCount() <= self.selected_cols[-1]:
                return

            selection = QItemSelection()
            rowranges = list(ranges(self.selected_rows))
            colranges = list(ranges(self.selected_cols))

            for rowstart, rowend in rowranges:
                for colstart, colend in colranges:
                    selection.append(
                        QItemSelectionRange(
                            view.model().index(rowstart, colstart),
                            view.model().index(rowend - 1, colend - 1)
                        )
                    )
            view.selectionModel().select(
                selection, QItemSelectionModel.ClearAndSelect)

    @staticmethod
    def get_selection(view):
        """
        Return the selected row and column indices of the selection in view.
        """
        selmodel = view.selectionModel()

        selection = selmodel.selection()
        model = view.model()
        # map through the proxies into input table.
        while isinstance(model, QAbstractProxyModel):
            selection = model.mapSelectionToSource(selection)
            model = model.sourceModel()

        assert isinstance(selmodel, BlockSelectionModel)
        assert isinstance(model, TableModel)

        row_spans, col_spans = selection_blocks(selection)
        rows = list(itertools.chain.from_iterable(itertools.starmap(range, row_spans)))
        cols = list(itertools.chain.from_iterable(itertools.starmap(range, col_spans)))
        rows = numpy.array(rows, dtype=numpy.intp)
        # map the rows through the applied sorting (if any)
        rows = model.mapToSourceRows(rows)
        rows = rows.tolist()
        return rows, cols

    @staticmethod
    def _get_model(view):
        model = view.model()
        while isinstance(model, QAbstractProxyModel):
            model = model.sourceModel()
        return model

    @gui.deferred
    def commit(self):
        """
        Commit/send the current selected row/column selection.
        """
        selected_data = table = rowsel = None
        view = self.tabs.currentWidget()
        if view and view.model() is not None:
            model = self._get_model(view)
            table = model.source  # The input data table

            # Selections of individual instances are not implemented
            # for SqlTables
            if isinstance(table, SqlTable):
                self.Outputs.selected_data.send(selected_data)
                self.Outputs.annotated_data.send(None)
                return

            rowsel, colsel = self.get_selection(view)
            self.selected_rows, self.selected_cols = rowsel, colsel

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

            # Send all data by default
            if not rowsel:
                selected_data = table
            else:
                selected_data = table.from_table(domain, table, rowsel)

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(create_annotated_table(table, rowsel))

    def copy(self):
        """
        Copy current table selection to the clipboard.
        """
        view = self.tabs.currentWidget()
        if view is not None:
            mime = table_selection_to_mime_data(view)
            QApplication.clipboard().setMimeData(
                mime, QClipboard.Clipboard
            )

    def send_report(self):
        view = self.tabs.currentWidget()
        if not view or not view.model():
            return
        model = self._get_model(view)
        self.report_data_brief(model.source)
        self.report_table(view)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDataTable).run(
        insert_dataset=[
            (0, Table("iris")),
            (1, Table("brown-selected")),
            (2, Table("housing"))
        ])
