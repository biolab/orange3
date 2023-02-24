import itertools
import concurrent.futures

from dataclasses import dataclass
from typing import Optional, Union

import numpy
from scipy.sparse import issparse

from AnyQt.QtWidgets import QTableView, QHeaderView, QApplication, QStyle
from AnyQt.QtGui import QColor, QClipboard
from AnyQt.QtCore import (
    Qt, QSize, QMetaObject, QAbstractProxyModel, QItemSelectionModel,
    QItemSelection, QItemSelectionRange,
)
from AnyQt.QtCore import Slot

import Orange.data
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable

from Orange.widgets import gui
from Orange.widgets.data.utils.models import RichTableModel, TableSliceProxy
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemdelegates import TableDataDelegate
from Orange.widgets.utils.itemselectionmodel import (
    BlockSelectionModel, ranges, selection_blocks
)
from Orange.widgets.utils.tableview import table_selection_to_mime_data
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.state_summary import format_summary_details
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

    buttons_area_orientation = Qt.Vertical

    show_distributions = Setting(False)
    show_attribute_labels = Setting(True)
    select_rows = Setting(True)
    auto_commit = Setting(True)

    color_by_class = Setting(True)
    selected_rows = Setting([], schema_only=True)
    selected_cols = Setting([], schema_only=True)

    settings_version = 1

    def __init__(self):
        super().__init__()
        self.input: Optional[InputData] = None
        self.__pending_selected_rows = self.selected_rows
        self.selected_rows = None
        self.__pending_selected_cols = self.selected_cols
        self.selected_cols = None
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
        header.sortIndicatorChanged.connect(self.update_selection)

        # QHeaderView does not 'reset' the model sort column,
        # because there is no guaranty (requirement) that the
        # models understand the -1 sort column.
        def sort_reset(index, order):
            if view.model() is not None and index == -1:
                view.model().sort(index, order)

        header.sortIndicatorChanged.connect(sort_reset)
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
        data: Optional[Table] = self.input.table if self.input else None
        slot = self.input
        if slot is not None and isinstance(slot.summary.len, concurrent.futures.Future):
            def update(_):
                QMetaObject.invokeMethod(
                    self, "_update_info", Qt.QueuedConnection)
            slot.summary.len.add_done_callback(update)

        self._update_input_summary()

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

        # Enable/disable view sorting based on data's type
        view.setSortingEnabled(is_sortable(data))
        header = view.horizontalHeader()
        header.setSectionsClickable(is_sortable(data))
        header.setSortIndicatorShown(is_sortable(data))

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

        selmodel = BlockSelectionModel(
            view.model(), parent=view, selectBlocks=not self.select_rows)
        view.setSelectionModel(selmodel)
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
        self.view.reset()

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
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

    @Slot()
    def _update_info(self):
        self._update_input_summary()

    def update_selection(self, *_):
        self.commit.deferred()

    def set_selection(self):
        if self.selected_rows and self.selected_cols:
            view = self.view
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

    @gui.deferred
    def commit(self):
        """
        Commit/send the current selected row/column selection.
        """
        selected_data = table = rowsel = None
        view = self.view
        if self.input is not None:
            model = self.input.model
            table = self.input.table

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


def is_sortable(table):
    if isinstance(table, SqlTable):
        return False
    elif isinstance(table, Orange.data.Table):
        return True
    else:
        return False


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWTable).run(
        input_data=Table("iris"),
    )
