import sys
import threading
import itertools
import concurrent.futures

from collections import namedtuple
from typing import List, Optional

from math import isnan

import numpy
from scipy.sparse import issparse

from AnyQt.QtWidgets import (
    QTableView, QHeaderView, QAbstractButton, QApplication, QStyleOptionHeader,
    QStyle, QStylePainter
)
from AnyQt.QtGui import QColor, QClipboard
from AnyQt.QtCore import (
    Qt, QSize, QEvent, QObject, QMetaObject,
    QAbstractProxyModel, QIdentityProxyModel, QModelIndex,
    QItemSelectionModel, QItemSelection, QItemSelectionRange,
)
from AnyQt.QtCore import pyqtSlot as Slot

import Orange.data
from Orange.data.storage import Storage
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import basic_stats

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemdelegates import TableDataDelegate
from Orange.widgets.utils.itemselectionmodel import (
    BlockSelectionModel, ranges, selection_blocks
)
from Orange.widgets.utils.tableview import TableView, \
    table_selection_to_mime_data
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, MultiInput, Output
from Orange.widgets.utils import datacaching
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.state_summary import format_summary_details


class RichTableModel(TableModel):
    """A TableModel with some extra bells and whistles/

    (adds support for gui.BarRole, include variable labels and icons
    in the header)
    """
    #: Rich header data flags.
    Name, Labels, Icon = 1, 2, 4

    def __init__(self, sourcedata, parent=None):
        super().__init__(sourcedata, parent)

        self._header_flags = RichTableModel.Name
        self._continuous = [var.is_continuous for var in self.vars]
        labels = []
        for var in self.vars:
            if isinstance(var, Orange.data.Variable):
                labels.extend(var.attributes.keys())
        self._labels = list(sorted(
            {label for label in labels if not label.startswith("_")}))

    def data(self, index, role=Qt.DisplayRole,
             # for faster local lookup
             _BarRole=gui.TableBarItem.BarRole):
        # pylint: disable=arguments-differ
        if role == _BarRole and self._continuous[index.column()]:
            val = super().data(index, TableModel.ValueRole)
            if val is None or isnan(val):
                return None

            dist = super().data(index, TableModel.VariableStatsRole)
            if dist is not None and dist.max > dist.min:
                return (val - dist.min) / (dist.max - dist.min)
            else:
                return None
        elif role == Qt.TextAlignmentRole and self._continuous[index.column()]:
            return Qt.AlignRight | Qt.AlignVCenter
        else:
            return super().data(index, role)

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is None:
                return super().headerData(
                    section, orientation, Qt.DisplayRole)

            lines = []
            if self._header_flags & RichTableModel.Name:
                lines.append(var.name)
            if self._header_flags & RichTableModel.Labels:
                lines.extend(str(var.attributes.get(label, ""))
                             for label in self._labels)
            return "\n".join(lines)
        elif orientation == Qt.Horizontal and role == Qt.DecorationRole and \
                self._header_flags & RichTableModel.Icon:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is not None:
                return gui.attributeIconDict[var]
            else:
                return None
        else:
            return super().headerData(section, orientation, role)

    def setRichHeaderFlags(self, flags):
        if flags != self._header_flags:
            self._header_flags = flags
            self.headerDataChanged.emit(
                Qt.Horizontal, 0, self.columnCount() - 1)

    def richHeaderFlags(self):
        return self._header_flags


class TableSliceProxy(QIdentityProxyModel):
    def __init__(self, parent=None, rowSlice=slice(0, -1), **kwargs):
        super().__init__(parent, **kwargs)
        self.__rowslice = rowSlice

    def setRowSlice(self, rowslice):
        if rowslice.step is not None and rowslice.step != 1:
            raise ValueError("invalid stride")

        if self.__rowslice != rowslice:
            self.beginResetModel()
            self.__rowslice = rowslice
            self.endResetModel()

    def mapToSource(self, proxyindex):
        model = self.sourceModel()
        if model is None or not proxyindex.isValid():
            return QModelIndex()

        row, col = proxyindex.row(), proxyindex.column()
        row = row + self.__rowslice.start
        assert 0 <= row < model.rowCount()
        return model.createIndex(row, col, proxyindex.internalPointer())

    def mapFromSource(self, sourceindex):
        model = self.sourceModel()
        if model is None or not sourceindex.isValid():
            return QModelIndex()
        row, col = sourceindex.row(), sourceindex.column()
        row = row - self.__rowslice.start
        assert 0 <= row < self.rowCount()
        return self.createIndex(row, col, sourceindex.internalPointer())

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        count = super().rowCount()
        start, stop, step = self.__rowslice.indices(count)
        assert step == 1
        return stop - start


TableSlot = namedtuple("TableSlot", ["input_id", "table", "summary", "view"])


class DataTableView(gui.HScrollStepMixin, TableView):
    dataset: Table
    input_slot: TableSlot


class TableBarItemDelegate(gui.TableBarItem, TableDataDelegate):
    pass


class OWDataTable(OWWidget):
    name = "Data Table"
    description = "View the dataset in a spreadsheet."
    icon = "icons/Table.svg"
    priority = 50
    keywords = []

    class Inputs:
        data = MultiInput("Data", Table, auto_summary=False, filter_none=True)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    buttons_area_orientation = Qt.Vertical

    show_distributions = Setting(False)
    dist_color_RGB = Setting((220, 220, 220, 255))
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

        self.dist_color = QColor(*self.dist_color_RGB)

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

    @staticmethod
    def sizeHint():
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
        slot = TableSlot(index, data, table_summary(data), view)
        view.input_slot = slot
        self._inputs[index] = slot
        self._setup_table_view(view, data)
        self.tabs.setCurrentWidget(view)

    @Inputs.data.insert
    def insert_dataset(self, index: int, data: Table):
        datasetname = getattr(data, "name", "Data")
        view = self._create_table_view()
        slot = TableSlot(None, data, table_summary(data), view)
        view.dataset = data
        view.input_slot = slot
        self._inputs.insert(index, slot)
        self.tabs.insertTab(index, view, datasetname)
        self._setup_table_view(view, data)
        self.tabs.setCurrentWidget(view)

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

        # Enable/disable view sorting based on data's type
        view.setSortingEnabled(is_sortable(data))
        header = view.horizontalHeader()
        header.setSectionsClickable(is_sortable(data))
        header.setSortIndicatorShown(is_sortable(data))
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

    #noinspection PyBroadException
    def set_corner_text(self, table, text):
        """Set table corner text."""
        # As this is an ugly hack, do everything in
        # try - except blocks, as it may stop working in newer Qt.
        # pylint: disable=broad-except
        if not hasattr(table, "btn") and not hasattr(table, "btnfailed"):
            try:
                btn = table.findChild(QAbstractButton)

                class Efc(QObject):
                    @staticmethod
                    def eventFilter(o, e):
                        if (isinstance(o, QAbstractButton) and
                                e.type() == QEvent.Paint):
                            # paint by hand (borrowed from QTableCornerButton)
                            btn = o
                            opt = QStyleOptionHeader()
                            opt.initFrom(btn)
                            state = QStyle.State_None
                            if btn.isEnabled():
                                state |= QStyle.State_Enabled
                            if btn.isActiveWindow():
                                state |= QStyle.State_Active
                            if btn.isDown():
                                state |= QStyle.State_Sunken
                            opt.state = state
                            opt.rect = btn.rect()
                            opt.text = btn.text()
                            opt.position = QStyleOptionHeader.OnlyOneSection
                            painter = QStylePainter(btn)
                            painter.drawControl(QStyle.CE_Header, opt)
                            return True     # eat event
                        return False
                table.efc = Efc()
                # disconnect default handler for clicks and connect a new one, which supports
                # both selection and deselection of all data
                btn.clicked.disconnect()
                btn.installEventFilter(table.efc)
                btn.clicked.connect(self._on_select_all)
                table.btn = btn

                if sys.platform == "darwin":
                    btn.setAttribute(Qt.WA_MacSmallSize)

            except Exception:
                table.btnfailed = True

        if hasattr(table, "btn"):
            try:
                btn = table.btn
                btn.setText(text)
                opt = QStyleOptionHeader()
                opt.text = btn.text()
                s = btn.style().sizeFromContents(
                    QStyle.CT_HeaderSection,
                    opt, QSize(),
                    btn)
                if s.isValid():
                    table.verticalHeader().setMinimumWidth(s.width())
            except Exception:
                pass

    def _set_input_summary(self, slot):
        def format_summary(summary):
            if isinstance(summary, ApproxSummary):
                length = summary.len.result() if summary.len.done() else \
                    summary.approx_len
            elif isinstance(summary, Summary):
                length = summary.len
            return length

        summary, details = self.info.NoInput, ""
        if slot:
            summary = format_summary(slot.summary)
            details = format_summary_details(slot.table)
        self.info.set_input_summary(summary, details)

        self.info_text.setText("\n".join(self._info_box_text(slot)))

    @staticmethod
    def _info_box_text(slot):
        def format_part(part):
            if isinstance(part, DenseArray):
                if not part.nans:
                    return ""
                perc = 100 * part.nans / (part.nans + part.non_nans)
                return f" ({perc:.1f} % missing data)"

            if isinstance(part, SparseArray):
                tag = "sparse"
            elif isinstance(part, SparseBoolArray):
                tag = "tags"
            else:  # isinstance(part, NotAvailable)
                return ""
            dens = 100 * part.non_nans / (part.nans + part.non_nans)
            return f" ({tag}, density {dens:.2f} %)"

        def desc(n, part):
            if n == 0:
                return f"No {part}s"
            elif n == 1:
                return f"1 {part}"
            else:
                return f"{n} {part}s"

        if slot is None:
            return ["No data."]
        summary = slot.summary
        text = []
        if isinstance(summary, ApproxSummary):
            if summary.len.done():
                text.append(f"{summary.len.result()} instances")
            else:
                text.append(f"~{summary.approx_len} instances")
        elif isinstance(summary, Summary):
            text.append(f"{summary.len} instances")
            if sum(p.nans for p in [summary.X, summary.Y, summary.M]) == 0:
                text[-1] += " (no missing data)"

        text.append(desc(len(summary.domain.attributes), "feature")
                    + format_part(summary.X))

        if not summary.domain.class_vars:
            text.append("No target variable.")
        else:
            if len(summary.domain.class_vars) > 1:
                c_text = desc(len(summary.domain.class_vars), "outcome")
            elif summary.domain.has_continuous_class:
                c_text = "Numeric outcome"
            else:
                c_text = "Target with " \
                    + desc(len(summary.domain.class_var.values), "value")
            text.append(c_text + format_part(summary.Y))

        text.append(desc(len(summary.domain.metas), "meta attribute")
                    + format_part(summary.M))
        return text

    def _on_select_all(self, _):
        data_info = self.tabs.currentWidget().input_slot.summary
        if len(self.selected_rows) == data_info.len \
                and len(self.selected_cols) == len(data_info.domain.variables):
            self.tabs.currentWidget().clearSelection()
        else:
            self.tabs.currentWidget().selectAll()

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

            labelnames = set()
            domain = model.source.domain
            for a in itertools.chain(domain.metas, domain.variables):
                labelnames.update(a.attributes.keys())
            labelnames = sorted(
                [label for label in labelnames if not label.startswith("_")])
            self.set_corner_text(view, "\n".join([""] + labelnames))
        else:
            model.setRichHeaderFlags(RichTableModel.Name)
            self.set_corner_text(view, "")

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


# Table Summary

# Basic statistics for X/Y/metas arrays
DenseArray = namedtuple(
    "DenseArray", ["nans", "non_nans", "stats"])
SparseArray = namedtuple(
    "SparseArray", ["nans", "non_nans", "stats"])
SparseBoolArray = namedtuple(
    "SparseBoolArray", ["nans", "non_nans", "stats"])
NotAvailable = namedtuple("NotAvailable", [])

#: Orange.data.Table summary
Summary = namedtuple(
    "Summary",
    ["len", "domain", "X", "Y", "M"])

#: Orange.data.sql.table.SqlTable summary
ApproxSummary = namedtuple(
    "ApproxSummary",
    ["approx_len", "len", "domain", "X", "Y", "M"])


def table_summary(table):
    if isinstance(table, SqlTable):
        approx_len = table.approx_len()
        len_future = concurrent.futures.Future()

        def _len():
            len_future.set_result(len(table))
        threading.Thread(target=_len).start()  # KILL ME !!!

        return ApproxSummary(approx_len, len_future, table.domain,
                             NotAvailable(), NotAvailable(), NotAvailable())
    else:
        domain = table.domain
        n_instances = len(table)
        # dist = basic_stats.DomainBasicStats(table, include_metas=True)
        bstats = datacaching.getCached(
            table, basic_stats.DomainBasicStats, (table, True)
        )

        dist = bstats.stats
        # pylint: disable=unbalanced-tuple-unpacking
        X_dist, Y_dist, M_dist = numpy.split(
            dist, numpy.cumsum([len(domain.attributes),
                                len(domain.class_vars)]))

        def parts(array, density, col_dist):
            array = numpy.atleast_2d(array)
            nans = sum([dist.nans for dist in col_dist])
            non_nans = sum([dist.non_nans for dist in col_dist])
            if density == Storage.DENSE:
                return DenseArray(nans, non_nans, col_dist)
            elif density == Storage.SPARSE:
                return SparseArray(nans, non_nans, col_dist)
            elif density == Storage.SPARSE_BOOL:
                return SparseBoolArray(nans, non_nans, col_dist)
            elif density == Storage.MISSING:
                return NotAvailable()
            else:
                assert False
                return None

        X_part = parts(table.X, table.X_density(), X_dist)
        Y_part = parts(table.Y, table.Y_density(), Y_dist)
        M_part = parts(table.metas, table.metas_density(), M_dist)
        return Summary(n_instances, domain, X_part, Y_part, M_part)


def is_sortable(table):
    if isinstance(table, SqlTable):
        return False
    elif isinstance(table, Orange.data.Table):
        return True
    else:
        return False


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDataTable).run(
        insert_dataset=[
            (0, Table("iris")),
            (1, Table("brown-selected")),
            (2, Table("housing"))
        ])
