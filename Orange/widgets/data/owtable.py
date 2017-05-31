import sys
import threading
import io
import csv
import itertools
import concurrent.futures

from collections import OrderedDict, namedtuple
from math import isnan

import numpy
from scipy.sparse import issparse

from AnyQt.QtWidgets import (
    QTableView, QHeaderView, QAbstractButton, QAction, QApplication,
    QStyleOptionHeader, QStyle, QStylePainter, QStyledItemDelegate
)
from AnyQt.QtGui import QColor, QKeySequence, QClipboard
from AnyQt.QtCore import (
    Qt, QSize, QEvent, QByteArray, QMimeData, QObject, QMetaObject,
    QAbstractProxyModel, QIdentityProxyModel, QModelIndex,
    QItemSelectionModel, QItemSelection, QItemSelectionRange,
    QT_VERSION
)
from AnyQt.QtCore import pyqtSlot as Slot

import Orange.data
from Orange.data.storage import Storage
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import basic_stats

from Orange.widgets import widget, gui
from Orange.widgets.settings import (Setting, ContextSetting,
                                     DomainContextHandler)
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils import datacaching
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import TableModel


class RichTableDecorator(QIdentityProxyModel):
    """A proxy model for a TableModel with some bells and whistles

    (adds support for gui.BarRole, include variable labels and icons
    in the header)
    """
    #: Rich header data flags.
    Name, Labels, Icon = 1, 2, 4

    def __init__(self, source, parent=None):
        super().__init__(parent)

        self._header_flags = RichTableDecorator.Name
        self._labels = []
        self._continuous = []

        self.setSourceModel(source)

    @property
    def source(self):
        return getattr(self.sourceModel(), "source", None)

    @property
    def vars(self):
        return getattr(self.sourceModel(), "vars", [])

    def setSourceModel(self, source):
        if source is not None and \
                not isinstance(source, TableModel):
            raise TypeError()

        if source is not None:
            self._continuous = [var.is_continuous for var in source.vars]
            labels = []
            for var in source.vars:
                if isinstance(var, Orange.data.Variable):
                    labels.extend(var.attributes.keys())
            self._labels = list(sorted(
                {label for label in labels if not label.startswith("_")}))
        else:
            self._continuous = []
            self._labels = []

        super().setSourceModel(source)

    def data(self, index, role=Qt.DisplayRole,
             # for faster local lookup
             _BarRole=gui.TableBarItem.BarRole):
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
        if self.sourceModel() is None:
            return None

        # NOTE: Always use `self.sourceModel().heaerData(...)` and not
        # super().headerData(...). The later does not work for zero length
        # source models
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            var = self.sourceModel().headerData(
                section, orientation, TableModel.VariableRole)
            if var is None:
                return self.sourceModel().headerData(
                    section, orientation, Qt.DisplayRole)

            lines = []
            if self._header_flags & RichTableDecorator.Name:
                lines.append(var.name)
            if self._header_flags & RichTableDecorator.Labels:
                lines.extend(str(var.attributes.get(label, ""))
                             for label in self._labels)
            return "\n".join(lines)
        elif orientation == Qt.Horizontal and role == Qt.DecorationRole and \
                self._header_flags & RichTableDecorator.Icon:
            var = self.sourceModel().headerData(
                section, orientation, TableModel.VariableRole)
            if var is not None:
                return gui.attributeIconDict[var]
            else:
                return None
        else:
            return self.sourceModel().headerData(section, orientation, role)

    def setRichHeaderFlags(self, flags):
        if flags != self._header_flags:
            self._header_flags = flags
            self.headerDataChanged.emit(
                Qt.Horizontal, 0, self.columnCount() - 1)

    def richHeaderFlags(self):
        return self._header_flags

    if QT_VERSION < 0xFFFFFF:  # TODO: change when QTBUG-44143 is fixed
        def sort(self, column, order):
            # Preempt the layout change notification
            self.layoutAboutToBeChanged.emit()
            # Block signals to suppress repeated layout[AboutToBe]Changed
            # TODO: Are any other signals emitted during a sort?
            self.blockSignals(True)
            try:
                rval = self.sourceModel().sort(column, order)
            finally:
                self.blockSignals(False)
            # Tidy up.
            self.layoutChanged.emit()
            return rval
    else:
        def sort(self, column, order):
            return self.sourceModel().sort(column, order)


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

    def setSourceModel(self, model):
        super().setSourceModel(model)

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


class BlockSelectionModel(QItemSelectionModel):
    """
    Item selection model ensuring the selection maintains a simple block
    like structure.

    e.g.

        [a b] c [d e]
        [f g] h [i j]

    is allowed but this is not

        [a] b  c  d e
        [f  g] h [i j]

    I.e. select the Cartesian product of row and column indices.

    """
    def __init__(self, model, parent=None, selectBlocks=True, **kwargs):
        super().__init__(model, parent, **kwargs)
        self.__selectBlocks = selectBlocks

    def select(self, selection, flags):
        """Reimplemented."""
        if isinstance(selection, QModelIndex):
            selection = QItemSelection(selection, selection)

        model = self.model()
        indexes = self.selectedIndexes()

        rows = set(ind.row() for ind in indexes)
        cols = set(ind.column() for ind in indexes)

        if flags & QItemSelectionModel.Select and \
                not flags & QItemSelectionModel.Clear and self.__selectBlocks:
            indexes = selection.indexes()
            sel_rows = set(ind.row() for ind in indexes).union(rows)
            sel_cols = set(ind.column() for ind in indexes).union(cols)

            selection = QItemSelection()

            for r_start, r_end in ranges(sorted(sel_rows)):
                for c_start, c_end in ranges(sorted(sel_cols)):
                    top_left = model.index(r_start, c_start)
                    bottom_right = model.index(r_end - 1, c_end - 1)
                    selection.select(top_left, bottom_right)
        elif self.__selectBlocks and flags & QItemSelectionModel.Deselect:
            indexes = selection.indexes()

            def to_ranges(indices):
                return list(range(*r) for r in ranges(indices))

            selected_rows = to_ranges(sorted(rows))
            selected_cols = to_ranges(sorted(cols))

            desel_rows = to_ranges(set(ind.row() for ind in indexes))
            desel_cols = to_ranges(set(ind.column() for ind in indexes))

            selection = QItemSelection()

            # deselection extended vertically
            for row_range, col_range in \
                    itertools.product(selected_rows, desel_cols):
                selection.select(
                    model.index(row_range.start, col_range.start),
                    model.index(row_range.stop - 1, col_range.stop - 1)
                )
            # deselection extended horizontally
            for row_range, col_range in \
                    itertools.product(desel_rows, selected_cols):
                selection.select(
                    model.index(row_range.start, col_range.start),
                    model.index(row_range.stop - 1, col_range.stop - 1)
                )

        QItemSelectionModel.select(self, selection, flags)

    def selectBlocks(self):
        """Is the block selection in effect."""
        return self.__selectBlocks

    def setSelectBlocks(self, state):
        """Set the block selection state.

        If set to False, the selection model behaves as the base
        QItemSelectionModel

        """
        self.__selectBlocks = state


def ranges(indices):
    """
    Group consecutive indices into `(start, stop)` tuple 'ranges'.

    >>> list(ranges([1, 2, 3, 5, 3, 4]))
    >>> [(1, 4), (5, 6), (3, 5)]

    """
    g = itertools.groupby(enumerate(indices),
                          key=lambda t: t[1] - t[0])
    for _, range_ind in g:
        range_ind = list(range_ind)
        _, start = range_ind[0]
        _, end = range_ind[-1]
        yield start, end + 1


def table_selection_to_mime_data(table):
    """Copy the current selection in a QTableView to the clipboard.
    """
    lines = table_selection_to_list(table)

    csv = lines_to_csv_string(lines, dialect="excel").encode("utf-8")
    tsv = lines_to_csv_string(lines, dialect="excel-tab").encode("utf-8")

    mime = QMimeData()
    mime.setData("text/csv", QByteArray(csv))
    mime.setData("text/tab-separated-values", QByteArray(tsv))
    mime.setData("text/plain", QByteArray(tsv))
    return mime


def lines_to_csv_string(lines, dialect="excel"):
    stream = io.StringIO()
    writer = csv.writer(stream, dialect=dialect)
    writer.writerows(lines)
    return stream.getvalue()


def table_selection_to_list(table):
    model = table.model()
    indexes = table.selectedIndexes()

    rows = sorted(set(index.row() for index in indexes))
    columns = sorted(set(index.column() for index in indexes))

    lines = []
    for row in rows:
        line = []
        for col in columns:
            val = model.index(row, col).data(Qt.DisplayRole)
            # TODO: use style item delegate displayText?
            line.append(str(val))
        lines.append(line)

    return lines


TableSlot = namedtuple("TableSlot", ["input_id", "table", "summary", "view"])


class OWDataTable(widget.OWWidget):
    name = "Data Table"
    description = "View the data set in a spreadsheet."
    icon = "icons/Table.svg"
    priority = 10

    buttons_area_orientation = Qt.Vertical

    class Inputs:
        data = Input("Data", Table, multiple=True)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    show_distributions = Setting(False)
    dist_color_RGB = Setting((220, 220, 220, 255))
    show_attribute_labels = Setting(True)
    select_rows = Setting(True)
    auto_commit = Setting(True)

    color_by_class = Setting(True)
    settingsHandler = DomainContextHandler(
        match_values=DomainContextHandler.MATCH_VALUES_ALL)
    selected_rows = ContextSetting([])
    selected_cols = ContextSetting([])

    def __init__(self):
        super().__init__()

        self._inputs = OrderedDict()

        self.dist_color = QColor(*self.dist_color_RGB)

        info_box = gui.vBox(self.controlArea, "Info")
        self.info_ex = gui.widgetLabel(info_box, 'No data on input.', )
        self.info_ex.setWordWrap(True)
        self.info_attr = gui.widgetLabel(info_box, ' ')
        self.info_attr.setWordWrap(True)
        self.info_class = gui.widgetLabel(info_box, ' ')
        self.info_class.setWordWrap(True)
        self.info_meta = gui.widgetLabel(info_box, ' ')
        self.info_meta.setWordWrap(True)
        info_box.setMinimumWidth(200)
        gui.separator(self.controlArea)

        box = gui.vBox(self.controlArea, "Variables")
        self.c_show_attribute_labels = gui.checkBox(
            box, self, "show_attribute_labels",
            "Show variable labels (if present)",
            callback=self._on_show_variable_labels_changed)

        gui.checkBox(box, self, "show_distributions",
                     'Visualize continuous values',
                     callback=self._on_distribution_color_changed)
        gui.checkBox(box, self, "color_by_class", 'Color by instance classes',
                     callback=self._on_distribution_color_changed)

        box = gui.vBox(self.controlArea, "Selection")

        gui.checkBox(box, self, "select_rows", "Select full rows",
                     callback=self._on_select_rows_changed)

        gui.rubber(self.controlArea)

        reset = gui.button(
            None, self, "Restore Original Order", callback=self.restore_order,
            tooltip="Show rows in the original order", autoDefault=False)
        self.buttonsArea.layout().insertWidget(0, reset)
        gui.auto_commit(self.buttonsArea, self, "auto_commit",
                        "Send Selected Rows", "Send Automatically")

        # GUI with tabs
        self.tabs = gui.tabWidget(self.mainArea)
        self.tabs.currentChanged.connect(self._on_current_tab_changed)

    def copy_to_clipboard(self):
        self.copy()

    def sizeHint(self):
        return QSize(800, 500)

    @Inputs.data
    def set_dataset(self, data, tid=None):
        """Set the input dataset."""
        self.closeContext()
        if data is not None:
            if tid in self._inputs:
                # update existing input slot
                slot = self._inputs[tid]
                view = slot.view
                # reset the (header) view state.
                view.setModel(None)
                view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
            else:
                view = QTableView()
                view.setSortingEnabled(True)
                view.setHorizontalScrollMode(QTableView.ScrollPerPixel)

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

            view.dataset = data
            self.tabs.addTab(view, getattr(data, "name", "Data"))

            self._setup_table_view(view, data)
            slot = TableSlot(tid, data, table_summary(data), view)
            view._input_slot = slot
            self._inputs[tid] = slot

            self.tabs.setCurrentIndex(self.tabs.indexOf(view))

            self.set_info(slot.summary)

            if isinstance(slot.summary.len, concurrent.futures.Future):
                def update(f):
                    QMetaObject.invokeMethod(
                        self, "_update_info", Qt.QueuedConnection)

                slot.summary.len.add_done_callback(update)

        elif tid in self._inputs:
            slot = self._inputs.pop(tid)
            view = slot.view
            view.hide()
            view.deleteLater()
            self.tabs.removeTab(self.tabs.indexOf(view))

            current = self.tabs.currentWidget()
            if current is not None:
                self.set_info(current._input_slot.summary)

        self.tabs.tabBar().setVisible(self.tabs.count() > 1)
        self.selected_rows = []
        self.selected_cols = []
        self.openContext(data)
        self.set_selection()
        self.commit()

    def _setup_table_view(self, view, data):
        """Setup the `view` (QTableView) with `data` (Orange.data.Table)
        """
        if data is None:
            view.setModel(None)
            return

        datamodel = TableModel(data)
        datamodel = RichTableDecorator(datamodel)

        rowcount = data.approx_len()

        if self.color_by_class and data.domain.has_discrete_class:
            color_schema = [
                QColor(*c) for c in data.domain.class_var.colors]
        else:
            color_schema = None
        if self.show_distributions:
            view.setItemDelegate(
                gui.TableBarItem(
                    self, color=self.dist_color, color_schema=color_schema)
            )
        else:
            view.setItemDelegate(QStyledItemDelegate(self))

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
        self._update_variable_labels(view)

        selmodel = BlockSelectionModel(
            view.model(), parent=view, selectBlocks=not self.select_rows)
        view.setSelectionModel(selmodel)
        view.selectionModel().selectionChanged.connect(self.update_selection)

    #noinspection PyBroadException
    def set_corner_text(self, table, text):
        """Set table corner text."""
        # As this is an ugly hack, do everything in
        # try - except blocks, as it may stop working in newer Qt.

        if not hasattr(table, "btn") and not hasattr(table, "btnfailed"):
            try:
                btn = table.findChild(QAbstractButton)

                class efc(QObject):
                    def eventFilter(self, o, e):
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
                table.efc = efc()
                btn.installEventFilter(table.efc)
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
                    btn).expandedTo(QApplication.globalStrut())
                if s.isValid():
                    table.verticalHeader().setMinimumWidth(s.width())
            except Exception:
                pass

    def _on_current_tab_changed(self, index):
        """Update the info box on current tab change"""
        view = self.tabs.widget(index)
        if view is not None and view.model() is not None:
            self.set_info(view._input_slot.summary)
        else:
            self.set_info(None)

    def _update_variable_labels(self, view):
        "Update the variable labels visibility for `view`"
        model = view.model()
        if isinstance(model, TableSliceProxy):
            model = model.sourceModel()

        if self.show_attribute_labels:
            model.setRichHeaderFlags(
                RichTableDecorator.Labels | RichTableDecorator.Name)

            labelnames = set()
            for a in model.source.domain:
                labelnames.update(a.attributes.keys())
            labelnames = sorted(
                [label for label in labelnames if not label.startswith("_")])
            self.set_corner_text(view, "\n".join([""] + labelnames))
        else:
            model.setRichHeaderFlags(RichTableDecorator.Name)
            self.set_corner_text(view, "")

    def _on_show_variable_labels_changed(self):
        """The variable labels (var.attribues) visibility was changed."""
        for slot in self._inputs.values():
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
                delegate = gui.TableBarItem(self, color=self.dist_color,
                                            color_schema=color_schema)
            else:
                delegate = QStyledItemDelegate(self)
            widget.setItemDelegate(delegate)
        tab = self.tabs.currentWidget()
        if tab:
            tab.reset()

    def _on_select_rows_changed(self):
        for slot in self._inputs.values():
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

    def set_info(self, summary):
        if summary is None:
            self.info_ex.setText("No data on input.")
            self.info_attr.setText("")
            self.info_class.setText("")
            self.info_meta.setText("")
        else:
            info_len, info_attr, info_class, info_meta = \
                format_summary(summary)

            self.info_ex.setText(info_len)
            self.info_attr.setText(info_attr)
            self.info_class.setText(info_class)
            self.info_meta.setText(info_meta)

    @Slot()
    def _update_info(self):
        current = self.tabs.currentWidget()
        if current is not None and current.model() is not None:
            self.set_info(current._input_slot.summary)

    def update_selection(self, *_):
        self.commit()

    def set_selection(self):
        if len(self.selected_rows) and len(self.selected_cols):
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

    def get_selection(self, view):
        """
        Return the selected row and column indices of the selection in view.
        """
        selection = view.selectionModel().selection()
        model = view.model()
        # map through the proxies into input table.
        while isinstance(model, QAbstractProxyModel):
            selection = model.mapSelectionToSource(selection)
            model = model.sourceModel()

        assert isinstance(model, TableModel)

        indexes = selection.indexes()

        rows = list(set(ind.row() for ind in indexes))
        # map the rows through the applied sorting (if any)
        rows = sorted(model.mapToTableRows(rows))
        cols = sorted(set(ind.column() for ind in indexes))
        return rows, cols

    @staticmethod
    def _get_model(view):
        model = view.model()
        while isinstance(model, QAbstractProxyModel):
            model = model.sourceModel()
        return model

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

            def select(data, rows, domain):
                """
                Select the data subset with specified rows and domain subsets.

                If either rows or domain is None they mean select all.
                """
                if rows is not None and domain is not None:
                    return data.from_table(domain, data, rows)
                elif rows is not None:
                    return data.from_table(data.domain, rows)
                elif domain is not None:
                    return data.from_table(domain, data)
                else:
                    return data

            domain = table.domain

            if len(colsel) < len(domain) + len(domain.metas):
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

            # Avoid a copy if all/none rows are selected.
            if not rowsel:
                selected_data = None
            elif len(rowsel) == len(table):
                selected_data = select(table, None, domain)
            else:
                selected_data = select(table, rowsel, domain)

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

        X_part = parts(table.X, table.X_density(), X_dist)
        Y_part = parts(table.Y, table.Y_density(), Y_dist)
        M_part = parts(table.metas, table.metas_density(), M_dist)
        return Summary(n_instances, domain, X_part, Y_part, M_part)


def format_summary(summary):
    text = []
    if isinstance(summary, ApproxSummary):
        if summary.len.done():
            text += ["{} instances".format(summary.len.result())]
        else:
            text += ["~{} instances".format(summary.approx_len)]

    elif isinstance(summary, Summary):
        text += ["{} instances".format(summary.len)]

        if sum(p.nans for p in [summary.X, summary.Y, summary.M]) == 0:
            text[-1] += " (no missing values)"

    def format_part(part):
        if isinstance(part, NotAvailable):
            return ""
        elif part.nans + part.non_nans == 0:
            return ""

        if isinstance(part, DenseArray):
            total = part.nans + part.non_nans
            miss = ("%.1f%%" % (100 * part.nans / total) if part.nans > 0
                    else "no")
            return " (%s missing values)" % miss
        elif isinstance(part, (SparseArray, SparseBoolArray)):
            text = " ({}, density {:.2f}%)"
            tag = "sparse" if isinstance(part, SparseArray) else "tags"
            total = part.nans + part.non_nans
            return text.format(tag, 100 * part.non_nans / total)
        else:
            # MISSING, N/A
            return ""

    def sp(n):
        if n == 0:
            return "No", "s"
        elif n == 1:
            return str(n), ''
        else:
            return str(n), 's'

    text += [("%s feature%s" % sp(len(summary.domain.attributes)))
             + format_part(summary.X)]

    if not summary.domain.class_vars:
        text += ["No target variable."]
    else:
        if len(summary.domain.class_vars) > 1:
            c_text = "%s outcome%s" % sp(len(summary.domain.class_vars))
        elif summary.domain.has_continuous_class:
            c_text = "Continuous target variable"
        else:
            c_text = "Discrete class with %s value%s" % sp(
                len(summary.domain.class_var.values))
        c_text += format_part(summary.Y)
        text += [c_text]

    text += [("%s meta attribute%s" % sp(len(summary.domain.metas)))
             + format_part(summary.M)]

    return text


def is_sortable(table):
    if isinstance(table, SqlTable):
        return False
    elif isinstance(table, Orange.data.Table):
        return True
    else:
        return False


def test_main():
    a = QApplication(sys.argv)
    ow = OWDataTable()

    iris = Table("iris")
    brown = Table("brown-selected")
    housing = Table("housing")
    ow.show()
    ow.raise_()

    ow.set_dataset(iris, iris.name)
    ow.set_dataset(brown, brown.name)
    ow.set_dataset(housing, housing.name)

    rval = a.exec()
#     ow.saveSettings()
    return rval


def test_model():
    app = QApplication([])
    view = QTableView(
        sortingEnabled=True
    )
    data = Orange.data.Table("lenses")
    model = TableModel(data)

    view.setModel(model)

    view.show()
    view.raise_()
    return app.exec()

if __name__ == "__main__":
    sys.exit(test_main())
