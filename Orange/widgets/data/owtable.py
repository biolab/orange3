import sys
import threading
import io
import csv
import concurrent.futures

from collections import OrderedDict, namedtuple
from math import isnan

import numpy

from PyQt4 import QtCore
from PyQt4 import QtGui

from PyQt4.QtGui import QIdentityProxyModel, QTableView
from PyQt4.QtCore import Qt, QMetaObject, QT_VERSION
from PyQt4.QtCore import pyqtSlot as Slot

import Orange.data
from Orange.data import ContinuousVariable
from Orange.data.storage import Storage
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import basic_stats

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import colorpalette, datacaching
from Orange.widgets.utils import itemmodels
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
                not isinstance(source, itemmodels.TableModel):
            raise TypeError()

        if source is not None:
            self._continuous = [isinstance(var, ContinuousVariable)
                                for var in source.vars]
            labels = []
            for var in source.vars:
                if isinstance(var, Orange.data.Variable):
                    labels.extend(var.attributes.keys())
            self._labels = list(sorted(set(labels)))
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
        else:
            return super().data(index, role)

    def headerData(self, section, orientation, role):
        # QIdentityProxyModel doesn’t show headers for empty models
        if self.sourceModel:
            return self.sourceModel().headerData(section, orientation, role)

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is None:
                return super().headerData(section, orientation, Qt.DisplayRole)

            lines = []
            if self._header_flags & RichTableDecorator.Name:
                lines.append(var.name)
            if self._header_flags & RichTableDecorator.Labels:
                lines.extend(str(var.attributes.get(label, ""))
                             for label in self._labels)
            return "\n".join(lines)
        elif orientation == Qt.Horizontal and role == Qt.DecorationRole and \
                self._header_flags & RichTableDecorator.Icon:
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


def table_selection_to_mime_data(table):
    """Copy the current selection in a QTableView to the clipboard.
    """
    lines = table_selection_to_list(table)

    csv = lines_to_csv_string(lines, dialect="excel")
    tsv = lines_to_csv_string(lines, dialect="excel-tab")

    mime = QtCore.QMimeData()
    mime.setData("text/csv", QtCore.QByteArray(csv))
    mime.setData("text/tab-separated-values", QtCore.QByteArray(tsv))
    mime.setData("text/plain", QtCore.QByteArray(tsv))
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
    description = "View data set in a spreadsheet."
    icon = "icons/Table.svg"
    priority = 100

    inputs = [("Data", Table, "set_dataset", widget.Multiple)]
    outputs = [("Selected Data", Table, widget.Default),
               ("Other Data", Table)]

    show_distributions = Setting(False)
    dist_color_RGB = Setting((220, 220, 220, 255))
    show_attribute_labels = Setting(True)
    auto_commit = Setting(True)

    color_settings = Setting(None)
    selected_schema_index = Setting(0)
    color_by_class = Setting(True)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.inputs = OrderedDict()

        self.dist_color = QtGui.QColor(*self.dist_color_RGB)
        self.selectionChangedFlag = False

        info_box = gui.widgetBox(self.controlArea, "Info")
        self.info_ex = gui.widgetLabel(info_box, 'No data on input.', )
        self.info_ex.setWordWrap(True)
        self.info_attr = gui.widgetLabel(info_box, ' ')
        self.info_attr.setWordWrap(True)
        self.info_class = gui.widgetLabel(info_box, ' ')
        self.info_class.setWordWrap(True)
        self.info_meta = gui.widgetLabel(info_box, ' ')
        self.info_meta.setWordWrap(True)

        gui.separator(info_box)
        gui.button(info_box, self, "Restore Original Order",
                   callback=self.restore_order,
                   tooltip="Show rows in the original order",
                   autoDefault=False)
        info_box.setMinimumWidth(200)
        gui.separator(self.controlArea)

        box = gui.widgetBox(self.controlArea, "Variables")
        self.c_show_attribute_labels = gui.checkBox(
            box, self, "show_attribute_labels",
            "Show variable labels (if present)",
            callback=self._on_show_variable_labels_changed)

        gui.checkBox(box, self, "show_distributions",
                     'Visualize continuous values',
                     callback=self._on_distribution_color_changed)
        gui.checkBox(box, self, "color_by_class", 'Color by instance classes',
                     callback=self._on_distribution_color_changed)
        gui.button(box, self, "Set colors", self.set_colors, autoDefault=False,
                   tooltip="Set the background color and color palette")

        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selected Rows", "Auto send is on")

        dlg = self.create_color_dialog()
        self.discPalette = dlg.getDiscretePalette("discPalette")

        # GUI with tabs
        self.tabs = gui.tabWidget(self.mainArea)
        self.tabs.currentChanged.connect(self._on_current_tab_changed)

        copy = QtGui.QAction("Copy", self, shortcut=QtGui.QKeySequence.Copy,
                             triggered=self.copy)
        self.addAction(copy)

    def sizeHint(self):
        return QtCore.QSize(800, 500)

    def create_color_dialog(self):
        c = colorpalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Default", "Default color",
                            QtGui.QColor(self.dist_color))
        c.setColorSchemas(self.color_settings, self.selected_schema_index)
        return c

    def set_colors(self):
        dlg = self.create_color_dialog()
        if dlg.exec():
            self.color_settings = dlg.getColorSchemas()
            self.selected_schema_index = dlg.selectedSchemaIndex
            self.discPalette = dlg.getDiscretePalette("discPalette")
            self.dist_color = QtGui.QColor(dlg.getColor("Default"))
            self.dist_color_RGB = (
                self.dist_color.red(), self.dist_color.green(),
                self.dist_color.blue(), self.dist_color.alpha()
            )
            if self.show_distributions:
                self._on_distribution_color_changed()

    def set_dataset(self, data, tid=None):
        """Set the input dataset."""

        if data is not None:
            if tid in self.inputs:
                # update existing input slot
                slot = self.inputs[tid]
                view = slot.view
                # reset the (header) view state.
                view.setModel(None)
                view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
            else:
                view = QTableView()
                view.setSelectionBehavior(QTableView.SelectRows)
                view.setSortingEnabled(True)
                view.setHorizontalScrollMode(QTableView.ScrollPerPixel)

                header = view.horizontalHeader()
                header.setMovable(True)
                header.setClickable(True)
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
            self.inputs[tid] = slot

            self.tabs.setCurrentIndex(self.tabs.indexOf(view))

            self.set_info(slot.summary)

            if isinstance(slot.summary.len, concurrent.futures.Future):
                def update(f):
                    QMetaObject.invokeMethod(
                        self, "_update_info", Qt.QueuedConnection)

                slot.summary.len.add_done_callback(update)

        elif tid in self.inputs:
            slot = self.inputs.pop(tid)
            view = slot.view
            view.hide()
            view.deleteLater()
            self.tabs.removeTab(self.tabs.indexOf(view))

            current = self.tabs.currentWidget()
            if current is not None:
                self.set_info(current._input_slot.summary)

        self.tabs.tabBar().setVisible(self.tabs.count() > 1)

    def _setup_table_view(self, view, data):
        """Setup the `view` (QTableView) with `data` (Orange.data.Table)
        """
        if data is None:
            view.setModel(None)
            return

        datamodel = TableModel(data)

        datamodel = RichTableDecorator(datamodel)

        color_schema = self.discPalette if self.color_by_class else None
        if self.show_distributions:
            view.setItemDelegate(
                gui.TableBarItem(
                    self, color=self.dist_color, color_schema=color_schema)
            )
        else:
            view.setItemDelegate(QtGui.QStyledItemDelegate(self))

        # Enable/disable view sorting based on data's type
        view.setSortingEnabled(is_sortable(data))
        header = view.horizontalHeader()
        header.setClickable(is_sortable(data))
        header.setSortIndicatorShown(is_sortable(data))

        view.setModel(datamodel)

        vheader = view.verticalHeader()
        option = view.viewOptions()
        size = view.style().sizeFromContents(
            QtGui.QStyle.CT_ItemViewItem, option,
            QtCore.QSize(20, 20), view)

        vheader.setDefaultSectionSize(size.height() + 2)

        # update the header (attribute names)
        self._update_variable_labels(view)

        view.selectionModel().selectionChanged.connect(self.update_selection)

    #noinspection PyBroadException
    def set_corner_text(self, table, text):
        """Set table corner text."""
        # As this is an ugly hack, do everything in
        # try - except blocks, as it may stop working in newer Qt.

        if not hasattr(table, "btn") and not hasattr(table, "btnfailed"):
            try:
                btn = table.findChild(QtGui.QAbstractButton)

                class efc(QtCore.QObject):
                    def eventFilter(self, o, e):
                        if (isinstance(o, QtGui.QAbstractButton) and
                                e.type() == QtCore.QEvent.Paint):
                            # paint by hand (borrowed from QTableCornerButton)
                            btn = o
                            opt = QtGui.QStyleOptionHeader()
                            opt.init(btn)
                            state = QtGui.QStyle.State_None
                            if btn.isEnabled():
                                state |= QtGui.QStyle.State_Enabled
                            if btn.isActiveWindow():
                                state |= QtGui.QStyle.State_Active
                            if btn.isDown():
                                state |= QtGui.QStyle.State_Sunken
                            opt.state = state
                            opt.rect = btn.rect()
                            opt.text = btn.text()
                            opt.position = \
                                QtGui.QStyleOptionHeader.OnlyOneSection
                            painter = QtGui.QStylePainter(btn)
                            painter.drawControl(QtGui.QStyle.CE_Header, opt)
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
                opt = QtGui.QStyleOptionHeader()
                opt.text = btn.text()
                s = btn.style().sizeFromContents(
                    QtGui.QStyle.CT_HeaderSection,
                    opt, QtCore.QSize(),
                    btn).expandedTo(QtGui.QApplication.globalStrut())
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

        if self.show_attribute_labels:
            model.setRichHeaderFlags(
                RichTableDecorator.Labels | RichTableDecorator.Name)

            labelnames = set()
            for a in model.source.domain:
                labelnames.update(a.attributes.keys())
            labelnames = sorted(list(labelnames))
            self.set_corner_text(view, "\n".join([""] + labelnames))
        else:
            model.setRichHeaderFlags(RichTableDecorator.Name)
            self.set_corner_text(view, "")

    def _on_show_variable_labels_changed(self):
        """The variable labels (var.attribues) visibility was changed."""
        for slot in self.inputs.values():
            self._update_variable_labels(slot.view)

    def _on_distribution_color_changed(self):
        for ti in range(self.tabs.count()):
            color_schema = self.discPalette if self.color_by_class else None
            if self.show_distributions:
                delegate = gui.TableBarItem(self, color=self.dist_color,
                                            color_schema=color_schema)
            else:
                delegate = QtGui.QStyledItemDelegate(self)
            self.tabs.widget(ti).setItemDelegate(delegate)
        tab = self.tabs.currentWidget()
        if tab:
            tab.reset()

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

    def get_current_selection(self):
        table = self.tabs.currentWidget()
        if table and table.model():
            proxy = table.model()
            rows = table.selectionModel().selectedRows()
            rows = sorted([proxy.mapToSource(ind).row() for ind in rows])
            source = proxy.sourceModel()
            rows = [source.headerData(row, Qt.Vertical, Qt.DisplayRole) - 1
                    for row in rows]
            return rows
        else:
            return []

    def commit(self):
        selected_data = other_data = None
        table = self.tabs.currentWidget()
        if table and table.model():
            model = table.model().sourceModel()
            selection = self.get_current_selection()

            # Avoid a copy if all/none rows are selected.
            if not selection:
                selected_data = None
                other_data = model.source
            elif len(selection) == len(model.source):
                selected_data = model.source
                other_data = None
            else:
                selected_data = model.source[selection]
                selection = set(selection)

                other = [i for i in range(len(model.source))
                         if i not in selection]
                other_data = model.source[other]

        self.send("Selected Data", selected_data)
        self.send("Other Data", other_data)

        self.selectionChangedFlag = False

    def copy(self):
        """
        Copy current table selection to the clipboard.
        """
        view = self.tabs.currentWidget()
        if view is not None:
            mime = table_selection_to_mime_data(view)
            QtGui.QApplication.clipboard().setMimeData(
                mime, QtGui.QClipboard.Clipboard
            )

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
        elif isinstance(summary.domain.class_var, ContinuousVariable):
            c_text = "Continuous target variable"
        else:
            c_text = "Discrete class with %s value%s" % sp(
                len(summary.domain.class_var.values))
        c_text += format_part(summary.Y)
        text += [c_text]

    text += [("%s meta attributes%s" % sp(len(summary.domain.metas)))
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
    a = QtGui.QApplication(sys.argv)
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
    app = QtGui.QApplication([])
    view = QtGui.QTableView(
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
