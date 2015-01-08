import sys
import threading
import traceback
import io
import csv
from math import isnan

from PyQt4 import QtCore
from PyQt4 import QtGui

from PyQt4.QtGui import QSortFilterProxyModel
from PyQt4.QtCore import Qt

from Orange.data import ContinuousVariable
from Orange.data.storage import Storage
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import basic_stats

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import colorpalette, datacaching
from Orange.widgets.utils import itemmodels


class TableModel(itemmodels.TableModel):
    #: header data flags
    Name, Labels, Icon = 1, 2, 4

    def __init__(self, data, parent=None):
        super().__init__(data, parent)
        self._dist = None
        self._continuous = [isinstance(var, ContinuousVariable)
                            for var in self.vars]
        self._header_flags = TableModel.Name

    def data(self, index, role=Qt.DisplayRole,
             # for faster local lookup
             _BarRole=gui.TableBarItem.BarRole):
        if role == _BarRole and self._continuous[index.column()]:
            val = super().data(index, self.ValueRole)
            if isnan(val):
                return None

            if self._dist is None:
                self._dist = datacaching.getCached(
                    self.source, basic_stats.DomainBasicStats,
                    (self.source, True)
                )
            dist = self._dist[index.column()]
            return (val - dist.min) / (dist.max - dist.min or 1)
        else:
            return super().data(index, role)

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            var = self.vars[section]
            lines = []
            if self._header_flags & TableModel.Name:
                lines.append(var.name)
            if self._header_flags & TableModel.Labels:
                lines.extend(str(var.attributes.get(label, ""))
                             for label in self._labels)
            return "\n".join(lines)
        elif orientation == Qt.Horizontal and role == Qt.DecorationRole and \
                self._header_flags & TableModel.Icon:
            var = self.vars[section]
            return gui.attributeIconDict[var]
        else:
            return super().headerData(section, orientation, role)

    def setRichHeaderFlags(self, flags):
        if flags != self._header_flags:
            self._header_flags = flags
            self.headerDataChanged.emit(Qt.Horizontal, 0, self.columnCount())

    def richHeaderFlags(self):
        return self._header_flags


class TableSortProxyModel(QSortFilterProxyModel):
    def lessThan(self, left, right):
        vleft = left.data(self.sortRole())
        vright = right.data(self.sortRole())
        # Sort NaN values to the end.
        # (note: if left is NaN the comparison is always false)
        return vleft < vright if not isnan(vright) else True


#noinspection PyArgumentList
class TableViewWithCopy(QtGui.QTableView):

    def keyPressEvent(self, event):
        if event == QtGui.QKeySequence.Copy:
            sel_model = self.selectionModel()
            #noinspection PyBroadException
            try:
                self.copy_selection_to_clipboard(sel_model)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        else:
            return QtGui.QTableView.keyPressEvent(self, event)

    def copy_selection_to_clipboard(self, selection_model):
        """
        Copy table selection to the clipboard.
        """
        mime = table_selection_to_mime_data(self)
        QtGui.QApplication.clipboard().setMimeData(
            mime, QtGui.QClipboard.Clipboard
        )


def table_selection_to_mime_data(table):
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
    selected_schema_index = Setting(0)
    color_by_class = Setting(True)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.datasets = {}          # key: id, value: Table
        self.views = {}

        self.dist_color = QtGui.QColor(*self.dist_color_RGB)

        self.color_settings = None
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
            callback=self.c_show_attribute_labels_clicked)
        self.c_show_attribute_labels.setEnabled(True)
        gui.checkBox(box, self, "show_distributions",
                     'Visualize continuous values',
                     callback=self.cb_show_distributions)
        gui.checkBox(box, self, "color_by_class", 'Color by instance classes',
                     callback=self.cb_show_distributions)
        gui.button(box, self, "Set colors", self.set_colors, autoDefault=False,
                   tooltip="Set the background color and color palette")

        gui.separator(self.controlArea)
        selection_box = gui.widgetBox(self.controlArea, "Selection")
        self.send_button = gui.button(selection_box, self, "Send selections",
                                      self.commit, default=True)
        cb = gui.checkBox(selection_box, self, "auto_commit",
                          "Commit on any change", callback=self.commit_if)
        gui.setStopper(self, self.send_button, cb, "selectionChangedFlag",
                       self.commit)

        gui.rubber(self.controlArea)

        dlg = self.create_color_dialog()
        self.discPalette = dlg.getDiscretePalette("discPalette")

        # GUI with tabs
        self.tabs = gui.tabWidget(self.mainArea)
        self.tabs.currentChanged.connect(self._on_current_tab_changed)

    def sizeHint(self):
        return QtCore.QSize(800, 500)

    def create_color_dialog(self):
        c = colorpalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Default", "Default color",
                            QtGui.QColor(QtCore.Qt.white))
        c.setColorSchemas(self.color_settings, self.selected_schema_index)
        return c

    def set_colors(self):
        dlg = self.create_color_dialog()
        if dlg.exec():
            self.color_settings = dlg.getColorSchemas()
            self.selected_schema_index = dlg.selectedSchemaIndex
            self.discPalette = dlg.getDiscretePalette("discPalette")
            self.dist_color_RGB = dlg.getColor("Default")

    def set_dataset(self, data, tid=None):
        "Set the input dataset"

        if data is not None:
            if tid in self.datasets:
                # remove existing table
                self.datasets.pop(tid)
                view = self.views.pop(tid)
                self.tabs.removeTab(self.tabs.indexOf(view))

            self.datasets[tid] = data

            table = TableViewWithCopy()
            table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
            table.setSortingEnabled(True)
            table.setHorizontalScrollMode(QtGui.QTableWidget.ScrollPerPixel)

            header = table.horizontalHeader()
            header.setMovable(True)
            header.setClickable(True)
            header.setSortIndicatorShown(True)
            header.setSortIndicator(-1, Qt.AscendingOrder)

            header.sortIndicatorChanged.connect(
                lambda index, order:
                    table.model().sort(index, order) if index == -1 else 0
            )
            self.views[tid] = table
            data_name = getattr(data, "name", "")
            self.tabs.addTab(table, data_name)

            self._setup_table_view(table, data)

            self.tabs.setCurrentIndex(self.tabs.indexOf(table))
            self.set_info(data)
            self.send_button.setEnabled(not self.auto_commit)

        elif tid in self.datasets:
            table = self.id2table[tid]
            table = self.views[tid]
            self.datasets.pop(tid)
            table.hide()
            self.tabs.removeTab(self.tabs.indexOf(table))

            self.set_info(self.datasets.get(self.table2id.get(
                self.tabs.currentWidget(), None), None))

        self.tabs.tabBar().setVisible(self.tabs.count() > 1)

        if not self.datasets:
            self.send_button.setEnabled(False)

    def _setup_table_view(self, view, data):
        """Setup the `view` (QTableView) with `data` (Orange.data.Table)
        """
        if data is None:
            view.setModel(None)
            return

        datamodel = TableModel(data)
        color_schema = self.discPalette if self.color_by_class else None
        if self.show_distributions:
            view.setItemDelegate(
                gui.TableBarItem(
                    self, color=self.dist_color, color_schema=color_schema)
            )
        else:
            view.setItemDelegate(QtGui.QStyledItemDelegate(self))

        proxy = TableSortProxyModel()
        proxy.setSourceModel(datamodel)
        proxy.setSortRole(TableModel.ValueRole)
        proxy.examples = data
        proxy.source = data
        view.setModel(proxy)

        vheader = view.verticalHeader()
        option = view.viewOptions()
        size = view.style().sizeFromContents(
            QtGui.QStyle.CT_ItemViewItem, option,
            QtCore.QSize(20, 20), view)

        vheader.setDefaultSectionSize(size.height() + 2)

        # update the header (attribute names)
        self._update_variable_labels(view)

        view.selectionModel().selectionChanged.connect(
            self.update_selection
        )

    #noinspection PyBroadException
    def set_corner_text(self, table, text):
        """
        Set table corner text. As this is an ugly hack, do everything in
        try - except blocks, as it may stop working in newer Qt.
        """

        if not hasattr(table, "btn") and not hasattr(table, "btnfailed"):
            try:
                btn = table.findChild(QtGui.QAbstractButton)

                class efc(QtCore.QObject):
                    def eventFilter(self, o, e):
                        if (isinstance(o, QtGui.QAbstractButton) and
                                e.type() == QtCore.QEvent.Paint):
                        #paint by hand (borrowed from QTableCornerButton)
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
            model = view.model().sourceModel()
            self.set_info(model.source)

    def _update_variable_labels(self, view):
        "Update the variable labels visibility for `view`"
        model = view.model().sourceModel()

        model.setRichHeaderFlags(
            TableModel.Labels | TableModel.Name if self.show_attribute_labels
            else TableModel.Name
        )

        if self.show_attribute_labels:
            labelnames = set()
            for a in model.source.domain:
                labelnames.update(a.attributes.keys())
            labelnames = sorted(list(labelnames))
            self.set_corner_text(view, "\n".join([""] + labelnames))
        else:
            self.set_corner_text(view, "")

    def c_show_attribute_labels_clicked(self):
        for table in self.views.values():
            self._update_variable_labels(table)

    def cb_show_distributions(self):
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

    # show data in the default order
    def restore_order(self):
        table = self.tabs.currentWidget()
        if table is not None:
            table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

    __no_missing = [""] * 3

    def __compute_density(self, data):
        def desc(part, frm, to):
            nans = sum(dist[i].nans for i in range(frm, to))
            non_nans = sum(dist[i].non_nans for i in range(frm, to))
            tot = nans + non_nans
            if tot == 0:
                return ""
            density = getattr(data, part + "_density")()
            if density == Storage.DENSE:
                dp = "%.1f%%" % (100 * nans / tot) if nans > 0 else "no"
                return " (%s missing values)" % dp
            s = " (sparse" if density == Storage.SPARSE else " (tags"
            return s + ", density %.2f %%)" % (100 * non_nans / tot)

        dist = datacaching.getCached(data,
                                     basic_stats.DomainBasicStats, (data, True))
        domain = data.domain
        descriptions = [desc(part, frm, to)
                        for part, frm, to in [
                            ("X", 0, len(domain.attributes)),
                            ("Y", len(domain.attributes), len(domain)),
                            ("metas", len(domain),
                             len(domain) + len(domain.metas))]]
        if all(not d or d == " (no missing values)" for d in descriptions):
            descriptions = self.__no_missing
        return descriptions

    def set_info(self, data):
        """Updates data info."""
        def sp(n):
            if n == 0:
                return "No", "s"
            elif n == 1:
                return str(n), ''
            else:
                return str(n), 's'

        if data is None:
            self.info_ex.setText('No data on input.')
            self.info_attr.setText('')
            self.info_meta.setText('')
            self.info_class.setText('')
        else:
            if isinstance(data, SqlTable):
                descriptions = ['', '', '']
            else:
                descriptions = datacaching.getCached(
                    data, self.__compute_density, (data, ))
            out_i = "~%s instance%s" % sp(data.approx_len())
            if descriptions is self.__no_missing:
                out_i += " (no missing values)"
            self.info_ex.setText(out_i)

            def update_num_inst():
                out_i = "%s instance%s" % sp(len(data))
                if descriptions is self.__no_missing:
                    out_i += " (no missing values)"
                self.info_ex.setText(out_i)

            threading.Thread(target=update_num_inst).start()

            self.info_attr.setText("%s feature%s" %
                                   sp(len(data.domain.attributes)) +
                                   descriptions[0])

            self.info_meta.setText("%s meta attribute%s" %
                                   sp(len(data.domain.metas)) + descriptions[2])

            if not data.domain.class_vars:
                out_c = 'No target variable.'
            else:
                if len(data.domain.class_vars) > 1:
                    out_c = "%s outcome%s" % sp(len(data.domain.class_vars))
                elif isinstance(data.domain.class_var, ContinuousVariable):
                    out_c = 'Continuous target variable'
                else:
                    out_c = 'Discrete class with %s value%s' % sp(
                        len(data.domain.class_var.values))
                out_c += descriptions[1]
            self.info_class.setText(out_c)

    def update_selection(self, *_):
        view = self.tabs.currentWidget()

        self.send_button.setEnabled(
            view.selectionModel().hasSelection()
            and not self.auto_commit
        )
        self.commit_if()

    def get_current_selection(self):
        table = self.tabs.currentWidget()
        if table and table.model():
            proxy = table.model()
            new = table.selectionModel().selectedIndexes()
            return sorted(set([proxy.mapToSource(ind).row() for ind in new]))
        else:
            return []

    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.selectionChangedFlag = True

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

if __name__ == "__main__":
    sys.exit(test_main())
