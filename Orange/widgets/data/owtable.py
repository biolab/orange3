import sys
import traceback
from math import isnan
from functools import reduce
from PyQt4 import QtCore
from PyQt4 import QtGui

from Orange.data.storage import Storage
from Orange.data.table import Table
from Orange.data import ContinuousVariable
from Orange.statistics import basic_stats

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import colorpalette, datacaching
from Orange.widgets.widget import Multiple, Default


##############################################################################

def safe_call(func):
    from functools import wraps
    # noinspection PyBroadException

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            traceback.print_exc(file=sys.stderr)
    return wrapper


#noinspection PyMethodOverriding
class ExampleTableModel(QtCore.QAbstractItemModel):
    def __init__(self, data, _, *args):
        def _n_cols(density, attrs):
            if density == Storage.MISSING:
                return 0
            elif density == Storage.DENSE:
                return len(attrs)
            else:
                return 1

        QtCore.QAbstractItemModel.__init__(self, *args)
        self.examples = data
        domain = self.domain = data.domain
        self.all_attrs = domain.attributes + domain.class_vars + domain.metas
        self.X_density = data.X_density()
        self.Y_density = data.Y_density()
        self.metas_density = data.metas_density()
        self.n_attr_cols = _n_cols(self.X_density, domain.attributes)
        self.n_attr_class_cols = self.n_attr_cols + _n_cols(self.Y_density,
                                                            domain.class_vars)
        self.n_cols = self.n_attr_class_cols + _n_cols(self.metas_density,
                                                       domain.metas)
        self.nvariables = len(domain)
        self.dist = None

        self.cls_color = QtGui.QColor(160, 160, 160)
        self.meta_color = QtGui.QColor(220, 220, 200)
        self.sorted_map = range(len(data))

        self.attr_labels = sorted(
            reduce(set.union, [attr.attributes for attr in self.all_attrs],
                   set()))
        self._show_attr_labels = False
        self._other_data = {}

    def get_show_attr_labels(self):
        return self._show_attr_labels

    def set_show_attr_labels(self, val):
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        self._show_attr_labels = val
        self.emit(QtCore.SIGNAL("headerDataChanged(Qt::Orientation, int, int)"),
                  QtCore.Qt.Horizontal, 0, len(self.all_attrs) - 1)
        self.emit(QtCore.SIGNAL("layoutChanged()"))
        self.emit(QtCore.SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
                  self.index(0, 0),
                  self.index(len(self.examples) - 1, len(self.all_attrs) - 1))

    show_attr_labels = QtCore.pyqtProperty("bool",
                                           fget=get_show_attr_labels,
                                           fset=set_show_attr_labels,
                                           )

    @safe_call
    def data(self, index, role):
        row, col = self.sorted_map[index.row()], index.column()
        example = self.examples[row]

        if role == gui.TableClassValueRole:
            return example.get_class()

        # check whether we have a sparse columns,
        # handle background color role while you are at it
        sp_data = attributes = None
        if col < self.n_attr_cols:
            if role == QtCore.Qt.BackgroundRole:
                return
            density = self.X_density
            if density != Storage.DENSE:
                sp_data, attributes = example.sparse_x, self.domain.attributes
        elif col < self.n_attr_class_cols:
            if role == QtCore.Qt.BackgroundRole:
                return self.cls_color
            density = self.Y_density
            if density != Storage.DENSE:
                sp_data, attributes = example.sparse_y, self.domain.class_vars
        else:
            if role == QtCore.Qt.BackgroundRole:
                return self.meta_color
            density = self.metas_density
            if density != Storage.DENSE:
                sp_data, attributes = \
                    example.sparse_metas, self.domain.class_vars

        if sp_data is not None:
            if role == QtCore.Qt.DisplayRole:
                if density == Storage.SPARSE:
                    return ", ".join(
                        "{}={}".format(attributes[i].name,
                                       attributes[i].repr_val(v))
                        for i, v in zip(sp_data.indices, sp_data.data))
                else:
                    return ", ".join(
                        attributes[i].name for i in sp_data.indices)

        else:   # not sparse
            attr = self.all_attrs[col]
            val = example[attr]
            if role == QtCore.Qt.DisplayRole:
                return str(val)
            elif (role == gui.TableBarItem.BarRole and
                    isinstance(attr, ContinuousVariable) and
                    not isnan(val)):
                if self.dist is None:
                    self.dist = datacaching.getCached(
                        self.examples, basic_stats.DomainBasicStats,
                        (self.examples, True))
                dist = self.dist[col]
                return (val - dist.min) / (dist.max - dist.min or 1)
            elif role == gui.TableValueRole:
                return val
            elif role == gui.TableVariable:
                return val.variable

        return self._other_data.get((index.row(), index.column(), role), None)

    def setData(self, index, variant, role):
        self._other_data[index.row(), index.column(), role] = variant
        self.emit(QtCore.SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
                  index, index)

    def index(self, row, col, parent=QtCore.QModelIndex()):
        return self.createIndex(row, col, 0)

    def parent(self, index):
        return QtCore.QModelIndex()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return max([len(self.examples)] +
                       [row for row, _, _ in self._other_data.keys()])

    def columnCount(self, index=QtCore.QModelIndex()):
        return self.n_cols

    def is_sparse(self, col):
        return (
            col < self.n_attr_cols and self.X_density > Storage.DENSE
            or
            self.n_attr_cols <= col < self.n_attr_class_cols
            and self.Y_density > Storage.DENSE
            or
            self.n_attr_class_cols < col and self.metas_density > Storage.DENSE)

    @safe_call
    def headerData(self, section, orientation, role):
        display_role = role == QtCore.Qt.DisplayRole
        if orientation == QtCore.Qt.Vertical:
            return section + 1 if display_role else None
        if self.is_sparse(section):
            return None

        attr = self.all_attrs[section]
        if role == QtCore.Qt.DisplayRole:
            if self.show_attr_labels:
                return attr.name + "\n".join(
                    str(attr.attributes.get(label, ""))
                    for label in self.attr_labels)
            else:
                return attr.name
        if role == QtCore.Qt.ToolTipRole:
            pairs = [(key, str(attr.attributes[key]))
                     for key in self.attr_labels if key in attr.attributes]
            tip = "<b>%s</b>" % attr.name
            tip = "<br>".join([tip] + ["%s = %s" % pair for pair in pairs])
            return tip

        return None

    def sort(self, column, order=QtCore.Qt.AscendingOrder):
        if self.is_sparse(column):
            return
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        attr = self.all_attrs[column]
        values = [(ex[attr], i) for i, ex in enumerate(self.examples)]
        values = sorted(values,
                        key=lambda t: t[0] if not isnan(t[0]) else sys.maxsize,
                        reverse=(order != QtCore.Qt.AscendingOrder))
        self.sorted_map = [v[1] for v in values]
        self.emit(QtCore.SIGNAL("layoutChanged()"))
        self.emit(QtCore.SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
                  self.index(0, 0),
                  self.index(len(self.examples) - 1, len(self.all_attrs) - 1)
                  )


#noinspection PyArgumentList
class TableViewWithCopy(QtGui.QTableView):
    def dataChanged(self, a, b):
        super().dataChanged(a, b)

    def keyPressEvent(self, event):
        if event == QtGui.QKeySequence.Copy:
            sel_model = self.selectionModel()
            #noinspection PyBroadException
            try:
                self.copy_selection_to_clipboard(sel_model)
            except Exception:
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            return QtGui.QTableView.keyPressEvent(self, event)

    def copy_selection_to_clipboard(self, selection_model):
        """Copy table selection to the clipboard.
        """
        # TODO: html/rtf table
        import csv
        from io import StringIO
        rows = selection_model.selectedRows(0)
        csv_str = StringIO()
        csv_writer = csv.writer(csv_str, dialect="excel")
        tsv_str = StringIO()
        tsv_writer = csv.writer(tsv_str, dialect="excel-tab")
        for row in rows:
            line = []
            for i in range(self.model().columnCount()):
                index = self.model().index(row.row(), i)
                val = index.data(QtCore.Qt.DisplayRole)
                line.append(str(val))
            csv_writer.writerow(line)
            tsv_writer.writerow(line)

        csv_lines = csv_str.getvalue()
        tsv_lines = tsv_str.getvalue()

        mime = QtCore.QMimeData()
        mime.setData("text/csv", QtCore.QByteArray(csv_lines))
        mime.setData("text/tab-separated-values", QtCore.QByteArray(tsv_lines))
        mime.setData("text/plain", QtCore.QByteArray(tsv_lines))
        QtGui.QApplication.clipboard().setMimeData(mime,
                                                   QtGui.QClipboard.Clipboard)


class OWDataTable(widget.OWWidget):
    name = "Data Table"
    description = "Shows data in a spreadsheet."
    long_description = """Data Table takes one or more data sets
    on its input and shows them in a tabular format."""
    icon = "icons/Table.svg"
    priority = 100
    author = "Ales Erjavec"
    author_email = "ales.erjavec(@at@)fri.uni-lj.si"
    inputs = [("Data", Table, "dataset", Multiple + Default)]
    outputs = [("Selected Data", Table, Default),
               ("Other Data", Table)]

    show_distributions = Setting(True)
    dist_color_RGB = Setting((220, 220, 220, 255))
    show_attribute_labels = Setting(True)
    auto_commit = Setting(False)
    selected_schema_index = Setting(0)
    color_by_class = Setting(True)

    def __init__(self):
        super().__init__()

        self.data = {}          # key: id, value: ExampleTable
        self.dist_color = QtGui.QColor(*self.dist_color_RGB)
        self.locale = QtCore.QLocale()
        self.color_settings = None
        self.selected_schema_index = 0
        self.color_by_class = True

        info_box = gui.widgetBox(self.controlArea, "Info")
        self.info_ex = gui.widgetLabel(info_box, 'No data on input.')
        self.info_attr = gui.widgetLabel(info_box, ' ')
        self.info_class = gui.widgetLabel(info_box, ' ')
        self.info_meta = gui.widgetLabel(info_box, ' ')
        gui.separator(info_box)
        gui.button(info_box, self, "Restore Original Order",
                   callback=self.reset_sort_clicked,
                   tooltip="Show rows in the original order")
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
                     callback=self.cbShowDistributions)
        gui.checkBox(box, self, "color_by_class", 'Color by instance classes',
                     callback=self.cbShowDistributions)
        gui.button(box, self, "Set colors", self.set_colors,
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
        self.id2table = {}  # key: widget id, value: table
        self.table2id = {}  # key: table, value: widget id
        self.tabs.currentChanged.connect(self.tabClicked)
        self.selectionChangedFlag = False

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

    def dataset(self, data, tid=None):
        """Generates a new table and adds it to a new tab when new data arrives;
        or hides the table and removes a tab when data==None;
        or replaces the table when new data arrives together with already
        existing id."""
        if data is not None:  # can be an empty table!
            if tid in self.data:
                # remove existing table
                self.data.pop(tid)
                self.id2table[tid].hide()
                self.tabs.removeTab(self.tabs.indexOf(self.id2table[tid]))
                self.table2id.pop(self.id2table.pop(tid))
            self.data[tid] = data

            table = TableViewWithCopy()     # QTableView()
            table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
            table.setSortingEnabled(True)
            table.setHorizontalScrollMode(QtGui.QTableWidget.ScrollPerPixel)
            table.horizontalHeader().setMovable(True)
            table.horizontalHeader().setClickable(True)
            table.horizontalHeader().setSortIndicatorShown(False)

            option = table.viewOptions()
            size = table.style().sizeFromContents(
                QtGui.QStyle.CT_ItemViewItem, option,
                QtCore.QSize(20, 20), table)

            table.verticalHeader().setDefaultSectionSize(size.height() + 2)

            self.id2table[tid] = table
            self.table2id[table] = tid
            tab_name = getattr(data, "name", "")
            if tab_name:
                tab_name += " "
            tab_name += "(" + str(tid[1]) + ")"
            if tid[2] is not None:
                tab_name += " [" + str(tid[2]) + "]"
            self.tabs.addTab(table, tab_name)

            self.progressBarInit()
            self.setTable(table, data)
            self.progressBarFinished()
            self.tabs.setCurrentIndex(self.tabs.indexOf(table))
            self.setInfo(data)
            self.send_button.setEnabled(not self.auto_commit)

        elif tid in self.data:
            table = self.id2table[tid]
            self.data.pop(tid)
            table.hide()
            self.tabs.removeTab(self.tabs.indexOf(table))
            self.table2id.pop(self.id2table.pop(tid))
            self.setInfo(self.data.get(self.table2id.get(
                self.tabs.currentWidget(), None), None))

        if not len(self.data):
            self.send_button.setEnabled(False)

    #TODO Implement
    def sendReport(self):
        """
        qTableInstance = self.tabs.currentWidget()
        id = self.table2id.get(qTableInstance, None)
        data = self.data.get(id, None)
        self.reportData(data)
        table = self.id2table[id]
        import OWReport
        self.reportRaw(OWReport.reportTable(table))
        """

    # Writes data into table, adjusts the column width.
    def setTable(self, table, data):
        if data is None:
            return
        QtGui.qApp.setOverrideCursor(QtCore.Qt.WaitCursor)
        table.oldSortingIndex = -1
        table.oldSortingOrder = 1

        datamodel = ExampleTableModel(data, self)
        color_schema = self.discPalette if self.color_by_class else None
        if self.show_distributions:
            table.setItemDelegate(gui.TableBarItem(
                self, color=self.dist_color, color_schema=color_schema))
        else:
            table.setItemDelegate(QtGui.QStyledItemDelegate(self))
        table.setModel(datamodel)

        def p():
            try:
                table.updateGeometries()
                table.viewport().update()
            except RuntimeError:
                pass

        size = table.verticalHeader().sectionSizeHint(0)
        table.verticalHeader().setDefaultSectionSize(size)
        self.connect(datamodel, QtCore.SIGNAL("layoutChanged()"),
                     lambda *args: QtCore.QTimer.singleShot(50, p))

        # set the header (attribute names)
        self.draw_attribute_labels(table)

        self.connect(table.horizontalHeader(),
                     QtCore.SIGNAL("sectionClicked(int)"), self.sort_by_column)
        self.connect(
            table.selectionModel(),
            QtCore.SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
            self.update_selection)

        QtGui.qApp.restoreOverrideCursor()

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

    def sort_by_column(self, index):
        table = self.tabs.currentWidget()
        table.horizontalHeader().setSortIndicatorShown(1)
        if index == table.oldSortingIndex:
            order = (table.oldSortingOrder == QtCore.Qt.AscendingOrder and
                     QtCore.Qt.DescendingOrder or QtCore.Qt.AscendingOrder)
        else:
            order = QtCore.Qt.AscendingOrder
        table.sortByColumn(index, order)
        table.oldSortingIndex = index
        table.oldSortingOrder = order

    def tabClicked(self, index):
        """Updates the info box when a tab is clicked."""
        qTableInstance = self.tabs.widget(index)
        tid = self.table2id.get(qTableInstance, None)
        self.setInfo(self.data.get(tid, None))
        self.update_selection()

    def draw_attribute_labels(self, table):
        table.model().show_attr_labels = bool(self.show_attribute_labels)
        if self.show_attribute_labels:
            labelnames = set()
            for a in table.model().examples.domain:
                labelnames.update(a.attributes.keys())
            labelnames = sorted(list(labelnames))
            self.set_corner_text(table, "\n".join([""] + labelnames))
        else:
            self.set_corner_text(table, "")
        table.repaint()

    def c_show_attribute_labels_clicked(self):
        for table in self.table2id:
            self.draw_attribute_labels(table)

    def cbShowDistributions(self):
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
    def reset_sort_clicked(self):
        table = self.tabs.currentWidget()
        if table:
            tid = self.table2id[table]
            data = self.data[tid]
            table.horizontalHeader().setSortIndicatorShown(False)
            self.progressBarInit()
            self.setTable(table, data)
            self.progressBarFinished()

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

    def setInfo(self, data):
        """
        Updates data info.
        """
        def sp(l):
            n = len(l)
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
            descriptions = datacaching.getCached(
                data, self.__compute_density, (data, ))
            out_i = "%s instance%s" % sp(data)
            if descriptions is self.__no_missing:
                out_i += " (no missing values)"
            self.info_ex.setText(out_i)

            self.info_attr.setText("%s feature%s" %
                                   sp(data.domain.attributes) + descriptions[0])

            self.info_meta.setText("%s meta attribute%s" %
                                   sp(data.domain.metas) + descriptions[2])

            if not data.domain.class_vars:
                out_c = 'No target variable.'
            else:
                if len(data.domain.class_vars) > 1:
                    out_c = "%s outcome%s" % sp(data.domain.class_vars)
                elif isinstance(data.domain.class_var, ContinuousVariable):
                    out_c = 'Continuous target variable'
                else:
                    out_c = 'Discrete class with %s value%s' % sp(
                        data.domain.class_var.values)
                out_c += descriptions[1]
            self.info_class.setText(out_c)

    def update_selection(self, *_):
        self.send_button.setEnabled(bool(self.get_current_selection())
                                    and not self.auto_commit)
        self.commit_if()

    def get_current_selection(self):
        table = self.tabs.currentWidget()
        if table and table.model():
            model = table.model()
            new = table.selectionModel().selectedIndexes()
            return sorted(set([model.sorted_map[ind.row()] for ind in new]))

    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.selectionChangedFlag = True

    def commit(self):
        selected_data = other_data = None
        table = self.tabs.currentWidget()
        if table and table.model():
            model = table.model()
            selection = self.get_current_selection()

            # Avoid a copy if all/none rows are selected.
            if not selection:
                selected_data = None
                other_data = model.examples
            elif len(selection) == len(model.examples):
                selected_data = model.examples
                other_data = None
            else:
                selected_data = model.examples[selection]
                selection = set(selection)

                other = [i for i in range(len(model.examples))
                         if i not in selection]
                other_data = model.examples[other]

        self.send("Selected Data", selected_data)
        self.send("Other Data", other_data)

        self.selectionChangedFlag = False


if __name__ == "__main__":
    a = QtGui.QApplication(sys.argv)
    ow = OWDataTable()

    data = Table("iris")

    ow.show()
    ow.dataset(data, data.name)
    a.exec()
    ow.saveSettings()
