import sys
from math import isnan
from functools import reduce
from PyQt4 import QtCore
from PyQt4 import QtGui

from Orange.data.table import Table
from Orange.data import ContinuousVariable
from Orange.statistics import basic_stats

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import colorpalette, datacaching
from Orange.widgets.basewidget import Multiple, Default
from Orange.widgets.gui import *


NAME = "Data Table"

DESCRIPTION = "Shows data in a spreadsheet."

LONG_DESCRIPTION = """Data Table widget takes one or more data sets
on its input and presents them in a spreadsheet format.

"""

ICON = "icons/Table.svg"

PRIORITY = 100

AUTHOR = "Ales Erjavec"

AUTHOR_EMAIL = "ales.erjavec(@at@)fri.uni-lj.si"

INPUTS = [("Data", Table, "data set", Multiple + Default)]

OUTPUTS = [("Selected Data", Table, Default),
           ("Other Data", Table)]

WIDGET_CLASS = "OWDataTable"

##############################################################################

def safe_call(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            print(func.__name__, "call error", ex, file=sys.stderr)
    return wrapper


#noinspection PyMethodOverriding
class ExampleTableModel(QtCore.QAbstractItemModel):
    def __init__(self, data, dist, *args):
        QtCore.QAbstractItemModel.__init__(self, *args)
        self.examples = data
        domain = self.domain = data.domain
        self.dist = dist
        self.nvariables = len(domain)
        self.n_attr_cols = 1 if data.X_is_sparse else len(domain.attributes)
        self.n_attr_class_cols = self.n_attr_cols + (
                           1 if data.Y_is_sparse else len(domain.class_vars))
        self.n_cols = self.n_attr_class_cols + (
                           1 if data.metas_is_sparse else len(domain.metas))
        self.all_attrs = (domain.attributes + domain.class_vars + domain.metas)
        self.cls_color = QtGui.QColor(160,160,160)
        self.meta_color = QtGui.QColor(220,220,200)
        self.sorted_map = range(len(data))

        self.attr_labels = sorted(reduce(set.union,
            [attr.attributes for attr in self.all_attrs], set()))
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
                  self.index(0,0),
                  self.index(len(self.examples) - 1, len(self.all_attrs) - 1)
                  )

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
        sp_data = vars = None
        if col < self.n_attr_cols:
            if role == QtCore.Qt.BackgroundRole:
                return
            if example.sparse_x is not None:
                sp_data, vars = example.sparse_x, self.domain.attributes
        elif self.n_attr_cols <= col < self.n_attr_class_cols:
            if role == QtCore.Qt.BackgroundRole:
                return self.cls_color
            if example.sparse_y is not None:
                sp_data, vars = example.sparse_y, self.domain.class_vars
        else:
            if role == QtCore.Qt.BackgroundRole:
                return self.meta_color
            if example.sparse_metas is not None:
                sp_data, vars = example.sparse_metas, self.domain.class_vars

        if sp_data is not None:
            if role == QtCore.Qt.DisplayRole:
                return ", ".join(
                    "{}={}".format(vars[i].name, vars[i].repr_val(v))
                    for i, v in zip(sp_data.indices, sp_data.data))
        else: #not sparse
            attr = self.all_attrs[col]
            val = example[attr]
            domain = self.examples.domain
            if role == QtCore.Qt.DisplayRole:
                return str(val)
            elif (role == TableBarItem.BarRole and
                    isinstance(attr, ContinuousVariable) and
                    not isnan(val)):
                dist = self.dist[col if col < self.nvariables else -1 - col]
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

    @safe_call
    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal:
            attr = self.all_attrs[section]
            if role ==QtCore.Qt.DisplayRole:
                values = [attr.name] + (
                    [str(attr.attributes.get(label, ""))
                     for label in self.attr_labels
                    ]
                    if self.show_attr_labels else [])
                return "\n".join(values)
            if role == QtCore.Qt.ToolTipRole:
                pairs = [(key, str(attr.attributes[key]))
                         for key in self.attr_labels if key in attr.attributes]
                tip = "<b>%s</b>" % attr.name
                tip = "<br>".join([tip] + ["%s = %s" % pair for pair in pairs])
                return tip
        else:
            if role == QtCore.Qt.DisplayRole:
                return section + 1
        return None

    def sort(self, column, order=QtCore.Qt.AscendingOrder):
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        attr = self.all_attrs[column]
        values = [(ex[attr], i) for i, ex in enumerate(self.examples)]
        values = sorted(values,
                        key=lambda t: t[0] if not isnan(t[0]) else sys.maxsize,
                        reverse=(order!=QtCore.Qt.AscendingOrder))
        self.sorted_map = [v[1] for v in values]
        self.emit(QtCore.SIGNAL("layoutChanged()"))
        self.emit(QtCore.SIGNAL("dataChanged(QModelIndex, QModelIndex)"),
                  self.index(0,0),
                  self.index(len(self.examples) - 1, len(self.all_attrs) - 1)
                  )


#noinspection PyArgumentList
class TableViewWithCopy(QtGui.QTableView):
    def dataChanged(self, a, b):
        self.resizeColumnsToContents()
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
        QtGui.QApplication.clipboard().setMimeData(mime, QtGui.QClipboard.Clipboard)


class OWDataTable(widget.OWWidget):
    show_distributions = Setting(True)
    dist_color_RGB = Setting((220, 220, 220, 255))
    show_attribute_labels = Setting(True)
    auto_commit = Setting(False)
    selected_schema_index = Setting(0)
    color_by_class = Setting(True)

    def __init__(self, parent=None, signalManager = None):
        super().__init__(parent, signalManager, "Data Table")

        self.inputs = [("Data", Table, self.dataset, Multiple + Default)]
        self.outputs = [("Selected Data", Table, Default),
                        ("Other Data", Table)]

        self.data = {}          # key: id, value: ExampleTable
        self.dist_color = QtGui.QColor(*self.dist_color_RGB)
        self.locale = QtCore.QLocale()
        self.color_settings = None
        self.selected_schema_index = 0
        self.color_by_class = True


        # info box
        info_box = gui.widgetBox(self.controlArea, "Info")
        self.info_ex = gui.widgetLabel(info_box, 'No data on input.')
        self.info_miss = gui.widgetLabel(info_box, ' ')
        gui.widgetLabel(info_box, ' ')
        self.info_attr = gui.widgetLabel(info_box, ' ')
        self.info_meta = gui.widgetLabel(info_box, ' ')
        gui.widgetLabel(info_box, ' ')
        self.info_class = gui.widgetLabel(info_box, ' ')
        gui.separator(info_box)
        gui.button(info_box, self, "Restore Original Order",
                   callback=self.reset_sort_clicked,
                   tooltip="Show rows in the original order")
        info_box.setMinimumWidth(200)


        gui.separator(self.controlArea)

        box = gui.widgetBox(self.controlArea, "Variables")
        self.c_show_attribute_labels = gui.checkBox(box, self,
            "show_attribute_labels", 'Show variable labels (if present)',
            callback=self.c_show_attribute_labels_clicked)
        self.c_show_attribute_labels.setEnabled(True)
        gui.checkBox(box, self, "show_distributions",
                     'Visualize continuous values',
                     callback=self.cbShowDistributions)
        gui.checkBox(box, self, "color_by_class", 'Color by instance classes',
                     callback=self.cbShowDistributions)
        gui.button(box, self, "Set colors", self.set_colors,
                   tooltip="Set the background color and color palette",
                           debuggingEnabled=0)

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
        self.connect(self.tabs, QtCore.SIGNAL("currentChanged(QWidget*)"), self.tabClicked)
        self.selectionChangedFlag = False


    def create_color_dialog(self):
        c = colorpalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Default", "Default color", QtGui.QColor(QtCore.Qt.white))
        c.setColorSchemas(self.color_settings, self.selected_schema_index)
        return c


    def set_colors(self):
        dlg = self.create_color_dialog()
        if dlg.exec():
            self.color_settings = dlg.getColorSchemas()
            self.selected_schema_index = dlg.selectedSchemaIndex
            self.discPalette = dlg.getDiscretePalette("discPalette")
            self.dist_color_RGB = dlg.getColor("Default")

    def dataset(self, data, id=None):
        """Generates a new table and adds it to a new tab when new data arrives;
        or hides the table and removes a tab when data==None;
        or replaces the table when new data arrives together with already existing id."""
        if data is not None:  # can be an empty table!
            if id in self.data:
                # remove existing table
                self.data.pop(id)
                self.id2table[id].hide()
                self.tabs.removeTab(self.tabs.indexOf(self.id2table[id]))
                self.table2id.pop(self.id2table.pop(id))
            self.data[id] = data

            table = TableViewWithCopy() #QTableView()
            table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
            table.setSortingEnabled(True)
            table.setHorizontalScrollMode(QtGui.QTableWidget.ScrollPerPixel)
            table.horizontalHeader().setMovable(True)
            table.horizontalHeader().setClickable(True)
            table.horizontalHeader().setSortIndicatorShown(False)

            option = table.viewOptions()
            size = table.style().sizeFromContents(QtGui.QStyle.CT_ItemViewItem, option, QtCore.QSize(20, 20), table) #QSize(20, QFontMetrics(option.font).lineSpacing()), table)

            table.verticalHeader().setDefaultSectionSize(size.height() + 2) #int(size.height() * 1.25) + 2)

            self.id2table[id] = table
            self.table2id[table] = id
            if data.name:
                tabName = "%s " % data.name
            else:
                tabName = ""
            tabName += "(" + str(id[1]) + ")"
            if id[2] is not None:
                tabName += " [" + str(id[2]) + "]"
            self.tabs.addTab(table, tabName)

            self.progressBarInit()
            self.setTable(table, data)
            self.progressBarFinished()
            self.tabs.setCurrentIndex(self.tabs.indexOf(table))
            self.setInfo(data)
            self.send_button.setEnabled(not self.auto_commit)

        elif id in self.data:
            table = self.id2table[id]
            self.data.pop(id)
            table.hide()
            self.tabs.removeTab(self.tabs.indexOf(table))
            self.table2id.pop(self.id2table.pop(id))
            self.setInfo(self.data.get(self.table2id.get(self.tabs.currentWidget(),None),None))

        if len(self.data) == 0:
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

        dist = datacaching.getCached(data, basic_stats.get_stats, (data,))
        datamodel = ExampleTableModel(data, dist, self)

        color_schema = self.discPalette if self.color_by_class else None
        if self.show_distributions:
            table.setItemDelegate(gui.TableBarItem(self,
                color=self.dist_color, color_schema=color_schema))
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

        id = self.table2id.get(table, None)

        # set the header (attribute names)
        self.draw_attribute_labels(table)

        self.connect(table.horizontalHeader(),
                     QtCore.SIGNAL("sectionClicked(int)"), self.sort_by_column)
        self.connect(table.selectionModel(),
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
                            opt.position = QtGui.QStyleOptionHeader.OnlyOneSection
                            painter = QtGui.QStylePainter(btn)
                            painter.drawControl(QtGui.QStyle.CE_Header, opt)
                            return True # eat evebt
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
                    QtGui.QStyle.CT_HeaderSection, opt, QtCore.QSize(), btn).expandedTo(
                        QtGui.QApplication.globalStrut())
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


    def tabClicked(self, qTableInstance):
        """Updates the info box when a tab is clicked."""
        id = self.table2id.get(qTableInstance,None)
        self.setInfo(self.data.get(id,None))
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
            id = self.table2id[table]
            data = self.data[id]
            table.horizontalHeader().setSortIndicatorShown(False)
            self.progressBarInit()
            self.setTable(table, data)
            self.progressBarFinished()

    def setInfo(self, data):
        """Updates data info.
        """
        def sp(l, capitalize=False):
            n = len(l)
            if n == 0:
                if capitalize:
                    return "No", "s"
                else:
                    return "no", "s"
            elif n == 1:
                return str(n), ''
            else:
                return str(n), 's'

        if data is None:
            self.info_ex.setText('No data on input.')
            self.info_miss.setText('')
            self.info_attr.setText('')
            self.info_meta.setText('')
            self.info_class.setText('')
        else:
            self.info_ex.setText("%s example%s," % sp(data))
            # TODO implement
            missData = [] #orange.Preprocessor_takeMissing(data)
            self.info_miss.setText('%s (%.1f%s) with missing values.' %
                (len(missData),
                 len(data) and 100.*len(missData)/len(data), "%"))
            self.info_attr.setText("%s attribute%s," %
                                   sp(data.domain.attributes, True))
            self.info_meta.setText("%s meta attribute%s." %
                                   sp(data.domain.metas))
            if data.domain.class_var is None:
                self.info_class.setText('No target variable.')
            elif isinstance(data.domain.class_var, ContinuousVariable):
                self.info_class.setText('Continuous target variable.')
            else:
                self.info_class.setText('Discrete class with %s value%s.'
                                        % sp(data.domain.class_var.values))

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
        table = self.tabs.currentWidget()
        if table and table.model():
            model = table.model()
            selected = self.get_current_selection()
            selection = [1 if i in selected else 0
                         for i in range(len(model.examples))]
            data = model.examples.select(selection)
            self.send("Selected Data", data if len(data) > 0 else None)
            data = model.examples.select(selection, 0)
            self.send("Other Data", data if len(data) > 0 else None)
        else:
            self.send("Selected Data", None)
            self.send("Other Data", None)

        self.selectionChangedFlag = False



if __name__=="__main__":
    a = QtGui.QApplication(sys.argv)
    ow = OWDataTable()

    d5 = Table('../../../jrs-small.basket')
#    d5 = Table('../../../jrs2012.basket')
#    d5 = Table('../../tests/iris.tab')
#    d5 = Table('../../tests/zoo.tab')
    ow.show()
    ow.dataset(d5,"adult_sample")
    a.exec()
    ow.saveSettings()
