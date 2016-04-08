import os

import pkg_resources
import pickle
from enum import IntEnum
from PyQt4.QtCore import Qt, pyqtSlot
from PyQt4.QtGui import (QApplication, QDialog, QPrinter, QIcon, QCursor,
                         QPrintDialog, QFileDialog, QTableView,
                         QStandardItemModel, QStandardItem, QHeaderView)
from Orange.widgets import gui
from Orange.widgets.webview import WebView
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.canvas.application.canvasmain import CanvasMainWindow
from Orange.canvas.gui.utils import message_critical


class Column(IntEnum):
    item = 0
    remove = 1
    scheme = 2


class ReportItem(QStandardItem):
    def __init__(self, name, html, scheme, module, icon_name, comment=""):
        self.name = name
        self.html = html
        self.scheme = scheme
        self.module = module
        self.icon_name = icon_name
        self.comment = comment
        try:
            path = pkg_resources.resource_filename(module, icon_name)
        except ImportError:
            path = ""
        except ValueError:
            path = ""
        icon = QIcon(path)
        self.id = id(icon)
        super().__init__(icon, name)


class ReportItemModel(QStandardItemModel):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent)

    def add_item(self, item):
        row = self.rowCount()
        self.setItem(row, Column.item, item)
        self.setItem(row, Column.remove, self._icon_item("Remove"))
        self.setItem(row, Column.scheme, self._icon_item("Open Scheme"))

    def get_item_by_id(self, item_id):
        for i in range(self.rowCount()):
            item = self.item(i)
            if str(item.id) == item_id:
                return item
        return None

    @staticmethod
    def _icon_item(tooltip):
        item = QStandardItem()
        item.setEditable(False)
        item.setToolTip(tooltip)
        return item


class ReportTable(QTableView):
    def __init__(self, parent):
        super().__init__(parent)
        self._icon_remove = QIcon(pkg_resources.resource_filename(
            __name__, "icons/delete.svg"))
        self._icon_scheme = QIcon(pkg_resources.resource_filename(
            __name__, "icons/scheme.svg"))

    def mouseMoveEvent(self, event):
        self._clear_icons()
        self._repaint(self.indexAt(event.pos()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            super().mouseReleaseEvent(event)
        self._clear_icons()
        self._repaint(self.indexAt(event.pos()))

    def leaveEvent(self, _):
        self._clear_icons()

    def _repaint(self, index):
        row, column = index.row(), index.column()
        if column in (Column.remove, Column.scheme):
            self.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))
        if row >= 0:
            self.model().item(row, Column.remove).setIcon(self._icon_remove)
            self.model().item(row, Column.scheme).setIcon(self._icon_scheme)

    def _clear_icons(self):
        model = self.model()
        for i in range(model.rowCount()):
            model.item(i, Column.remove).setIcon(QIcon())
            model.item(i, Column.scheme).setIcon(QIcon())


class OWReport(OWWidget):
    name = "Report"
    save_dir = Setting("")
    open_dir = Setting("")

    def __init__(self):
        super().__init__()
        self._setup_ui_()
        index_file = pkg_resources.resource_filename(__name__, "index.html")
        self.report_html_template = open(index_file, "r").read()

    def _setup_ui_(self):
        self.table_model = ReportItemModel(0, len(Column.__members__))
        self.table = ReportTable(self.controlArea)
        self.table.setModel(self.table_model)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setWordWrap(False)
        self.table.setMouseTracking(True)
        self.table.verticalHeader().setResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.setFixedWidth(250)
        self.table.setColumnWidth(Column.item, 200)
        self.table.setColumnWidth(Column.remove, 23)
        self.table.setColumnWidth(Column.scheme, 25)
        self.table.clicked.connect(self._table_clicked)
        self.table.selectionModel().selectionChanged.connect(
            self._table_selection_changed)
        self.controlArea.layout().addWidget(self.table)

        self.last_scheme = None
        self.scheme_button = gui.button(
            self.controlArea, self, "Back to Last Scheme",
            callback=self._show_last_scheme
        )
        box = gui.hBox(self.controlArea)
        box.setContentsMargins(-6, 0, -6, 0)
        self.save_button = gui.button(
            box, self, "Save", callback=self._save_report
        )
        self.print_button = gui.button(
            box, self, "Print", callback=self._print_report
        )
        self.report_view = WebView(self.mainArea, bridge=self)

    def __getstate__(self):
        rep_dict = self.__dict__.copy()
        for key in ('_OWWidget__env', 'controlArea', 'mainArea',
                    'report_view', 'table', 'table_model'):
            del rep_dict[key]
        items_len = self.table_model.rowCount()
        return rep_dict, [self.table_model.item(i) for i in range(items_len)]

    def __setstate__(self, state):
        rep_dict, items = state
        self.__dict__.update(rep_dict)
        self._setup_ui_()
        for i in range(len(items)):
            item = items[i]
            self.table_model.add_item(
                ReportItem(item.name, item.html, item.scheme,
                           item.module, item.icon_name, item.comment)
            )

    def _table_clicked(self, index):
        if index.column() == Column.remove:
            self._remove_item(index.row())
            indexes = self.table.selectionModel().selectedIndexes()
            if indexes:
                item = self.table_model.item(indexes[0].row())
                self._scroll_to_item(item)
                self._change_selected_item(item)
        if index.column() == Column.scheme:
            self._show_scheme(index.row())

    def _table_selection_changed(self, new_selection, _):
        if new_selection.indexes():
            item = self.table_model.item(new_selection.indexes()[0].row())
            self._scroll_to_item(item)
            self._change_selected_item(item)

    def _remove_item(self, row):
        self.table_model.removeRow(row)
        self._build_html()

    def _add_item(self, widget):
        name = widget.get_widget_name_extension()
        name = "{} - {}".format(widget.name, name) if name else widget.name
        item = ReportItem(name, widget.report_html, self._get_scheme(),
                          widget.__module__, widget.icon)
        self.table_model.add_item(item)
        return item

    def _build_html(self):
        html = self.report_html_template
        html += "<body>"
        for i in range(self.table_model.rowCount()):
            item = self.table_model.item(i)
            html += "<div id='{}' class='normal' " \
                    "onClick='pybridge._select_item(this.id)'>{}<div " \
                    "class='textwrapper'><textarea required='required' " \
                    "placeholder='Write a comment...'" \
                    "onInput='pybridge._add_comment(this.parentNode." \
                    "parentNode.id, this.value)'>{}</textarea></div>" \
                    "</div>".format(item.id, item.html, item.comment)
        html += "</body></html>"
        self.report_view.setHtml(html)

    def _scroll_to_item(self, item):
        self.report_view.evalJS("document.getElementById('{}')."
                                "scrollIntoView();".format(item.id))

    def _change_selected_item(self, item):
        self.report_view.evalJS(
            "var sel_el = document.getElementsByClassName('selected')[0]; "
            "if (sel_el.id != {}) "
            "   sel_el.className = 'normal';".format(item.id))
        self.report_view.evalJS("document.getElementById('{}')."
                                "className = 'selected';".format(item.id))

    @pyqtSlot(str)
    def _select_item(self, item_id):
        item = self.table_model.get_item_by_id(item_id)
        self.table.selectRow(self.table_model.indexFromItem(item).row())
        self._change_selected_item(item)

    @pyqtSlot(str, str)
    def _add_comment(self, item_id, value):
        item = self.table_model.get_item_by_id(item_id)
        item.comment = value

    def make_report(self, widget):
        item = self._add_item(widget)
        self._build_html()
        self._scroll_to_item(item)
        self.table.selectRow(self.table_model.rowCount() - 1)

    def _get_scheme(self):
        canvas = self.get_canvas_instance()
        return canvas.get_scheme_xml() if canvas else None

    def _show_scheme(self, row):
        scheme = self.table_model.item(row).scheme
        canvas = self.get_canvas_instance()
        if canvas:
            document = canvas.current_document()
            if document.isModifiedStrict():
                self.last_scheme = canvas.get_scheme_xml()
            canvas.load_scheme_xml(scheme)

    def _show_last_scheme(self):
        if self.last_scheme:
            canvas = self.get_canvas_instance()
            if canvas:
                canvas.load_scheme_xml(self.last_scheme)

    def _save_report(self):
        filename = QFileDialog.getSaveFileName(
            self, "Save Report", self.save_dir,
            "HTML (*.html);;PDF (*.pdf);;Report (*.report)")
        if not filename:
            return

        self.save_dir = os.path.dirname(filename)
        self.saveSettings()
        _, extension = os.path.splitext(filename)
        if extension == ".pdf":
            printer = QPrinter()
            printer.setPageSize(QPrinter.A4)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(filename)
            self.report_view.print_(printer)
        elif extension == ".report":
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            frame = self.report_view.page().currentFrame()
            with open(filename, "w") as f:
                f.write(frame.documentElement().toInnerXml())

    def _print_report(self):
        printer = QPrinter()
        print_dialog = QPrintDialog(printer, self)
        print_dialog.setWindowTitle("Print report")
        if print_dialog.exec_() != QDialog.Accepted:
            return
        self.report_view.print_(printer)

    def open_report(self):
        filename = QFileDialog.getOpenFileName(
            self, "Open Report", self.open_dir, "Report (*.report)")
        if not filename:
            return

        self.open_dir = os.path.dirname(filename)
        self.saveSettings()

        try:
            with open(filename, 'rb') as f:
                report = pickle.load(f)
        except (IOError, AttributeError) as e:
            message_critical(
                 self.tr("Could not load an Orange Report file"),
                 title=self.tr("Error"),
                 informative_text=self.tr("An unexpected error occurred "
                                          "while loading '%s'.") % filename,
                 exc_info=True,
                 parent=self)
            return
        self.set_instance(report)
        self = report
        self._build_html()
        self.table.selectRow(0)
        self.show()
        self.raise_()

    @staticmethod
    def set_instance(report):
        app_inst = QApplication.instance()
        app_inst._report_window = report

    @staticmethod
    def get_instance():
        app_inst = QApplication.instance()
        if not hasattr(app_inst, "_report_window"):
            report = OWReport()
            app_inst._report_window = report
            app_inst.sendPostedEvents(report, 0)
            app_inst.aboutToQuit.connect(report.deleteLater)
        return app_inst._report_window

    @staticmethod
    def get_canvas_instance():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, CanvasMainWindow):
                return widget


if __name__ == "__main__":
    import sys
    from Orange.data import Table
    from Orange.widgets.data.owfile import OWFile
    from Orange.widgets.data.owtable import OWDataTable
    from Orange.widgets.data.owdiscretize import OWDiscretize
    from Orange.widgets.classify.owrandomforest import OWRandomForest

    iris = Table("iris")
    app = QApplication(sys.argv)

    main = OWReport.get_instance()
    file = OWFile()
    file.create_report_html()
    main.make_report(file)

    table = OWDataTable()
    table.set_dataset(iris)
    table.create_report_html()
    main.make_report(table)

    main = OWReport.get_instance()
    disc = OWDiscretize()
    disc.create_report_html()
    main.make_report(disc)

    learner = OWRandomForest()
    learner.create_report_html()
    main.make_report(learner)

    main.show()
    main.saveSettings()
    assert main.table_model.rowCount() == 4

    sys.exit(app.exec_())
