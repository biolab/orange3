import os
import pkg_resources
from PyQt4.QtCore import Qt, pyqtSlot
from PyQt4.QtGui import (QApplication, QDialog, QPrinter, QIcon, QFont,
                         QPrintDialog, QFileDialog, QTableView, QCursor,
                         QStandardItemModel, QStandardItem, QHeaderView)
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.canvas.application.canvasmain import CanvasMainWindow


class ReportItem(QStandardItem):
    def __init__(self, icon, text, html, scheme):
        super().__init__(icon, text)
        self.id = id(icon)
        self.html = html
        self.scheme = scheme
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)


class ReportItemModel(QStandardItemModel):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent)

    def add_item(self, item):
        row = self.rowCount()
        self.setItem(row, 0, item)
        self.setItem(row, 1, self._icon_item("Remove", "icons/delete.png"))
        self.setItem(row, 2, self._icon_item("Scheme", "icons/scheme.png"))

    def get_item_by_id(self, item_id):
        for i in range(self.rowCount()):
            item = self.item(i)
            if str(item.id) == item_id:
                return item
        return None

    @staticmethod
    def _icon_item(tooltip, path):
        path = pkg_resources.resource_filename(__name__, path)
        item = QStandardItem()
        item.setIcon(QIcon(path))
        item.setEditable(False)
        item.setToolTip(tooltip)
        return item


class OWReport(OWWidget):
    name = "Report"
    save_dir = Setting("")

    def __init__(self):
        super().__init__()
        self.table_model = ReportItemModel(0, 3)
        self.table = QTableView(self.controlArea)
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
        self.table.setFixedWidth(252)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 25)
        self.table.setColumnWidth(2, 25)
        self.table.clicked.connect(self._table_clicked)
        self.table.entered.connect(self._table_entered)
        self.controlArea.layout().addWidget(self.table)

        self.save_button = gui.button(
            self.controlArea, self, "Save", callback=self._save_report
        )
        self.print_button = gui.button(
            self.controlArea, self, "Print", callback=self._print_report
        )

        self.report_view = gui.WebviewWidget(self.mainArea, bridge=self)
        index_file = pkg_resources.resource_filename(__name__, "index.html")
        self.report_html_template = open(index_file, "r").read()

    def _table_clicked(self, index):
        if index.column() == 0:
            item = self.table_model.item(index.row())
            self._scroll_to_item(item)
            self._change_selected_item(item)
        if index.column() == 1:
            self._remove_item(index.row())
        if index.column() == 2:
            self._show_scheme(index.row())

    def _table_entered(self, index):
        if index.column() in (1, 2):
            self.table.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.table.setCursor(QCursor(Qt.ArrowCursor))

    def _show_scheme(self, row):
        scheme = self.table_model.item(row).scheme
        canvas = self.get_canvas_instance()
        if canvas:
            canvas.load_scheme_xml(scheme)

    def _get_scheme(self):
        canvas = self.get_canvas_instance()
        return canvas.get_scheme_xml() if canvas else None

    def _remove_item(self, row):
        self.table_model.removeRow(row)
        self._build_html()
        indexes = self.table.selectionModel().selectedIndexes()
        if indexes:
            item = self.table_model.item(indexes[0].row())
            self._scroll_to_item(item)
            self._change_selected_item(item)

    def _add_item(self, widget, name=None):
        path = pkg_resources.resource_filename(widget.__module__, widget.icon)
        item = ReportItem(QIcon(path), name if name else widget.name,
                          widget.report_html, self._get_scheme())
        self.table_model.add_item(item)
        return item

    def _build_html(self):
        html = self.report_html_template
        html += "<body>"
        for i in range(self.table_model.rowCount()):
            item = self.table_model.item(i)
            html += "<div id='%s' class='normal' " \
                    "onClick='pybridge._select_item(this.id)'>%s</div>" % \
                    (item.id, item.html)
        html += "</body></html>"
        self.report_view.setHtml(html)

    def _scroll_to_item(self, item):
        self.report_view.evalJS("document.getElementById('%s')."
                                "scrollIntoView();" % item.id)

    def _change_selected_item(self, item):
        self.report_view.evalJS("document.getElementsByClassName('selected')"
                                "[0].className = 'normal';")
        self.report_view.evalJS("document.getElementById('%s')."
                                "className = 'selected';" % item.id)

    @pyqtSlot(str)
    def _select_item(self, item_id):
        item = self.table_model.get_item_by_id(item_id)
        self.table.selectRow(self.table_model.indexFromItem(item).row())
        self._change_selected_item(item)

    def make_report(self, widget, name=None):
        item = self._add_item(widget, name)
        self._build_html()
        self._scroll_to_item(item)
        self._change_selected_item(item)
        self.table.selectRow(self.table_model.rowCount() - 1)

    def _save_report(self):
        filename = QFileDialog.getSaveFileName(self, "Save Report",
                                               self.save_dir,
                                               "HTML (*.html);;PDF (*.pdf)")
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
    zoo = Table("zoo")
    app = QApplication(sys.argv)

    main = OWReport.get_instance()
    file = OWFile()
    file.create_report_html()
    main.make_report(file)

    table = OWDataTable()
    table.create_report_html()
    main.make_report(table)

    main = OWReport.get_instance()
    disc = OWDiscretize()
    disc.set_data(zoo)
    disc.create_report_html()
    main.make_report(disc)

    rf = OWRandomForest()
    rf.set_data(iris)
    rf.create_report_html()
    main.make_report(rf)

    main.show()
    main.saveSettings()
    assert main.table_model.rowCount() == 4

    sys.exit(app.exec_())
