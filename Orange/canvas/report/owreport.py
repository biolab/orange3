import os
import pkg_resources
from PyQt4.QtCore import Qt
from PyQt4.QtGui import (QApplication, QDialog, QPrinter, QIcon,
                         QPrintDialog, QFileDialog, QMenu)
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.canvas.application.canvasmain import CanvasMainWindow


class OWReport(OWWidget):
    name = "Report"
    save_dir = Setting("")

    def __init__(self):
        super().__init__()
        # TODO - ko kliknes na webview, oznaci item
        self.widget_list_items = []
        self.widget_list = gui.listBox(
            self.controlArea, self,
            labels="widget_list_items", callback=self._reload,
            enableDragDrop=True, dragDropCallback=self._reload
        )
        self.widget_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.widget_list.customContextMenuRequested.connect(self._show_menu)

        self.save_button = gui.button(
            self.controlArea, self, "Save",
            callback=self._save_report, default=True
        )
        self.print_button = gui.button(
            self.controlArea, self, "Print", callback=self._print_report
        )

        self.widget_list_items_schemas = {}

        self.report_view_items = {}
        self.report_view = gui.WebviewWidget(self.mainArea)
        frame = self.report_view.page().mainFrame()
        frame.setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAsNeeded)
        self.javascript = frame.evaluateJavaScript

        index_file = pkg_resources.resource_filename(__name__, "index.html")
        self.report_html_template = open(index_file, "r").read()

    def _reload(self):
        self._build_html()

    def _show_menu(self, pos):
        widget_list_menu = QMenu(self)
        widget_list_menu.addAction("Show scheme", self._show_scheme)
        widget_list_menu.addAction("Remove", self._remove_widget_item)
        widget_list_menu.addAction("Remove All", self._clear)
        widget_list_menu.popup(self.mapToGlobal(pos))

    def _show_scheme(self):
        selected_row = self.widget_list.currentRow()
        if selected_row >= 0:
            selected_item = self.widget_list_items[selected_row]
            scheme = self.widget_list_items_schemas[selected_item]
            canvas = self.get_canvas_instance()
            if canvas:
                canvas.load_scheme_xml(scheme)

    def _get_scheme(self):
        canvas = self.get_canvas_instance()
        return canvas.get_scheme_xml() if canvas else None

    def _clear(self):
        self.widget_list_items = []
        self.report_view_items = {}
        self._build_html()

    def _remove_widget_item(self):
        selected_row = self.widget_list.currentRow()
        if selected_row >= 0:
            items = self.widget_list_items
            selected_item = items.pop(selected_row)
            self.widget_list_items = items
            del self.report_view_items[selected_item]
            if selected_row < len(self.report_view_items):
                self.widget_list.setCurrentRow(selected_row)
            self._build_html()

    def _add_widget_item(self, widget):
        items = self.widget_list_items
        path = pkg_resources.resource_filename(widget.__module__, widget.icon)
        icon = QIcon(path)
        items.append((widget.name, icon))
        self.report_view_items[(widget.name, icon)] = widget.report_html
        self.widget_list_items = items
        self.widget_list_items_schemas[(widget.name,
                                        icon)] = self._get_scheme()

    def _build_html(self):
        n_widgets = len(self.widget_list_items)
        if not n_widgets:
            return

        selected_row = self.widget_list.currentRow()
        if selected_row < 0 and n_widgets:
            selected_row = n_widgets - 1
            self.widget_list.setCurrentRow(selected_row)

        html = self.report_html_template
        html += "<body>"
        for i, (item_name, item_icon) in enumerate(self.widget_list_items):
            html += "<div id='%s' class='%s'>%s</div>" % (
                id(item_icon),
                "selected" if i == selected_row else "normal",
                self.report_view_items[(item_name, item_icon)]
            )
        html += "</body></html>"
        self.report_view.setHtml(html)

        if selected_row < len(self.widget_list_items):
            self.javascript(
                "document.getElementById('%s').scrollIntoView();"
                % id(self.widget_list_items[selected_row][1]))

    def make_report(self, widget):
        self._add_widget_item(widget)
        self._build_html()

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
    file = OWFile()
    file.create_report_html()
    main.make_report(file)

    rf = OWRandomForest()
    rf.set_data(iris)
    rf.create_report_html()
    main.make_report(rf)

    main.show()
    main.saveSettings()
    assert len(main.widget_list_items) == 5

    sys.exit(app.exec_())
