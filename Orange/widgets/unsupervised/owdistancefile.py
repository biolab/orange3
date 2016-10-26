import os, sys

from PyQt4 import QtGui, QtCore

from Orange.misc import DistMatrix
from Orange.widgets import widget, gui
from Orange.data import get_sample_datasets_dir
from Orange.widgets.utils.filedialogs import RecentPathsWComboMixin


class OWDistanceFile(widget.OWWidget, RecentPathsWComboMixin):
    name = "Distance File"
    id = "orange.widgets.unsupervised.distancefile"
    description = "Read distances from a file."
    icon = "icons/DistanceFile.svg"
    priority = 10
    category = "Data"
    keywords = ["data", "distances", "load", "read"]
    outputs = [("Distances", DistMatrix)]

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        RecentPathsWComboMixin.__init__(self)
        self.loaded_file = ""

        vbox = gui.vBox(self.controlArea, "Distance File", addSpace=True)
        box = gui.hBox(vbox)
        self.file_combo.setMinimumWidth(300)
        box.layout().addWidget(self.file_combo)
        self.file_combo.activated[int].connect(self.select_file)

        button = gui.button(box, self, '...', callback=self.browse_file)
        button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        button.setSizePolicy(
            QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)

        button = gui.button(
            box, self, "Reload", callback=self.reload, default=True)
        button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
        button.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

        box = gui.vBox(self.controlArea, "Info", addSpace=True)
        self.infoa = gui.widgetLabel(box, 'No data loaded.')
        self.warnings = gui.widgetLabel(box, ' ')
        #Set word wrap, so long warnings won't expand the widget
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(
            QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.MinimumExpanding)

        box = gui.hBox(self.controlArea)
        gui.button(
            box, self, "Browse documentation data sets",
            callback=lambda: self.browse_file(True), autoDefault=False)
        gui.rubber(box)
        box.layout().addWidget(self.report_button)
        self.report_button.setFixedWidth(170)

        self.set_file_list()
        QtCore.QTimer.singleShot(0, self.open_file)

    def set_file_list(self):
        super().set_file_list()

    def reload(self):
        return self.open_file()

    def select_file(self, n):
        super().select_file(n)
        self.set_file_list()
        self.open_file()

    def browse_file(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QtGui.QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with documentation data sets")
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open Distance File', start_file, "(*.dst)")
        if not filename:
            return
        self.add_path(filename)
        self.open_file()

    # Open a file, create data from it and send it over the data channel
    def open_file(self):
        self.clear_messages()
        fn = self.last_path()
        if not fn:
            return
        if not os.path.exists(fn):
            dir_name, basename = os.path.split(fn)
            if os.path.exists(os.path.join(".", basename)):
                fn = os.path.join(".", basename)
                self.information("Loading '{}' from the current directory."
                                 .format(basename))
        if fn == "(none)":
            self.send("Distances", None)
            self.infoa.setText("No data loaded")
            self.infob.setText("")
            self.warnings.setText("")
            return

        self.loaded_file = ""

        try:
            distances = DistMatrix.from_file(fn)
            self.loaded_file = fn
        except Exception as exc:
            err_value = str(exc)
            self.error("Invalid file format")
            self.infoa.setText('Data was not loaded due to an error.')
            self.warnings.setText(err_value)
            distances = None

        if distances is not None:
            self.infoa.setText(
                "{} points(s), ".format(len(distances)) +
                (["unlabelled", "labelled"][distances.row_items is not None]))
            self.warnings.setText("")
            file_name = os.path.split(fn)[1]
            if "." in file_name:
                distances.name = file_name[:file_name.rfind('.')]
            else:
                distances.name = file_name

        self.send("Distances", distances)

    def send_report(self):
        if not self.loaded_file:
            self.report_paragraph("No data was loaded.")
        else:
            self.report_items([("File name", self.loaded_file)])

if __name__ == "__main__":
    a = QtGui.QApplication(sys.argv)
    ow = OWDistanceFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
