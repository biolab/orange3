import os, sys
from PyQt4 import QtGui
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table, get_sample_datasets_dir
from Orange.data.io import FileFormats
from Orange.widgets.widget import OutputSignal


def add_origin(examples, filename):
    """Adds attribute with file location to each variable"""
    vars = examples.domain.variables + examples.domain.metas
    strings = [var for var in vars if var.is_string]
    dir_name, basename = os.path.split(filename)
    for var in strings:
        if "type" in var.attributes and "origin" not in var.attributes:
            var.attributes["origin"] = dir_name


class OWFile(widget.OWWidget):
    name = "File"
    id = "orange.widgets.data.file"
    description = "Read a data from an input file " \
                  "and send the data table to the output."
    icon = "icons/File.svg"
    author = "Janez Demsar"
    maintainer_email = "janez.demsar(@at@)fri.uni-lj.si"
    priority = 10
    category = "Data"
    keywords = ["data", "file", "load", "read"]
    outputs = [OutputSignal(
        "Data", Table,
        doc="Attribute-valued data set read from the input file.")]

    want_main_area = False

    recent_files = Setting(["(none)"])
    new_variables = Setting(False)

    dlgFormats = (
        "All readable files ({})\n".format(
            " ".join("*" + c for c in FileFormats.readers)) +
        "\n".join("{} (*{})".format(FileFormats.names[ext], ext)
                  for ext in FileFormats.readers))

    def __init__(self):
        super().__init__()
        self.domain = None
        self.recent_files = [fn for fn in self.recent_files
                             if os.path.exists(fn)]
        self.loaded_file = ""

        vbox = gui.widgetBox(self.controlArea, "Data File", addSpace=True)
        box = gui.widgetBox(vbox, orientation=0)
        self.file_combo = QtGui.QComboBox(box)
        self.file_combo.setMinimumWidth(300)
        box.layout().addWidget(self.file_combo)
        self.file_combo.activated[int].connect(self.select_file)

        button = gui.button(box, self, '...', callback=self.browse_file)
        button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        button.setSizePolicy(
            QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)

        button = gui.button(box, self, "Reload",
                            callback=self.reload, default=True)
        button.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
        button.setSizePolicy(
            QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

        gui.checkBox(vbox, self, "new_variables",
                     "Columns with same name in different files " +
                     "represent different variables")

        box = gui.widgetBox(self.controlArea, "Info", addSpace=True)
        self.infoa = gui.widgetLabel(box, 'No data loaded.')
        self.infob = gui.widgetLabel(box, ' ')
        self.warnings = gui.widgetLabel(box, ' ')
        #Set word wrap, so long warnings won't expand the widget
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(
            QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.MinimumExpanding)

        self.set_file_list()
        if len(self.recent_files) > 0:
            self.open_file(self.recent_files[0])

    def set_file_list(self):
        self.file_combo.clear()
        if not self.recent_files:
            self.file_combo.addItem("(none)")
        for file in self.recent_files:
            if file == "(none)":
                self.file_combo.addItem("(none)")
            else:
                self.file_combo.addItem(os.path.split(file)[1])
        self.file_combo.addItem("Browse documentation data sets...")

    def reload(self):
        if self.recent_files:
            return self.open_file(self.recent_files[0])

    def select_file(self, n):
        if n < len(self.recent_files) :
            name = self.recent_files[n]
            del self.recent_files[n]
            self.recent_files.insert(0, name)
        elif n:
            self.browse_file(True)

        if len(self.recent_files) > 0:
            self.set_file_list()
            self.open_file(self.recent_files[0])

    def browse_file(self, in_demos=0):
        if in_demos:
            try:
                start_file = get_sample_datasets_dir()
            except AttributeError:
                start_file = ""
            if not start_file or not os.path.exists(start_file):
                widgets_dir = os.path.dirname(gui.__file__)
                orange_dir = os.path.dirname(widgets_dir)
                start_file = os.path.join(orange_dir, "doc", "datasets")
            if not start_file or not os.path.exists(start_file):
                d = os.getcwd()
                if os.path.basename(d) == "canvas":
                    d = os.path.dirname(d)
                start_file = os.path.join(os.path.dirname(d), "doc", "datasets")
            if not os.path.exists(start_file):
                QtGui.QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with example data sets")
                return
        else:
            if self.recent_files and self.recent_files[0] != "(none)":
                start_file = self.recent_files[0]
            else:
                start_file = os.path.expanduser("~/")

        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open Orange Data File', start_file, self.dlgFormats)
        if not filename:
            return
        if filename in self.recent_files:
            self.recent_files.remove(filename)
        self.recent_files.insert(0, filename)
        self.set_file_list()
        self.open_file(self.recent_files[0])

    # Open a file, create data from it and send it over the data channel
    def open_file(self, fn):
        self.error()
        self.warning()
        self.information()

        if not os.path.exists(fn):
            dir_name, basename = os.path.split(fn)
            if os.path.exists(os.path.join(".", basename)):
                fn = os.path.join(".", basename)
                self.information("Loading '{}' from the current directory."
                                 .format(basename))
        if fn == "(none)":
            self.send("Data", None)
            self.infoa.setText("No data loaded")
            self.infob.setText("")
            self.warnings.setText("")
            return

        self.loaded_file = ""

        data = None
        err_value = None
        try:
            # TODO handle self.new_variables
            data = Table(fn)
            self.loaded_file = fn
        except Exception as exc:
            err_value = str(exc)
            if "is being loaded as" in str(err_value):
                try:
                    data = Table(fn)
                    self.loaded_file = fn
                    self.warning(0, err_value)
                except:
                    data = None
        if err_value is not None:
            self.error(err_value)
            self.infoa.setText('Data was not loaded due to an error.')
            self.infob.setText('Error:')
            self.warnings.setText(err_value)

        if data is None:
            self.dataReport = None
        else:
            domain = data.domain
            self.infoa.setText(
                "{} instance(s), {} feature(s), {} meta attributes"
                .format(len(data), len(domain.attributes), len(domain.metas)))
            if domain.has_continuous_class:
                self.infob.setText("Regression; numerical class.")
            elif domain.has_discrete_class:
                self.infob.setText("Classification; " +
                                   "discrete class with {} values."
                                   .format(len(domain.class_var.values)))
            elif data.domain.class_vars:
                self.infob.setText("Multi-target; {} target variables."
                                   .format(len(data.domain.class_vars)))
            else:
                self.infob.setText("Data has no target variable.")
            self.warnings.setText("")

            add_origin(data, fn)
            # make new data and send it
            file_name = os.path.split(fn)[1]
            if "." in file_name:
                data.name = file_name[:file_name.rfind('.')]
            else:
                data.name = file_name

            self.dataReport = self.prepareDataReport(data)
        self.send("Data", data)

    def sendReport(self):
        dataReport = getattr(self, "dataReport", None)
        if dataReport:
            self.reportSettings(
                "File",
                [("File name", self.loaded_file),
                 ("Format", self.formats.get(os.path.splitext(
                     self.loaded_file)[1], "unknown format"))])
            self.reportData(self.dataReport)

if __name__ == "__main__":
    a = QtGui.QApplication(sys.argv)
    ow = OWFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
