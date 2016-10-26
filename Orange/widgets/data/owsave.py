import os.path
from operator import attrgetter

from PyQt4 import QtGui

from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.data.io import FileFormat
from Orange.widgets.utils import filedialogs


class OWSave(widget.OWWidget):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    category = "Data"
    keywords = ["data", "save"]

    inputs = [("Data", Table, "dataset")]

    want_main_area = False
    resizing_enabled = False

    last_dir = Setting("")
    last_filter = Setting("")
    auto_save = Setting(False)

    formats = [(f.DESCRIPTION, f.EXTENSIONS)
               for f in sorted(set(FileFormat.writers.values()),
                               key=attrgetter("PRIORITY"))]
    filters = ['{} (*{})'.format(x[0], ' *'.join(x[1])) for x in formats]

    def __init__(self):
        super().__init__()
        self.data = None
        self.filename = ""
        self.writer = None

        self.save = gui.auto_commit(
            self.controlArea, self, "auto_save", "Save", box=False,
            commit=self.save_file, callback=self.adjust_label,
            disabled=True, addSpace=True)
        self.saveAs = gui.button(
            self.controlArea, self, "Save As...",
            callback=self.save_file_as, disabled=True)
        self.saveAs.setMinimumWidth(220)
        self.adjustSize()

    def adjust_label(self):
        if self.filename:
            filename = os.path.split(self.filename)[1]
            text = ["Save as '{}'", "Auto save as '{}'"][self.auto_save]
            self.save.button.setText(text.format(filename))

    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.saveAs.setDisabled(data is None)
        if data is not None:
            self.save_file()

    def save_file_as(self):
        file_name = self.filename or \
            os.path.join(self.last_dir or os.path.expanduser("~"),
                         getattr(self.data, 'name', ''))
        filename, writer, filter = filedialogs.get_file_name(
                file_name, self.last_filter, FileFormat.writers)
        if not filename:
            return
        self.filename = filename
        self.writer = writer
        self.unconditional_save_file()
        self.last_dir = os.path.split(self.filename)[0]
        self.last_filter = filter
        self.adjust_label()

    def save_file(self):
        if self.data is None:
            return
        if not self.filename:
            self.save_file_as()
        else:
            try:
                self.writer.write(self.filename, self.data)
            except Exception as errValue:
                self.error(str(errValue))
            else:
                self.error()


if __name__ == "__main__":
    import sys

    a = QtGui.QApplication(sys.argv)
    table = Table("iris")
    ow = OWSave()
    ow.show()
    ow.dataset(table)
    a.exec()
    ow.saveSettings()
