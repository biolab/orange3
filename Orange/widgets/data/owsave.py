import os.path

from PyQt4 import QtGui

from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.data.io import FileFormats


class OWSave(widget.OWWidget):
    name = "Save"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    author = "Martin Frlin"
    category = "Data"
    keywords = ["data", "save"]

    inputs = [("Data", Table, "dataset")]

    want_main_area = False
    resizing_enabled = False

    last_dir = Setting("")

    def __init__(self, parent=None, signalManager=None, settings=None,
                 data=None, file_formats=None):
        super().__init__(self, parent, signalManager, settings, "Save")
        self.data = None
        self.filename = ""
        self.format_index = 0
        self.file_formats = FileFormats.writers
        if file_formats:
            self.file_formats = file_formats
        self.formats = tuple((FileFormats.names[ext], ext)
                             for ext in self.file_formats)
        self.comboBoxFormat = gui.comboBox(
            self.controlArea, self, value='format_index',
            items=['{} (*{})'.format(*x) for x in self.formats],
            box='File Format', callback=self.reset_filename)
        box = gui.widgetBox(self.controlArea)
        self.save = gui.button(box, self, "Save", callback=self.save_file,
                               default=True, disabled=True)
        gui.separator(box)
        self.saveAs = gui.button(box, self, "Save as ...",
                                 callback=self.save_file_as, disabled=True)
        self.setMinimumWidth(320)
        self.adjustSize()
        if data:
            self.dataset(data)

    def reset_filename(self):
        base, ext = os.path.splitext(self.filename)
        if ext in self.file_formats:
            self.filename = base + self.formats[self.format_index][1]
            self.save.setText("Save as '%s'" % os.path.split(self.filename)[1])

    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.saveAs.setDisabled(data is None)

    def save_file_as(self):
        f = self.formats[self.format_index]
        home_dir = os.path.expanduser("~")
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save', self.filename or self.last_dir or home_dir,
            '{} (*{})'.format(*f))
        if not filename:
            return
        self.filename = filename
        if os.path.splitext(filename)[1] != f[1]:
            self.filename += f[1]
        self.last_dir, file_name = os.path.split(self.filename)
        self.save.setText("Save as '%s'" % file_name)
        self.save.setDisabled(False)
        self.save_file()

    def save_file(self):
        if not self.filename:
            self.save_file_as()
        elif self.data is not None:
            try:
                ext = self.formats[self.format_index][1]
                format = self.file_formats[ext]
                format().write(self.filename, self.data)
                self.error()
            except Exception as errValue:
                self.error(str(errValue))


if __name__ == "__main__":
    import sys

    a = QtGui.QApplication(sys.argv)
    table = Table("iris")
    ow = OWSave()
    ow.show()
    ow.dataset(table)
    a.exec()
    ow.saveSettings()
