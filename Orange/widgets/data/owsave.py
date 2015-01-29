import os.path

from PyQt4 import QtGui

from Orange.data.io import save_csv, save_tab_delimited
from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting


class OWSave(widget.OWWidget):
    name = "Save"
    description = "Saves data to a file."
    icon = "icons/Save.svg"
    author = "Martin Frlin"
    category = "Data"
    keywords = ["data", "save"]

    inputs = [("Data", Table, "dataset")]

    want_main_area = False

    format_index = Setting(0)
    last_dir = Setting("")

    formats = (('.tab', 'Tab-delimited file', save_tab_delimited),
               ('.csv', 'Comma-separated values', save_csv))

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(self, parent, signalManager, settings, "Save")
        self.data = None
        self.filename = ""
        self.comboBoxFormat = gui.comboBox(
            self.controlArea, self, value='format_index',
            items=['{} (*{})'.format(f[1], f[0]) for f in self.formats],
            box='File Format', callback=self.reset_filename)
        box = gui.widgetBox(self.controlArea)
        self.save = gui.button(box, self, "Save", callback=self.save_file,
                               default=True, disabled=True)
        gui.separator(box)
        self.saveAs = gui.button(box, self, "Save as ...",
                                 callback=self.save_file_as, disabled=True)
        self.setMinimumWidth(320)
        self.adjustSize()

    def reset_filename(self):
        if self.filename[-4:] in {f[0] for f in self.formats}:
            self.filename = self.filename[:-4] + self.formats[self.format_index][0]
            self.save.setText("Save as '%s'" % os.path.split(self.filename)[1])

    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.saveAs.setDisabled(data is None)

    def save_file_as(self):
        f = self.formats[self.format_index]
        home_dir = os.path.expanduser("~")
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save Orange Data File',
            self.filename or self.last_dir or home_dir,
            '{} (*{})'.format(f[1], f[0]))
        if not filename:
            return
        self.filename = filename
        if os.path.splitext(filename)[1] != f[0]:
            self.filename += f[0]
        self.last_dir, file_name = os.path.split(self.filename)
        self.save.setText("Save as '%s'" % file_name)
        self.save.setDisabled(False)
        self.save_file()

    def save_file(self):
        if not self.filename:
            self.save_file_as()
        elif self.data is not None:
            try:
                self.formats[self.format_index][2](self.filename, self.data)
                self.error()
            except Exception as errValue:
                self.error(str(errValue))


if __name__ == "__main__":
    import sys
    a = QtGui.QApplication(sys.argv)
    data = Table("iris")
    ow = OWSave()
    ow.show()
    ow.dataset(data)
    a.exec()
    ow.saveSettings()
