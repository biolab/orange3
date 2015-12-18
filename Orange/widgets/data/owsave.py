import os.path

from PyQt4 import QtGui

from Orange.data.table import Table
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.data.io import FileFormat


class OWSave(widget.OWWidget):
    name = "Save Data"
    description = "Save data to an output file."
    icon = "icons/Save.svg"
    author = "Martin Frlin"
    category = "Data"
    keywords = ["data", "save"]

    inputs = [("Data", Table, "dataset")]

    want_main_area = False
    resizing_enabled = False

    last_dir = Setting("")
    auto_save = Setting(True)

    def __init__(self, data=None, file_formats=None):
        super().__init__()
        self.data = None
        self.filename = ""
        self.format_index = 0
        self.file_formats = file_formats or FileFormat.writers
        self.formats = [(f.DESCRIPTION, f.EXTENSIONS)
                        for f in sorted(set(self.file_formats.values()),
                                        key=lambda f: f.OWSAVE_PRIORITY)]

        self.info = gui.widgetLabel(self.controlArea, 'Output filename not '
                                                      'specified.')
        self.comboBoxFormat = gui.comboBox(
            self.controlArea, self, value='format_index',
            items=['{} (*{})'.format(x[0], ' *'.join(x[1]))
                   for x in self.formats],
            box='File Format')

        box = gui.widgetBox(self.controlArea)
        self.save = gui.auto_commit(self.controlArea, self, "auto_save",
                                    "Save",
                                    commit=self.save_file)
#        self.save = gui.button(box, self, "Save", callback=self.save_file,
#                               default=True, disabled=True)
        gui.separator(box)
        self.saveAs = gui.button(box, self, "Save as ...",
                                 callback=self.save_file_as, disabled=True)
        self.setMinimumWidth(320)
        self.adjustSize()
        if data:
            self.dataset(data)

    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.saveAs.setDisabled(data is None)
        self.save_file()

    def save_file_as(self):
        format_name, format_extensions = self.formats[self.format_index]
        home_dir = os.path.expanduser("~")
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save as ...',
            self.filename or os.path.join((self.last_dir or home_dir), getattr(self.data, 'name', '')),
            '{} (*{})'.format(format_name, ' *'.join(format_extensions)))
        if not filename:
            return
        for ext in format_extensions:
            if filename.endswith(ext):
                break
        else:
            filename += format_extensions[0]
        self.filename = filename
        self.last_dir, file_name = os.path.split(self.filename)
        self.info.setText("Save as '%s'" %  file_name)
        self.save.button.setText("Save as '%s'" % file_name)
        self.save.setDisabled(False)
        self.save_file()

    def save_file(self):
        if not self.filename:
            self.save_file_as()
        elif self.data is not None:
            try:
                ext = self.formats[self.format_index][1]
                if not isinstance(ext, str):
                    ext = ext[0]  # is e.g. a tuple of extensions
                self.file_formats[ext].write(self.filename, self.data)
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
