import os.path
import re

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

    last_dir = Setting("")

    savers = {".tab": save_tab_delimited, ".csv": save_csv}
    dlgFormats = 'Tab-delimited files (*.tab)\nComma separated (*.csv)'
    re_filterExtension = re.compile(r"\(\*(?P<ext>\.[^ )]+)")

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(self, parent, signalManager, settings, "Save")
        self.data = None
        self.filename = ""
        box = gui.widgetBox(self.controlArea)
        self.save = gui.button(box, self, "Save", callback=self.saveFile,
                               default=True, disabled=True)
        gui.separator(box)
        self.saveAs = gui.button(box, self, "Save as ...",
                                 callback=self.saveFileAs, disabled=True)
        self.setMinimumWidth(320)
        self.adjustSize()

    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data is None)
        self.saveAs.setDisabled(data is None)

    def saveFileAs(self):
        home_dir = os.path.expanduser("~")
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save Orange Data File',
            self.filename or self.last_dir or home_dir,
            self.dlgFormats)
        if not filename:
            return
        self.filename = filename
        self.last_dir, file_name = os.path.split(filename)
        self.save.setText("Save as '%s'" % file_name)
        self.save.setDisabled(False)
        self.saveFile()

    def saveFile(self):
        if not self.filename:
            self.saveFileAs()
        else:
            self.error()
            if self.data is not None:
                file_ext = os.path.splitext(self.filename)[1].lower() or ".tab"
                try:

                    self.savers[file_ext](self.filename, self.data)
                except Exception as errValue:
                    self.error(str(errValue))
                    return
                self.error()


if __name__ == "__main__":
    import sys
    a = QtGui.QApplication(sys.argv)
    data = Table("iris")
    ow = OWSave()
    ow.show()
    ow.dataset(data)
    a.exec()
    ow.saveSettings()
