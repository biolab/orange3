import re, os.path

from PyQt4 import QtCore
from PyQt4 import QtGui

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table

class OWSave(widget.OWWidget):
    _name = "Save"
    _descritpion = "Saves data to a file."
    _icon = "icons/Save.svg"
    _author = "Martin Frlin"
    _category = "Data"
    _keywords = ["data", "save"]

    inputs = [("Data", Table, "dataset")]

    want_main_area = False

    recentFiles = Setting([])
    selectedFileName = Setting("None")

    savers = {".txt": orange.saveTxt, ".tab": orange.saveTabDelimited,
              ".names": orange.saveC45, ".test": orange.saveC45, ".data": orange.saveC45,
              ".csv": orange.saveCsv
              }

    # exclude C50 since it has the same extension and we do not need saving to it anyway
    registeredFileTypes = [ft for ft in orange.getRegisteredFileTypes() if len(ft)>3 and ft[3] and not ft[0]=="C50"]

    dlgFormats = 'Tab-delimited files (*.tab)\nHeaderless tab-delimited (*.txt)\nComma separated (*.csv)\nC4.5 files (*.data)\nRetis files (*.rda *.rdo)\n' \
                 + "\n".join("%s (%s)" % (ft[:2]) for ft in registeredFileTypes) \
                 + "\nAll files(*.*)"

    savers.update(dict((ft[1][1:].lower(), ft[3]) for ft in registeredFileTypes))
    re_filterExtension = re.compile(r"\(\*(?P<ext>\.[^ )]+)")

    def __init__(self,parent=None, signalManager = None, settings=None):
        super().__init__(self, parent, signalManager, settings, "Save", resizingEnabled = 0)


        self.data = None
        self.filename = ""

#        vb = OWGUI.widgetBox(self.controlArea)

        rfbox = gui.widgetBox(self.controlArea, "Filename", orientation="horizontal", addSpace=True)
        self.filecombo = gui.comboBox(rfbox, self, "filename")
        self.filecombo.setMinimumWidth(200)
#        browse = OWGUI.button(rfbox, self, "...", callback = self.browseFile, width=25)
        button = gui.button(rfbox, self, '...', callback = self.browseFile, disabled=0)
        button.setIcon(self.style().standardIcon(QtCore.QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QtCore.QSizePolicy.Maximum, QtCore.QSizePolicy.Fixed)

        fbox = gui.widgetBox(self.controlArea, "Save")
        self.save = gui.button(fbox, self, "Save current data", callback = self.saveFile, default=True)
        self.save.setDisabled(1)

        gui.rubber(self.controlArea)

        #self.adjustSize()
        self.setFilelist()
        self.resize(260,100)
        self.filecombo.setCurrentIndex(0)

        if self.selectedFileName != "":
            if os.path.exists(self.selectedFileName):
                self.openFile(self.selectedFileName)
            else:
                self.selectedFileName = ""


    def dataset(self, data):
        self.data = data
        self.save.setDisabled(data == None)

    def browseFile(self):
        if self.recentFiles:
            startfile = self.recentFiles[0]
        else:
            startfile = os.path.expanduser("~/")

#        filename, selectedFilter = QFileDialog.getSaveFileNameAndFilter(self, 'Save Orange Data File', startfile,
#                        self.dlgFormats, self.dlgFormats.splitlines()[0])
#        filename = str(filename)
#       The preceding lines should work as per API, but do not; it's probably a PyQt bug as per March 2010.
#       The following is a workaround.
#       (As a consequence, filter selection is not taken into account when appending a default extension.)
        filename, selectedFilter = QtGui.QFileDialog.getSaveFileName(self, 'Save Orange Data File', startfile,
                         self.dlgFormats), self.dlgFormats.splitlines()[0]
        filename = filename # unicode() was here
        if not filename or not os.path.split(filename)[1]:
            return

        ext = os.path.splitext(filename)[1].lower()
        if not ext in self.savers:
            filt_ext = self.re_filterExtension.search(str(selectedFilter)).group("ext")
            if filt_ext == ".*":
                filt_ext = ".tab"
            filename += filt_ext


        self.addFileToList(filename)
        self.saveFile()

    def saveFile(self, *index):
        self.error()
        if self.data is not None:
            combotext = self.filecombo.currentText() # unicode() was here
            if combotext == "(none)":
                QtGui.QMessageBox.information( None, "Error saving data", "Unable to save data. Select first a file name by clicking the '...' button.", QMessageBox.Ok + QMessageBox.Default)
                return
            filename = self.recentFiles[self.filecombo.currentIndex()]
            fileExt = os.path.splitext(filename)[1].lower()
            if fileExt == "":
                fileExt = ".tab"
            try:
                self.savers[fileExt](filename, self.data)
            except Exception as errValue:
                self.error(str(errValue))
                return
            self.error()

    def addFileToList(self,fn):
        if fn in self.recentFiles:
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0,fn)
        self.setFilelist()

    def setFilelist(self):
        "Set the GUI filelist"
        self.filecombo.clear()

        if self.recentFiles:
            self.filecombo.addItems([os.path.split(file)[1] for file in self.recentFiles])
        else:
            self.filecombo.addItem("(none)")


if __name__ == "__main__":
    import sys
    a = QtGui.QApplication(sys.argv)
    owf = OWSave()
    owf.show()
    a.exec_()
    owf.saveSettings()
