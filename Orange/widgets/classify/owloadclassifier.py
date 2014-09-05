import os
import pickle

from PyQt4 import QtGui
import Orange.classification

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWLoadClassifier(widget.OWWidget):
    name = "Load Classifier"
    description = "Load a classifier from disk."
    priority = 3050
    icon = "icons/LoadClassifier.svg"

    outputs = [
        {
            "name": "Classifier",
            "type": Orange.classification.Model,
            "flags": widget.Dynamic
        }
    ]
    #: List of recent filenames.
    history = Setting([])
    #: Current (last selected) filename or None.
    filename = Setting(None)

    FILTER = "Pickle files (*.pickle *.pck)\nAll files (*.*)"

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selectedIndex = -1

        box = gui.widgetBox(
            self.controlArea, self.tr("File"), orientation=QtGui.QHBoxLayout()
        )

        self.filesCB = gui.comboBox(
            box, self, "selectedIndex", callback=self._on_recent)
        self.filesCB.setMinimumContentsLength(20)

        self.loadbutton = gui.button(box, self, "...", callback=self.browse)
        self.loadbutton.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        self.loadbutton.setSizePolicy(QtGui.QSizePolicy.Maximum,
                                      QtGui.QSizePolicy.Fixed)

        self.reloadbutton = gui.button(
            box, self, "Reload", callback=self.reload, default=True)
        self.reloadbutton.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
        self.reloadbutton.setSizePolicy(QtGui.QSizePolicy.Maximum,
                                        QtGui.QSizePolicy.Fixed)

        # filter valid existing filenames
        self.history = list(filter(os.path.isfile, self.history))[:20]
        for filename in self.history:
            self.filesCB.addItem(os.path.basename(filename), userData=filename)

        # restore the current selection if the filename is
        # in the history list
        if self.filename in self.history:
            self.selectedIndex = self.history.index(self.filename)
        else:
            self.selectedIndex = -1
            self.filename = None
            self.reloadbutton.setEnabled(False)

    def browse(self):
        """Select a filename using an open file dialog."""
        if self.filename is None:
            startdir = QtGui.QDesktopServices.storageLocation(
                QtGui.QDesktopServices.DocumentsLocation)
        else:
            startdir = os.path.dirname(self.filename)

        filename = QtGui.QFileDialog.getOpenFileName(
            self, self.tr("Open"), directory=startdir, filter=self.FILTER)

        if filename:
            self.load(filename)

    def reload(self):
        """Reload the current file."""
        self.load(self.filename)

    def load(self, filename):
        """Load the object from filename and send it to output."""
        try:
            classifier = pickle.load(open(filename, "rb"))
        except pickle.UnpicklingError:
            raise  # TODO: error reporting
        except os.error:
            raise  # TODO: error reporting
        else:
            self._remember(filename)
            self.send("Classifier", classifier)

    def _remember(self, filename):
        """
        Remember `filename` was accessed.
        """
        if filename in self.history:
            index = self.history.index(filename)
            del self.history[index]
            self.filesCB.removeItem(index)

        self.history.insert(0, filename)

        self.filesCB.insertItem(0, os.path.basename(filename),
                                userData=filename)
        self.selectedIndex = 0
        self.filename = filename
        self.reloadbutton.setEnabled(self.selectedIndex != -1)

    def _on_recent(self):
        self.load(self.history[self.selectedIndex])


def main():
    app = QtGui.QApplication([])
    w = OWLoadClassifier()
    w.show()
    return app.exec_()

if __name__ == "__main__":
    import sys
    sys.exit(main())
