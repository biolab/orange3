import os
import pickle

from PyQt4 import QtGui

import Orange.classification

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting


class OWSaveClassifier(widget.OWWidget):
    name = "Save Classifier"
    description = "Save a trained classifier to an output file."
    icon = "icons/SaveClassifier.svg"
    priority = 3000

    inputs = [("Classifer", Orange.classification.Model, "setModel")]

    #: Current (last selected) filename or None.
    filename = Setting(None)
    #: A list of recent filenames.
    history = Setting([])

    FILTER = "Pickle files (*.pickle *.pck)\nAll files (*.*)"

    want_main_area = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selectedIndex = -1
        #: input model/classifier
        self.model = None

        box = gui.widgetBox(self.controlArea, self.tr("File"),
                            orientation=QtGui.QHBoxLayout())
        self.filesCB = gui.comboBox(box, self, "selectedIndex",
                                    callback=self._on_recent)
        self.filesCB.setMinimumContentsLength(20)

        button = gui.button(
            box, self, "...", callback=self.browse, default=True
        )
        button.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon)
        )
        button.setSizePolicy(QtGui.QSizePolicy.Maximum,
                             QtGui.QSizePolicy.Fixed)

        self.savebutton = gui.button(
            self.controlArea, self, "Save", callback=self.savecurrent,
            default=True
        )
        self.savebutton.setEnabled(False)

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

    def setModel(self, model):
        """Set input classifier."""
        self.model = model
        self.savebutton.setEnabled(
            not (model is None or self.filename is None))

    def save(self, filename):
        """Save the model to filename (model must not be None)."""
        assert self.model is not None
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.model, f)
        except pickle.PicklingError:
            raise
        except os.error:
            raise
        else:
            self._remember(filename)

    def savecurrent(self):
        """
        Save the model to current selected filename.

        Do nothing if model or current filename are None.

        """
        if self.model is not None and self.filename is not None:
            self.save(self.filename)

    def browse(self):
        """Select a filename using a Save file dialog."""
        if self.filename is None:
            startdir = QtGui.QDesktopServices.storageLocation(
                QtGui.QDesktopServices.DocumentsLocation
            )
        else:
            startdir = os.path.dirname(self.filename)

        filename = QtGui.QFileDialog.getSaveFileName(
            self, self.tr("Save"), directory=startdir, filter=self.FILTER
        )
        if filename:
            if self.model is not None:
                self.save(filename)
            else:
                self._remember(filename)

    def _on_recent(self):
        filename = self.history[self.selectedIndex]
        self._remember(filename)

    def _remember(self, filename):
        if filename in self.history:
            index = self.history.index(filename)
            del self.history[index]
            self.filesCB.removeItem(index)

        self.history.insert(0, filename)
        self.filesCB.insertItem(0, os.path.basename(filename),
                                userData=filename)

        self.filename = filename
        self.selectedIndex = 0
        self.savebutton.setEnabled(
            not (self.model is None or self.filename is None))


def main():
    app = QtGui.QApplication([])
    w = OWSaveClassifier()
    w.show()
    return app.exec_()

if __name__ == "__main__":
    import sys
    sys.exit(main())

